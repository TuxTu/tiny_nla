#!/usr/bin/env python3
"""Generate verified (buggy, correct) program pairs by AST mutation of MBPP.

Why mutate instead of scraping a real bug corpus: mutation gives you three
things real bugs don't.
  1. a bug-class label, so the per-class delta / z-clustering probe has ground truth
  2. a known location
  3. a target that is canonical BY CONSTRUCTION — the inverse of the mutation is
     *the* minimal fix, so there is no arbitrary choice among equivalent repairs

Bug classes mirror bigcode/humanevalpack's `bug_type` taxonomy exactly, so
mutation-trained models can be evaluated on HumanEvalPack's hand-written bugs
without a train/eval distribution shift.

Every mutant is EXECUTED: the original must pass its tests and the mutant must
fail them.  A syntactic mutation is not necessarily a semantic bug, and an
ineffective mutant is a mislabelled training pair.  Timeouts count as failures
(mutation genuinely produces infinite loops — HumanEvalPack sees them too).

  python scripts/build_mutation_pairs.py --out data/repair/mbpp_mutants.parquet
"""

import argparse
import ast
import copy
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

# humanevalpack bug_type vocabulary
VALUE, OPERATOR, VARIABLE, FUNCTION, MISSING, EXCESS = (
    "value misuse", "operator misuse", "variable misuse",
    "function misuse", "missing logic", "excess logic",
)

# bigcode/humanevalpack python bug_type distribution (44/33/31/25/23/8 of 164)
HEP_PROPS = {VALUE: 44 / 164, MISSING: 33 / 164, EXCESS: 31 / 164,
             OPERATOR: 25 / 164, VARIABLE: 23 / 164, FUNCTION: 8 / 164}

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add,
             ast.FloorDiv: ast.Div, ast.Div: ast.FloorDiv}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}
_FUNC_SWAP = {"max": "min", "min": "max", "sorted": "reversed", "reversed": "sorted",
              "any": "all", "all": "any", "upper": "lower", "lower": "upper",
              "abs": "int", "sum": "max", "int": "float", "len": "sum",
              "list": "set", "set": "list", "tuple": "list", "str": "repr",
              "round": "int", "float": "int",
              "split": "rsplit", "rsplit": "split", "find": "rfind",
              "rfind": "find", "index": "rindex", "strip": "lstrip",
              "lstrip": "rstrip", "rstrip": "lstrip",
              "startswith": "endswith", "endswith": "startswith",
              "sort": "reverse", "reverse": "sort", "keys": "values",
              "values": "keys", "isupper": "islower", "islower": "isupper",
              "append": "remove", "extend": "append", "pop": "remove",
              "ceil": "floor", "floor": "ceil"}
_UNWRAPPABLE = {"abs", "int", "float", "sorted", "list", "set", "str", "len", "round"}


def _sites(tree):
    """Yield (bug_type, path_index, mutate_fn) for every applicable mutation site.

    mutate_fn takes a fresh deep copy of the tree and applies one edit in place.
    """
    nodes = list(ast.walk(tree))
    names = sorted({n.id for n in nodes
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})

    for i, node in enumerate(nodes):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and type(node.ops[0]) in _CMP_SWAP:
            new = _CMP_SWAP[type(node.ops[0])]
            yield OPERATOR, i, lambda t, i=i, new=new: (
                setattr(list(ast.walk(t))[i], "ops", [new()]))

        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
            new = _BIN_SWAP[type(node.op)]
            yield OPERATOR, i, lambda t, i=i, new=new: (
                setattr(list(ast.walk(t))[i], "op", new()))

        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAP:
            new = _BOOL_SWAP[type(node.op)]
            yield OPERATOR, i, lambda t, i=i, new=new: (
                setattr(list(ast.walk(t))[i], "op", new()))

        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                yield VALUE, i, lambda t, i=i, v=v: (
                    setattr(list(ast.walk(t))[i], "value", not v))
            elif isinstance(v, int):
                nv = 1 if v == 0 else (0 if v == 1 else v + 1)
                yield VALUE, i, lambda t, i=i, nv=nv: (
                    setattr(list(ast.walk(t))[i], "value", nv))
            elif isinstance(v, float):
                yield VALUE, i, lambda t, i=i, nv=v + 1.0: (
                    setattr(list(ast.walk(t))[i], "value", nv))

        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and len(names) > 1:
            other = [n for n in names if n != node.id]
            if other:
                pick = other[hash(node.id + str(i)) % len(other)]
                yield VARIABLE, i, lambda t, i=i, pick=pick: (
                    setattr(list(ast.walk(t))[i], "id", pick))

        elif isinstance(node, ast.Call):
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None)
            if fname in _FUNC_SWAP:
                new = _FUNC_SWAP[fname]
                attr = "id" if isinstance(f, ast.Name) else "attr"
                yield FUNCTION, i, lambda t, i=i, new=new, attr=attr: (
                    setattr(list(ast.walk(t))[i].func, attr, new))
            # missing logic: drop a single-argument wrapper, abs(x) -> x
            if fname in _UNWRAPPABLE and len(node.args) == 1 and not node.keywords:
                yield MISSING, i, lambda t, i=i: _replace(t, i, list(ast.walk(t))[i].args[0])

        if isinstance(node, ast.BoolOp) and len(node.values) > 1:
            yield MISSING, i, lambda t, i=i: _replace(t, i, list(ast.walk(t))[i].values[0])

        # missing logic: drop a negation / an operand / a chained comparison
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield MISSING, i, lambda t, i=i: _replace(t, i, list(ast.walk(t))[i].operand)

        if isinstance(node, ast.BinOp):
            yield MISSING, i, lambda t, i=i: _replace(t, i, list(ast.walk(t))[i].left)

        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            yield MISSING, i, lambda t, i=i: (
                setattr(list(ast.walk(t))[i], "ops", list(ast.walk(t))[i].ops[:1]),
                setattr(list(ast.walk(t))[i], "comparators",
                        list(ast.walk(t))[i].comparators[:1]))

        if isinstance(node, ast.Return) and node.value is not None:
            yield EXCESS, i, lambda t, i=i: _wrap_add(list(ast.walk(t))[i])

        # excess logic: negate a guard / add a redundant term to a subscript
        if isinstance(node, ast.If):
            yield EXCESS, i, lambda t, i=i: _negate(list(ast.walk(t))[i])

        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name):
            yield EXCESS, i, lambda t, i=i: _wrap_add_slice(list(ast.walk(t))[i])


def _replace(tree, idx, new_node):
    """Replace nodes[idx] with new_node by finding its parent."""
    target = list(ast.walk(tree))[idx]
    for parent in ast.walk(tree):
        for field, val in ast.iter_fields(parent):
            if val is target:
                setattr(parent, field, new_node)
                return
            if isinstance(val, list):
                for k, item in enumerate(val):
                    if item is target:
                        val[k] = new_node
                        return


def _wrap_add(ret_node):
    ret_node.value = ast.BinOp(left=ret_node.value, op=ast.Add(),
                               right=ast.Constant(value=1))


def _negate(if_node):
    if_node.test = ast.UnaryOp(op=ast.Not(), operand=if_node.test)


def _wrap_add_slice(sub_node):
    sub_node.slice = ast.BinOp(left=sub_node.slice, op=ast.Add(),
                               right=ast.Constant(value=1))


def canonical(src):
    """Normalise formatting so buggy and correct differ ONLY by the mutation."""
    return ast.unparse(ast.parse(src))


def run_tests(code, setup, tests, timeout=4.0):
    prog = "\n".join([code, setup or "", *tests])
    try:
        r = subprocess.run([sys.executable, "-"], input=prog, text=True,
                           capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False          # infinite loop == failing, which is a real bug class
    except Exception:
        return False


def located_diff(buggy, correct):
    """Return (line_idx, buggy_line, correct_line) if exactly one line differs."""
    b, c = buggy.split("\n"), correct.split("\n")
    if len(b) != len(c):
        return None
    diff = [i for i, (x, y) in enumerate(zip(b, c)) if x != y]
    if len(diff) != 1:
        return None
    i = diff[0]
    return i, b[i], c[i]


def process(row, max_per_problem, seed):
    code, setup = row["code"], row.get("test_setup_code") or ""
    tests = list(row["test_list"])
    try:
        correct = canonical(code)
    except SyntaxError:
        return []
    if not run_tests(correct, setup, tests):
        return []                      # original must pass, else the pair is junk

    tree = ast.parse(correct)
    rng = random.Random(seed + int(row["task_id"]))

    # Draw sites round-robin across bug types rather than uniformly. Name nodes
    # vastly outnumber every other site, so a uniform shuffle yields ~58%
    # "variable misuse" against HumanEvalPack's 14% -- a train/eval shift.
    by_type = {}
    for bug_type, idx, fn in _sites(tree):
        by_type.setdefault(bug_type, []).append((bug_type, idx, fn))
    for v in by_type.values():
        rng.shuffle(v)
    order = sorted(by_type, key=lambda k: -HEP_PROPS.get(k, 0))
    sites = []
    while any(by_type.values()):
        for k in order:
            if by_type.get(k):
                sites.append(by_type[k].pop())

    out, seen = [], set()
    for bug_type, idx, fn in sites:
        if len(out) >= max_per_problem:
            break
        t = copy.deepcopy(tree)
        try:
            fn(t)
            buggy = ast.unparse(t)
        except Exception:
            continue
        if buggy == correct or buggy in seen:
            continue
        seen.add(buggy)
        loc = located_diff(buggy, correct)
        if loc is None:
            continue                   # keep pairs single-line so the target is clean
        if run_tests(buggy, setup, tests):
            continue                   # mutation was semantically inert
        line_idx, buggy_line, correct_line = loc
        out.append({
            "task_id": int(row["task_id"]),
            "spec": row["text"],
            "correct_code": correct,
            "buggy_code": buggy,
            "bug_type": bug_type,
            "line_idx": line_idx,
            "buggy_line": buggy_line,
            "replacement_line": correct_line,
            "target_located": f"line {line_idx}: {correct_line.strip()}",
            "tests": "\n".join(tests),
            "test_setup": setup,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mbpp", default=str(PROJ / "data/repair/mbpp_all.parquet"))
    p.add_argument("--out", default=str(PROJ / "data/repair/mbpp_mutants.parquet"))
    p.add_argument("--max-per-problem", type=int, default=24)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--balance", default="cap", choices=["exact", "cap", "none"],
                   help="exact: match HumanEvalPack proportions (costly -- bound "
                        "by the scarcest class). cap: keep everything but cap any "
                        "single class at --max-class-frac. none: raw.")
    p.add_argument("--max-class-frac", type=float, default=0.25)
    args = p.parse_args()

    import pandas as pd
    from multiprocessing import Pool

    df = pd.read_parquet(args.mbpp)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    print(f"{len(rows)} MBPP problems, up to {args.max_per_problem} mutants each")

    with Pool(args.workers) as pool:
        chunks = pool.starmap(
            process, [(r, args.max_per_problem, args.seed) for r in rows], chunksize=4)

    pairs = [x for c in chunks for x in c]
    if not pairs:
        print("ERROR: no verified pairs produced")
        sys.exit(1)

    out = pd.DataFrame(pairs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw_path = str(Path(args.out).with_name(Path(args.out).stem + "_raw.parquet"))
    out.to_parquet(raw_path, index=False)
    print(f"wrote {raw_path}  ({len(out)} raw pairs, {out.task_id.nunique()} programs)\n")

    if args.balance == "exact":
        # Subsample to HumanEvalPack proportions. The binding class is whichever
        # is scarcest relative to its target share.
        counts = Counter(out.bug_type)
        print("raw (pre-balance) distribution:")
        for k, v in counts.most_common():
            print(f"  {k:<18} {v:>6}  {v/len(out):>5.1%}   "
                  f"supports n_total={int(v / HEP_PROPS[k]):>6}")
        n_total = min(int(counts[k] / HEP_PROPS[k]) for k in HEP_PROPS if counts.get(k))
        keep = []
        rng = random.Random(args.seed)
        for k, prop in HEP_PROPS.items():
            idx = out.index[out.bug_type == k].tolist()
            rng.shuffle(idx)
            keep += idx[: int(round(n_total * prop))]
        print(f"balance=exact: {len(out)} -> {len(keep)} pairs "
              f"(bound by scarcest class relative to its target share)")
        out = out.loc[sorted(keep)].reset_index(drop=True)

    elif args.balance == "cap":
        # Keep everything, but stop any one class from dominating. Per-class
        # metrics are invariant to the mix anyway, so exact matching is not
        # worth discarding most of the corpus for.
        rng = random.Random(args.seed)
        keep = []
        for k in Counter(out.bug_type):
            idx = out.index[out.bug_type == k].tolist()
            rng.shuffle(idx)
            keep += idx
        n = len(out)
        cap = None
        for _ in range(50):                       # cap is relative to the kept total
            tot = 0
            for k in Counter(out.bug_type):
                c = int(out.bug_type.eq(k).sum())
                tot += min(c, int(n * args.max_class_frac)) if cap is None else min(c, cap)
            if cap is not None and abs(tot - n) < 2:
                break
            n = tot
            cap = int(n * args.max_class_frac)
        keep = []
        for k in Counter(out.bug_type):
            idx = out.index[out.bug_type == k].tolist()
            rng.shuffle(idx)
            keep += idx[:cap]
        print(f"balance=cap: {len(out)} -> {len(keep)} pairs "
              f"(no class above {args.max_class_frac:.0%})")
        out = out.loc[sorted(keep)].reset_index(drop=True)

    out.to_parquet(args.out, index=False)

    print(f"\n{len(out)} verified pairs from {out.task_id.nunique()} problems "
          f"({len(out)/max(out.task_id.nunique(),1):.1f} per problem)")
    print("\nbug_type distribution:")
    for k, v in Counter(out.bug_type).most_common():
        print(f"  {k:<18} {v:>6}  {v/len(out):>5.1%}")
    nl = out.correct_code.str.count("\n") + 1
    print(f"\nprogram lines: mean={nl.mean():.1f} median={nl.median():.0f} p90={nl.quantile(.9):.0f}")
    print(f"wrote {args.out}")

    sample = str(Path(args.out).with_suffix(".sample.jsonl"))
    with open(sample, "w") as f:
        for r in out.groupby("bug_type").head(2).to_dict("records"):
            f.write(json.dumps(r) + "\n")
    print(f"wrote {sample}")


if __name__ == "__main__":
    main()

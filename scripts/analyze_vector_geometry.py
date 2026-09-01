#!/usr/bin/env python3
"""Geometry of the activation vectors: is there structure a classifier can use?

Answers the measurements the framework listed as open gates:
  - participation ratio -> effective rank r. The Gaussian channel estimate says
    the actor's realised 92 bits equals capacity at r ~= 133; r >> 133 means
    decoder headroom, r ~= 133 means only more taps help.
  - cos(E(buggy), E(correct)) -> HARD GATE. If ~1.00 the bug is invisible in
    the vector and every repair architecture downstream is dead.
  - is delta = v_correct - v_buggy consistent within a bug class? That is the
    linear-representation hypothesis for "bugginess", which §03 says has no
    theoretical support here and must be measured.
  - can a linear probe read bug_type off the vector at all?

No sklearn in this env, so the probe is closed-form one-vs-rest ridge on a PCA
basis -- weaker than a tuned classifier, so treat its accuracy as a floor.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJ = Path(__file__).resolve().parent.parent


def load_vecs(path, limit=None):
    # Slice BEFORE materialising: the CSN file is 153k x 4096 = 2.5 GB and we
    # usually only want 20k rows of it.
    if limit:
        pf = pq.ParquetFile(path); batches=[]; got=0
        for b in pf.iter_batches(batch_size=4096, columns=["activation_vector"]):
            batches.append(b); got += b.num_rows
            if got >= limit: break
        t = pa.Table.from_batches(batches)
    else:
        t = pq.read_table(path, columns=["activation_vector"])
    col = t.column("activation_vector").combine_chunks()
    d = col.type.list_size if hasattr(col.type, "list_size") else len(col[0])
    v = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, d)
    return v[:limit] if limit else v


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)


def _eigvals(Xc):
    """Eigenvalues of the covariance. The Gram trick (n x n) is only cheaper when
    n << d; at n=20k, d=4096 it means eigendecomposing a 20000x20000 matrix,
    which takes hours. Pick whichever side is smaller."""
    n, d = Xc.shape
    C = (Xc.T @ Xc) / n if n >= d else (Xc @ Xc.T) / n
    return np.linalg.eigvalsh(C)[::-1].clip(0)


def spectrum(X, name):
    Xc = X - X.mean(0, keepdims=True)
    ev = _eigvals(Xc)
    pr = ev.sum() ** 2 / (ev ** 2).sum()          # participation ratio
    tot = ev.sum()
    c = np.cumsum(ev) / tot
    k90 = int(np.searchsorted(c, 0.90) + 1)
    k99 = int(np.searchsorted(c, 0.99) + 1)
    print(f"  {name:<22} PR(eff rank)={pr:7.1f}   dims for 90% var={k90:5d}   99%={k99:5d}")
    return pr


def anisotropy(X, name, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    U = unit(X[idx])
    S = U @ U.T
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"  {name:<22} mean pairwise cos={off.mean():+.4f}  std={off.std():.4f}  "
          f"mean norm={np.linalg.norm(X, axis=1).mean():.1f}")


def ridge_probe(X, y, n_class, lam=1e2, seed=0, folds=4):
    """Closed-form one-vs-rest ridge on PCA features; returns mean CV accuracy."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    accs = []
    for f in range(folds):
        te = np.zeros(len(X), bool); te[f::folds] = True
        Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
        mu, sd = Xtr.mean(0), Xtr.std(0).clip(1e-6)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        Y = np.eye(n_class)[ytr]
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
        accs.append((np.argmax(Xte @ W, 1) == yte).mean())
    return float(np.mean(accs))


def pca_basis(X, k=256):
    """Top-k principal directions via the d x d covariance (see _eigvals)."""
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / len(Xc)
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-csn", type=int, default=20000)
    p.add_argument("--n-text", type=int, default=20000)
    p.add_argument("--pca-dim", type=int, default=256)
    args = p.parse_args()

    D = PROJ / "data/coderec"
    print("=" * 78); print("1. NORMS AND ANISOTROPY"); print("=" * 78)
    csn = load_vecs(D / "vec_csn_code.parquet", args.n_csn)
    bug = load_vecs(D / "vec_buggy.parquet")
    cor = load_vecs(D / "vec_mbpp_code.parquet")
    txt = load_vecs(PROJ / "data/split50/av_sft_train.parquet", args.n_text)
    for X, n in [(csn, "code (CSN)"), (bug, "code (MBPP buggy)"), (cor, "code (MBPP correct)"), (txt, "text (UltraFineWeb)")]:
        anisotropy(X, n)

    print(); print("=" * 78); print("2. EFFECTIVE RANK"); print("=" * 78)
    for X, n in [(csn, "code (CSN)"), (bug, "code (MBPP buggy)"), (txt, "text (UltraFineWeb)")]:
        spectrum(X, n)
    print("  [framework: realised 92 bits == Gaussian capacity at r ~= 133]")

    print(); print("=" * 78); print("3. HARD GATE: cos(E(buggy), E(correct))"); print("=" * 78)
    pairs = pq.read_table(D / "repair_pairs.parquet", columns=["task_id", "bug_type"]).to_pandas()
    progs = pq.read_table(D / "mbpp_programs.parquet", columns=["task_id"]).to_pandas()
    pos = {t: i for i, t in enumerate(progs.task_id)}
    ok = pairs.task_id.map(lambda t: t in pos).values
    ci = pairs.task_id[ok].map(pos).values
    vb, vc = bug[ok], cor[ci]
    cs = (unit(vb) * unit(vc)).sum(1)
    print(f"  n={len(cs)}   mean={cs.mean():.4f}  median={np.median(cs):.4f}  "
          f"p5={np.percentile(cs,5):.4f}  p95={np.percentile(cs,95):.4f}")
    rng = np.random.default_rng(0)
    ctrl = (unit(vb) * unit(vc[rng.permutation(len(vc))])).sum(1)
    print(f"  control (mismatched pairs): mean={ctrl.mean():.4f}")
    print(f"  -> separation {cs.mean()-ctrl.mean():+.4f}  (if ~0, the bug is invisible)")

    print(); print("=" * 78); print("4. IS delta = v_correct - v_buggy CONSISTENT?"); print("=" * 78)
    delta = vc - vb
    bt = pairs.bug_type[ok].values
    classes = sorted(set(bt)); cidx = {c: i for i, c in enumerate(classes)}
    U = unit(delta)
    for c in classes:
        m = bt == c
        if m.sum() < 20: continue
        sub = U[m][:400]
        S = sub @ sub.T
        within = S[~np.eye(len(S), dtype=bool)].mean()
        other = U[~m][:400]
        across = (sub @ other.T).mean()
        print(f"  {c:<18} n={m.sum():5d}  within-class cos={within:+.4f}   across={across:+.4f}")
    Sall = U[:800] @ U[:800].T
    print(f"  {'ALL':<18} n={len(U):5d}  global mean cos={Sall[~np.eye(len(Sall),dtype=bool)].mean():+.4f}")

    print(); print("=" * 78); print("5. LINEAR PROBE: can bug_type be read off the vector?"); print("=" * 78)
    y = np.array([cidx[c] for c in bt])
    chance = np.bincount(y).max() / len(y)
    B = pca_basis(np.concatenate([vb, delta]), args.pca_dim)
    for X, name in [(vb, "v_buggy"), (delta, "delta = v_cor - v_bug"), (vc, "v_correct")]:
        acc = ridge_probe((X - X.mean(0)) @ B, y, len(classes))
        print(f"  {name:<24} 4-fold acc={acc:.3f}   (majority-class baseline {chance:.3f})")

    print(); print("=" * 78); print("6. CODE vs TEXT SEPARABILITY"); print("=" * 78)
    n = min(len(csn), len(txt), 8000)
    X = np.concatenate([csn[:n], txt[:n]]); y2 = np.array([0]*n + [1]*n)
    B2 = pca_basis(X, args.pca_dim)
    print(f"  code vs text 4-fold acc={ridge_probe((X-X.mean(0))@B2, y2, 2):.3f}  (chance 0.500)")


if __name__ == "__main__":
    main()

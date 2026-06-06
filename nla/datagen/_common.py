"""YAML config loading for pipeline scripts.

Mirrors the original NLA's approach: each script accepts a --config YAML file
whose keys map to CLI argument names. CLI args override config values.

Usage in a script:
    import argparse
    from nla.config import add_config_arg, apply_config

    p = argparse.ArgumentParser(...)
    add_config_arg(p)
    # ... add other args ...
    args = apply_config(p)  # replaces parse_args()
"""

import argparse
import sys
import yaml


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file, returning a flat dict of arg-name → value.

    Nested keys are flattened: corpus.name → corpus_name, stage0.foo → foo, etc.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"config must be a top-level mapping, got {type(cfg).__name__}"
    flat: dict[str, object] = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_key == "name":
                    flat[key] = sub_val           # corpus.name → corpus
                else:
                    flat[f"{key}_{sub_key}"] = sub_val  # corpus.config → corpus_config
        else:
            flat[key] = val
    return flat


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --config argument to a parser."""
    parser.add_argument(
        "--config", default=None,
        help="YAML config file. Keys map to --arg names. CLI args override config values.",
    )


def apply_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args, applying YAML config defaults if --config is given.

    Sniffs raw argv for --config before argparse validation, loads the YAML,
    injects its values as defaults and clears `required` on args satisfied by
    the config. CLI args always override config values.
    """
    # Sniff --config from raw argv before argparse validates
    raw_argv = sys.argv[1:]
    config_path = None
    for i, a in enumerate(raw_argv):
        if a == "--config":
            if i + 1 < len(raw_argv):
                config_path = raw_argv[i + 1]
            break
        elif a.startswith("--config="):
            config_path = a.split("=", 1)[1]
            break

    if config_path:
        cfg = load_yaml_config(config_path)
        parser.set_defaults(**{k: v for k, v in cfg.items()})
        # Clear `required` on any action satisfied by the config, so
        # parse_args doesn't reject the config-provided value
        cfg_dests = set(cfg.keys())
        for action in parser._actions:
            if action.dest in cfg_dests:
                action.required = False

    return parser.parse_args()

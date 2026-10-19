"""YAML config loading with `key.sub=value` command-line overrides."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODALITIES = ("vision", "mag", "grav")


def load(path="configs/base.yaml", overrides=()):
    with open(ROOT / path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    for m in cfg["modalities"]:
        assert m in MODALITIES, f"unknown modality {m}"
    return cfg


def path(cfg, key):
    return ROOT / cfg[key]

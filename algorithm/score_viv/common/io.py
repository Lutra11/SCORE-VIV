"""Configuration loading, CLI overrides, and structured output writers."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import yaml


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return cfg


def _coerce(text: str):
    """Interpret an override value: number, bool, list, else string."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def set_nested(cfg: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        nxt = node.get(p) if isinstance(node, dict) else None
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def merge_overrides(cfg: dict, overrides) -> dict:
    """Apply 'a.b.c=value' CLI overrides in place; returns cfg."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value: {item}")
        key, _, val = item.partition("=")
        set_nested(cfg, key.strip(), _coerce(val.strip()))
    return cfg


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def write_csv(path, rows) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(v) for k, v in r.items()})


def _cell(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    return v


def fmt(v, digits=6):
    """Format scalars for markdown tables."""
    if v is None:
        return "PENDING"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def write_markdown(path, title, sections) -> None:
    """sections: list of (heading, [table-or-lines]). Tables: list of dict rows."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        if isinstance(body, str):
            lines.append(body)
            lines.append("")
            continue
        rows = list(body)
        if not rows:
            lines.append("_(none)_")
            lines.append("")
            continue
        cols = []
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")

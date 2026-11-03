"""Common infrastructure: configuration, IO, seeds, hashes, run metadata."""

from .io import (
    ensure_dir,
    fmt,
    load_config,
    merge_overrides,
    set_nested,
    write_csv,
    write_json,
    write_markdown,
)
from .runtime import (
    PACKAGE,
    PACKAGE_VERSION,
    git_revision,
    input_hashes,
    make_rng,
    now_iso,
    run_metadata,
    seed_everything,
)

__all__ = [
    "PACKAGE", "PACKAGE_VERSION",
    "load_config", "merge_overrides", "set_nested",
    "ensure_dir", "write_json", "write_csv", "write_markdown", "fmt",
    "make_rng", "seed_everything", "input_hashes", "git_revision",
    "now_iso", "run_metadata",
]

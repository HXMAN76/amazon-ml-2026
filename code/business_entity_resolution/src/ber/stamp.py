"""Stage cache stamps for the Makefile.

`python -m ber.stamp <stage> [--sections=a,b] [file ...]` hashes the stage's params section plus the given source files
and rewrites WORK/.stamps/<stage>.hash only when the hash changed. Make depends on that file, so a
stage reruns exactly when its params or code changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from ber import config


def digest(sections: list[str], files: list[str]) -> str:
    """Hash of the given params sections and source files."""
    h = hashlib.sha256()
    cfg = config.load()
    for sec in sections:
        h.update(json.dumps(cfg.get(sec, {}), sort_keys=True).encode())
    for f in sorted(files):
        h.update(f.encode())
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    """CLI: refresh a stage stamp file only when its params or sources changed."""
    stage, rest = sys.argv[1], sys.argv[2:]
    sections = [stage]
    if rest and rest[0].startswith("--sections="):  # extra params sections this stage depends on
        sections = rest[0].split("=", 1)[1].split(",")
        rest = rest[1:]
    files = rest
    stamp = config.paths()["stamps"] / f"{stage}.hash"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    new = digest(sections, files)
    if not stamp.exists() or stamp.read_text() != new:
        stamp.write_text(new)


if __name__ == "__main__":
    main()

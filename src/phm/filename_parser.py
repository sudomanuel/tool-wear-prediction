"""
filename_parser.py — parsea {A|R}{exp_id}_p{n}.txt
"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_PATTERN = re.compile(r'^(?P<dir>[AR])(?P<exp_id>\d+)_p(?P<contact_id>\d+)$',
                      re.IGNORECASE)


@dataclass
class SegmentMeta:
    filename: str
    direction_code: str   # 'A' o 'R'
    experiment_id: int
    contact_id: int


def parse_segment_name(filename: str) -> Optional[SegmentMeta]:
    """Devuelve metadata si el nombre coincide con el patron, sino None."""
    stem = Path(filename).stem
    m = _PATTERN.match(stem)
    if m is None:
        return None
    return SegmentMeta(
        filename=filename,
        direction_code=m.group('dir').upper(),
        experiment_id=int(m.group('exp_id')),
        contact_id=int(m.group('contact_id')),
    )


def scan_segments(directory: Path) -> dict:
    """Devuelve {experiment_id: {(dir, contact_id): Path}}."""
    directory = Path(directory)
    out: dict = {}
    if not directory.exists():
        return out
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() != '.txt':
            continue
        meta = parse_segment_name(f.name)
        if meta is None:
            continue
        out.setdefault(meta.experiment_id, {})[(meta.direction_code, meta.contact_id)] = f
    return out

"""Extract one archived persecond session day from a yearly tar into a temp dir."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path


def persecond_tar_path(archive_root: Path, kind: str, year: str) -> Path:
    return archive_root / f"persecond-{kind}-{year}.tar.gz"


def live_persecond_day_dir(data_root: Path, kind: str, day: str) -> Path:
    return data_root / "persecond" / kind / day


class PersecondDayResolver:
    """Resolve a persecond RDS path, extracting one day from archive if needed."""

    def __init__(
        self,
        data_root: Path,
        archive_root: Path | None = None,
        extract_root: Path | None = None,
    ):
        self.data_root = Path(data_root)
        self.archive_root = Path(archive_root) if archive_root is not None else self.data_root / "archive"
        self._extract_root = Path(extract_root) if extract_root is not None else None
        self._owned_tmps: list[Path] = []
        self._extract_parent_dir: Path | None = None
        self._day_dirs: dict[tuple[str, str], Path] = {}

    def resolve(self, path: str) -> str:
        source = Path(path)
        if source.exists():
            return path
        try:
            day = source.parent.name
            kind = source.parent.parent.name
            periodicity = source.parent.parent.parent.name
        except Exception:
            return path
        if periodicity != "persecond":
            return path
        day_dir = self.ensure_day(kind, day)
        if day_dir is None:
            return path
        return str(day_dir / source.name)

    def ensure_day(self, kind: str, day: str) -> Path | None:
        key = (kind, day)
        if key in self._day_dirs:
            return self._day_dirs[key]
        live = live_persecond_day_dir(self.data_root, kind, day)
        if live.is_dir():
            self._day_dirs[key] = live
            return live
        year = day[:4]
        tar_path = persecond_tar_path(self.archive_root, kind, year)
        if not tar_path.is_file():
            return None
        dest_parent = self._shared_extract_parent()
        dest_parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["tar", "-xzf", str(tar_path), "-C", str(dest_parent), day],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        extracted = dest_parent / day
        if not extracted.is_dir():
            return None
        self._day_dirs[key] = extracted
        return extracted

    def _shared_extract_parent(self) -> Path:
        if self._extract_parent_dir is not None:
            return self._extract_parent_dir
        if self._extract_root is not None:
            parent = self._extract_root / str(uuid.uuid4())
        else:
            parent = Path(tempfile.gettempdir()) / "rds-untar" / str(os.getpid()) / str(uuid.uuid4())
        self._extract_parent_dir = parent
        self._owned_tmps.append(parent)
        return parent

    def cleanup(self) -> None:
        for tmp in self._owned_tmps:
            shutil.rmtree(tmp, ignore_errors=True)
        self._owned_tmps.clear()
        self._extract_parent_dir = None
        self._day_dirs.clear()

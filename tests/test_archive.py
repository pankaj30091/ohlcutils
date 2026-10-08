import hashlib
import subprocess
from pathlib import Path

from ohlcutils.archive import PersecondDayResolver, persecond_tar_path


def _make_day(root: Path, kind: str, day: str, names: list[str]) -> Path:
    day_dir = root / "persecond" / kind / day
    day_dir.mkdir(parents=True)
    for name in names:
        (day_dir / name).write_text("payload-" + name, encoding="utf-8")
    return day_dir


def _tar_year(data_root: Path, archive_root: Path, kind: str, year: str, days: list[str]) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = persecond_tar_path(archive_root, kind, year)
    parent = data_root / "persecond" / kind
    subprocess.run(["tar", "-czf", str(dest), "-C", str(parent), *days], check=True)
    return dest


def test_live_day_needs_no_tar(tmp_path: Path):
    day = "2018-01-02"
    _make_day(tmp_path, "stk", day, ["AAA_STK____2018-01-02.rds"])
    resolver = PersecondDayResolver(tmp_path, archive_root=tmp_path / "archive")
    try:
        path = tmp_path / "persecond" / "stk" / day / "AAA_STK____2018-01-02.rds"
        assert resolver.resolve(str(path)) == str(path)
        assert resolver._owned_tmps == []
    finally:
        resolver.cleanup()


def test_extract_one_day_then_cleanup(tmp_path: Path):
    day = "2018-01-02"
    other = "2018-01-03"
    _make_day(tmp_path, "stk", day, ["AAA_STK____2018-01-02.rds"])
    _make_day(tmp_path, "stk", other, ["BBB_STK____2018-01-03.rds"])
    archive_root = tmp_path / "archive"
    _tar_year(tmp_path, archive_root, "stk", "2018", [day, other])
    live_day = tmp_path / "persecond" / "stk" / day
    other_dir = tmp_path / "persecond" / "stk" / other
    import shutil

    shutil.rmtree(live_day)
    shutil.rmtree(other_dir)

    missing = tmp_path / "persecond" / "stk" / day / "AAA_STK____2018-01-02.rds"
    assert not missing.exists()
    extract_root = tmp_path / "extracts"
    resolver = PersecondDayResolver(tmp_path, archive_root=archive_root, extract_root=extract_root)
    try:
        resolved = Path(resolver.resolve(str(missing)))
        assert resolved.exists()
        assert resolved.read_text(encoding="utf-8") == "payload-AAA_STK____2018-01-02.rds"
        assert resolved.parent.name == day
        other_extracted = extract_root.rglob("BBB_STK____2018-01-03.rds")
        assert list(other_extracted) == []
        owned = list(resolver._owned_tmps)
        assert owned
    finally:
        resolver.cleanup()
    for owned in owned:
        assert not owned.exists()
    assert not missing.exists()


def test_missing_tar_returns_original_path(tmp_path: Path):
    path = tmp_path / "persecond" / "stk" / "2018-01-02" / "AAA_STK____2018-01-02.rds"
    resolver = PersecondDayResolver(tmp_path, archive_root=tmp_path / "archive")
    try:
        assert resolver.resolve(str(path)) == str(path)
    finally:
        resolver.cleanup()


def test_tar_member_checksum_stable(tmp_path: Path):
    day = "2014-04-28"
    _make_day(tmp_path, "ind", day, ["NIFTY_IND____2014-04-28.rds"])
    archive_root = tmp_path / "archive"
    dest = _tar_year(tmp_path, archive_root, "ind", "2014", [day])
    listing = subprocess.check_output(["tar", "-tzf", str(dest)], text=True)
    digest = hashlib.sha256("".join(sorted(listing.splitlines(True))).encode()).hexdigest()
    listing2 = subprocess.check_output(["tar", "-tzf", str(dest)], text=True)
    digest2 = hashlib.sha256("".join(sorted(listing2.splitlines(True))).encode()).hexdigest()
    assert digest == digest2
    top = sorted({line.split("/")[0] for line in listing.splitlines() if line.strip()})
    assert top == [day]

from pathlib import Path
import os
import pytest

from mbe_tools.jsonl_selector import select_jsonl


def test_select_jsonl_prefers_run_then_parsed(tmp_path: Path):
    run = tmp_path / "run.jsonl"
    parsed = tmp_path / "parsed.jsonl"
    other = tmp_path / "other.jsonl"

    parsed.write_text("{}\n", encoding="utf-8")
    run.write_text("{}\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")

    # run.jsonl preferred
    assert select_jsonl(None, cwd=str(tmp_path)) == str(run)

    # remove run -> parsed chosen
    run.unlink()
    assert select_jsonl(None, cwd=str(tmp_path)) == str(parsed)

    # remove parsed -> newest *.jsonl chosen among others
    parsed.unlink()
    # set mtimes: make other older, then create newest
    os.utime(other, (other.stat().st_atime, other.stat().st_mtime - 100))
    newest = tmp_path / "newest.jsonl"
    newest.write_text("{}\n", encoding="utf-8")
    assert select_jsonl(None, cwd=str(tmp_path)) == str(newest)


def test_select_jsonl_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        select_jsonl(None, cwd=str(tmp_path))

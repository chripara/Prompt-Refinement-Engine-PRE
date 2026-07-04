"""US-PRE-E00-S02 — the derived/explicit seed is recorded in the decision log."""
import json

from pre.core.logging import decision_log


def test_record_appends_jsonl_entry_with_seed(tmp_path, monkeypatch):
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setenv(decision_log.LOG_PATH_ENV, str(log_file))

    decision_log.record(stage="seed", seed=12345, explicit=False, checkpoint="sdxl-base-1.0")

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["stage"] == "seed"
    assert entry["seed"] == 12345
    assert entry["explicit"] is False
    assert "ts" in entry


def test_record_appends_multiple_entries(tmp_path, monkeypatch):
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setenv(decision_log.LOG_PATH_ENV, str(log_file))

    decision_log.record(stage="seed", seed=1)
    decision_log.record(stage="seed", seed=2)

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

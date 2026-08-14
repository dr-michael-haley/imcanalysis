from __future__ import annotations

import json
from pathlib import Path

from SpatialBiologyToolkit.napari_sbt.storage import append_audit, experiment_paths


def test_append_audit_adds_jsonl_events_without_replacing_previous_lines(
    tmp_path: Path,
):
    paths = experiment_paths(tmp_path / "experiment")

    append_audit(paths, {"action": "first"})
    append_audit(paths, {"action": "second"})

    events = [
        json.loads(line)
        for line in paths.label_audit.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["action"] for event in events] == ["first", "second"]
    assert all("timestamp" in event for event in events)

import json

import pytest

from staggerlab import cli
from staggerlab.study import run_study


def test_study_artifacts_and_acceptance(tmp_path):
    report = run_study(tmp_path)
    assert report["all_passed"]
    assert len(report["checks"]) == 10
    assert (
        json.loads((tmp_path / "validation.json").read_text())["metrics"] == report["metrics"]
    )
    for name in ("spatial", "temporal", "fields", "energy", "pressure_spectrum"):
        assert len((tmp_path / f"{name}.csv").read_text().splitlines()) >= 5
    assert (tmp_path / "staggerlab_audit.png").stat().st_size > 30_000


@pytest.mark.parametrize("passed,code", [(True, 0), (False, 1)])
def test_cli_propagates_acceptance(monkeypatch, capsys, passed, code):
    monkeypatch.setattr(cli, "run_study", lambda output: {"all_passed": passed, "checks": {}})
    assert cli.main(["--output", "ignored"]) == code
    assert json.loads(capsys.readouterr().out)["all_passed"] == passed

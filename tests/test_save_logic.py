from types import SimpleNamespace

from utils.io_utils import create_unique_experiment_dir
from utils.logging_utils import is_artifact_save_iter


def test_artifacts_are_saved_every_thousand_iterations():
    args = SimpleNamespace(artifact_save_interval=1000)

    assert not is_artifact_save_iter(0, args)
    assert not is_artifact_save_iter(998, args)
    assert is_artifact_save_iter(999, args)
    assert is_artifact_save_iter(1999, args)


def test_artifact_saves_can_be_disabled():
    args = SimpleNamespace(artifact_save_interval=0)

    assert not is_artifact_save_iter(0, args)
    assert not is_artifact_save_iter(999, args)


def test_unique_experiment_directory_uses_incrementing_suffix(tmp_path):
    first = create_unique_experiment_dir(tmp_path, "experiment")
    second = create_unique_experiment_dir(tmp_path, "experiment")
    third = create_unique_experiment_dir(tmp_path, "experiment")

    assert first == str(tmp_path / "experiment")
    assert second == str(tmp_path / "experiment_1")
    assert third == str(tmp_path / "experiment_2")

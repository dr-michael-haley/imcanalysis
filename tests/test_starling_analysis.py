from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from SpatialBiologyToolkit.config.models import StarlingConfig
from SpatialBiologyToolkit.scripts.starling_analysis import _train


def _run_training(tmp_path: Path, *, logging_enabled: bool, logger_cls: Mock):
    model = Mock()
    starling = SimpleNamespace(ST=Mock(return_value=model))
    result = _train(
        Mock(),
        cfg=StarlingConfig(tensorboard_logging=logging_enabled),
        qc_dir=tmp_path,
        cell_size_col="area",
        starling_module=starling,
        early_stopping_cls=Mock(),
        tensorboard_logger_cls=logger_cls,
    )
    return result, model


def test_missing_logging_backend_fails_before_model_creation(tmp_path):
    error = ModuleNotFoundError("Neither tensorboard nor tensorboardX is available")
    starling = SimpleNamespace(ST=Mock())
    with pytest.raises(ModuleNotFoundError, match="starling.tensorboard_logging: false") as caught:
        _train(
            Mock(),
            cfg=StarlingConfig(),
            qc_dir=tmp_path,
            cell_size_col=None,
            starling_module=starling,
            early_stopping_cls=Mock(),
            tensorboard_logger_cls=Mock(side_effect=error),
        )
    assert "python -m pip install tensorboard" in str(caught.value)
    assert caught.value.__cause__ is error
    starling.ST.assert_not_called()


def test_disabled_logging_trains_without_backend(tmp_path):
    logger_cls = Mock(side_effect=ModuleNotFoundError("No logging backend"))
    result, model = _run_training(tmp_path, logging_enabled=False, logger_cls=logger_cls)
    assert result is model
    logger_cls.assert_not_called()
    model.train_and_fit.assert_called_once()
    assert model.train_and_fit.call_args.kwargs["logger"] is False


def test_enabled_logging_passes_logger_to_training(tmp_path):
    logger_cls = Mock()
    result, model = _run_training(tmp_path, logging_enabled=True, logger_cls=logger_cls)
    assert result is model
    logger_cls.assert_called_once_with(save_dir=str(tmp_path / "lightning_logs"), name="starling")
    model.train_and_fit.assert_called_once()
    assert model.train_and_fit.call_args.kwargs["logger"] is logger_cls.return_value

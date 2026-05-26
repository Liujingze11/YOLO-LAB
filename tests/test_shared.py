"""Unit tests for shared core library."""
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from shared.config import TrainConfig
from shared.train_logger import (
    append_full_val_log, append_train_log, ensure_train_csv_header,
    extract_seg_val_metrics, get_timestamp,
)
from shared.train_core import (
    build_train_kwargs, list_experiments, get_val_metrics,
)


class TestTrainConfig:
    def test_defaults(self):
        c = TrainConfig()
        assert c.epochs == 150
        assert c.imgsz == 640
        assert c.batch == 16
        assert c.use_augment is True
        assert c.device == "cpu" or c.device == "0"

    def test_save_dir(self):
        c = TrainConfig(results_dir="/tmp/results", experiment_name="exp1")
        assert c.save_dir == "/tmp/results/exp1"

    def test_pt_paths(self):
        c = TrainConfig(results_dir="/tmp/results", experiment_name="test")
        assert c.best_pt == "/tmp/results/test/weights/best.pt"
        assert c.last_pt == "/tmp/results/test/weights/last.pt"

    def test_custom_values(self):
        c = TrainConfig(epochs=50, imgsz=320, batch=4, device="cpu")
        assert c.epochs == 50
        assert c.imgsz == 320
        assert c.batch == 4
        assert c.device == "cpu"


class TestBuildTrainKwargs:
    def test_minimal_kwargs(self, tmp_path):
        c = TrainConfig(
            data_yaml=str(tmp_path / "data.yaml"),
            results_dir=str(tmp_path / "results"),
            experiment_name="test",
        )
        kwargs = build_train_kwargs(c, use_augment=False)
        assert kwargs["data"] == str(tmp_path / "data.yaml")
        assert kwargs["epochs"] == 150
        assert kwargs["imgsz"] == 640
        assert kwargs["batch"] == 16
        assert "hsv_h" not in kwargs

    def test_with_augmentation(self, tmp_path):
        c = TrainConfig(
            data_yaml=str(tmp_path / "data.yaml"),
            results_dir=str(tmp_path / "results"),
            experiment_name="test",
        )
        kwargs = build_train_kwargs(c, use_augment=True)
        assert "hsv_h" in kwargs
        assert kwargs["hsv_h"] == 0.015
        assert kwargs["mosaic"] == 1.0


class TestListExperiments:
    def test_empty_dir(self, tmp_path):
        assert list_experiments(str(tmp_path)) == []

    def test_with_dirs(self, tmp_path):
        (tmp_path / "exp2").mkdir()
        (tmp_path / "exp1").mkdir()
        (tmp_path / "file.txt").write_text("hello")
        result = list_experiments(str(tmp_path))
        assert result == ["exp1", "exp2"]

    def test_nonexistent_dir(self):
        assert list_experiments("/nonexistent/path") == []


class TestTrainLogger:
    def test_train_csv_header(self, tmp_path):
        csv_path = str(tmp_path / "train_log.csv")
        ensure_train_csv_header(csv_path)
        with open(csv_path, encoding="utf-8-sig") as f:
            header = f.readline().strip()
        assert "time" in header
        assert "mode" in header
        assert "experiment_name" in header

    def test_append_train_log(self, tmp_path):
        c = TrainConfig(
            log_dir=str(tmp_path),
            results_dir=str(tmp_path / "results"),
            experiment_name="test_exp",
            data_yaml="/tmp/data.yaml",
            model_file="/tmp/model.pt",
        )
        append_train_log(c, mode="new_train", status="started",
                         notes="test run")
        csv_path = str(tmp_path / "train_log.csv")
        assert os.path.exists(csv_path)
        with open(csv_path, encoding="utf-8-sig") as f:
            lines = f.readlines()
        assert len(lines) >= 2  # header + data row
        assert "test_exp" in lines[1]
        assert "new_train" in lines[1]

    def test_get_timestamp(self):
        ts = get_timestamp()
        assert len(ts) >= 16
        assert "-" in ts


class TestMetricsExtraction:
    def test_extract_empty(self):
        summary, per_class = extract_seg_val_metrics(
            _FakeMetrics(), {}, {}
        )
        assert "box_p" in summary
        assert per_class == []


class _FakeMetrics:
    """Minimal fake to test extract_seg_val_metrics."""
    names = {}

    @staticmethod
    def mean_results():
        return [0.5, 0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.3]

    @staticmethod
    def class_result(idx):
        return []

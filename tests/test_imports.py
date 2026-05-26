"""Smoke test: verify all modules import correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# GUI needs its dir too
sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))


def test_shared_config():
    from shared.config import TrainConfig
    assert TrainConfig is not None


def test_shared_train_logger():
    from shared.train_logger import append_train_log, append_full_val_log
    assert append_train_log is not None
    assert append_full_val_log is not None


def test_shared_train_core():
    from shared.train_core import build_train_kwargs, list_experiments
    assert build_train_kwargs is not None
    assert list_experiments is not None


def test_gui_utils():
    from gui.utils import engine_cmd, log_append, load_presets
    assert engine_cmd is not None


def test_gui_tabs():
    from gui.tabs.train_tab import TrainTab
    from gui.tabs.infer_tab import InferTab
    from gui.tabs.tools_tab import ToolsTab
    from gui.tabs.log_viewer_tab import LogViewerTab
    assert TrainTab is not None
    assert InferTab is not None


def test_gui_workers():
    from gui.workers import TrainWorker, InferWorker, ToolWorker
    assert TrainWorker is not None


def test_gui_gpu_manager():
    from gui.gpu_manager import check_gpu_capability, is_cuda_available
    result = check_gpu_capability()
    assert "status" in result


def test_cli_imports():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli" / "scripts"))
    from config import TrainConfig
    from shared.train_logger import append_train_log
    assert TrainConfig is not None

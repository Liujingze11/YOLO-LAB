"""Unified TrainConfig shared by GUI and CLI."""
from dataclasses import dataclass, field
import os


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


@dataclass
class TrainConfig:
    data_yaml: str = ""
    model_file: str = ""
    results_dir: str = ""
    log_dir: str = ""

    epochs: int = 150
    imgsz: int = 640
    batch: int = 16
    device: str = field(default_factory=_default_device)

    experiment_name: str = "experiment"

    # data augmentation
    use_augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    @property
    def save_dir(self) -> str:
        return os.path.join(self.results_dir, self.experiment_name)

    @property
    def last_pt(self) -> str:
        return os.path.join(self.save_dir, "weights", "last.pt")

    @property
    def best_pt(self) -> str:
        return os.path.join(self.save_dir, "weights", "best.pt")

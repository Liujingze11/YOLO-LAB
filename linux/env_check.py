"""Pre-launch env check."""
import subprocess
import sys
import shutil

REQUIRED = [
    ("PySide6", "PySide6"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("yaml", "pyyaml"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
]


def missing_packages():
    result = []
    for imp_name, pip_name in REQUIRED:
        try:
            __import__(imp_name)
        except ImportError:
            result.append((imp_name, pip_name))
    return result


def run_checks_and_fix():
    """Check environment; return False if user chose to exit."""
    missing = missing_packages()
    if not missing:
        return True
    print("[YOLO-LAB] 检测到缺失依赖:")
    for imp, pip in missing:
        print(f"  - {imp} (pip install {pip})")
    print(f"\n运行: pip install {' '.join(p for _, p in missing)}")
    return True

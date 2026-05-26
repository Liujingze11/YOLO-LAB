"""Pre-launch env check — must work WITHOUT any third-party packages."""
import subprocess
import sys
import shutil

# Packages required for the GUI to start
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


def _native_dialog(title, message, question=False):
    """Show a native OS dialog. Returns True if user clicked Yes/OK."""
    try:
        if sys.platform == "linux":
            for cmd, args in [
                (["zenity", "--question", "--title", title, "--text", message, "--width=500"], False),
                (["kdialog", "--title", title, "--yesno", message], False),
            ]:
                if shutil.which(cmd[0]):
                    return subprocess.run(cmd).returncode == 0
        elif sys.platform == "darwin":
            buttons = '{"取消", "安装"}' if question else '{"OK"}'
            default = '"安装"' if question else '"OK"'
            r = subprocess.run([
                "osascript", "-e",
                f'display dialog "{message}" with title "{title}" buttons {buttons} default button {default}'
            ], capture_output=True, text=True)
            return "安装" in r.stdout if question else True
        elif sys.platform == "win32":
            import ctypes
            flags = 0x24 if question else 0x40  # YesNo | Info
            return ctypes.windll.user32.MessageBoxW(0, message, title, flags) == 6
    except Exception:
        pass

    # Fallback: print to console
    print(f"\n{'='*50}\n{title}\n{'='*50}\n{message}\n")
    return False


def _native_info(title, message):
    _native_dialog(title, message, question=False)


def run_checks_and_fix():
    """Returns True if env is ready to proceed, False if restart needed."""
    missing = missing_packages()
    if not missing:
        return True

    pkg_list = [pip for _, pip in missing]
    names = [imp for imp, _ in missing]
    msg = "YOLO Lab 检测到缺少以下组件：\n\n"
    msg += "\n".join(f"  • {n}" for n in names)
    msg += "\n\n是否自动安装？\n\n（安装完成后请重新启动应用）"

    if _native_dialog("环境检查", msg, question=True):
        cmd = [sys.executable, "-m", "pip", "install"] + pkg_list
        try:
            subprocess.run(cmd, check=True)
            _native_info("环境检查", "安装完成！请重新启动应用。")
        except Exception:
            _native_info("环境检查",
                        f"安装失败。请手动运行：\n\npip install {' '.join(pkg_list)}")
    return False

"""YOLO-LAB unified entry point — no-arg → GUI, subcommand → CLI."""
import sys
from pathlib import Path

# Ensure the linux/ directory is on sys.path for all imports
LINUX_ROOT = Path(__file__).resolve().parent
if str(LINUX_ROOT) not in sys.path:
    sys.path.insert(0, str(LINUX_ROOT))


def main():
    if len(sys.argv) > 1:
        # Has subcommand → CLI mode
        from cli import run_cli
        run_cli(sys.argv[1:])
    else:
        # No arguments → GUI mode
        from gui.main_window import run_gui
        run_gui()


if __name__ == "__main__":
    main()

"""Re-export TrainConfig from shared core library."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import TrainConfig

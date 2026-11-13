import sys
from pathlib import Path

ABSA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABSA_DIR / "scripts"))

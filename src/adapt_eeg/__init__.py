# root level config 
from pathlib import Path

## path to data
DATA_ROOT = Path(__file__).parents[2] / "data"


if __name__ == "__main__":
    print(f"DATA_ROOT: {DATA_ROOT.resolve()}")
    
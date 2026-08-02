from pathlib import Path
from types import SimpleNamespace

from adapt_eeg.compute_raw_itpc import main

args_map = {
    "data_root": Path("data/raw/datasets_4Hz"),
    "output_dir": Path("results/datasets_4Hz"),
    "window_duration": 1.5,
    "window_shift": 0.5,
}

args = SimpleNamespace(**args_map) 

main(args)

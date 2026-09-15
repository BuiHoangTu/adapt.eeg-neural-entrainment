"""Audit how the legacy EEGLAB line files were divided into epochs."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATA_ROOT = Path("data/raw/itpc-30event-good")


def line_number(path: Path) -> int:
    return int(path.parent.name.removeprefix("line"))


def load_fields(path: Path) -> dict:
    return loadmat(
        path,
        simplify_cells=True,
        variable_names=("trials", "pnts", "srate", "xmin", "xmax", "urevent"),
    )


def main() -> None:
    files = sorted(DATA_ROOT.glob("line*/*.set"), key=lambda p: (line_number(p), p.name))
    if not files:
        raise FileNotFoundError(f"No .set files found below {DATA_ROOT}")

    trial_counts: dict[int, set[int]] = {}
    print("line  files       epochs  anchors  epoch_s  anchor_step_s  anchor_span_s")
    for line in sorted({line_number(path) for path in files}):
        line_files = [path for path in files if line_number(path) == line]
        headers = [load_fields(path) for path in line_files]
        trial_counts[line] = {int(header["trials"]) for header in headers}

        representative = headers[0]
        times = np.asarray(
            [float(event["init_time"]) for event in representative["urevent"]]
        )
        steps = np.diff(times)
        epoch_s = int(representative["pnts"]) / float(representative["srate"])
        print(
            f"{line:>4}  {len(line_files):>5}  "
            f"{str(sorted(trial_counts[line])):>12}  {len(times):>7}  "
            f"{epoch_s:>7.4f}  {np.median(steps):>13.6f}  "
            f"{(times[-1] - times[0]):>13.6f}"
        )

    print(f"\nTotal files: {len(files)}")


if __name__ == "__main__":
    main()

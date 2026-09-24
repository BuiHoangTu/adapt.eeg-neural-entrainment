#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available on PATH." >&2
    exit 1
fi

mkdir -p /tmp/adapt-eeg-numba-cache /tmp/adapt-eeg-mpl-cache
export NUMBA_CACHE_DIR=/tmp/adapt-eeg-numba-cache
export MPLCONFIGDIR=/tmp/adapt-eeg-mpl-cache
export PYTHONPATH="$project_root/src"

python_command=(conda run --no-capture-output -n nbm python)

echo "[1/5] Deriving poem rhythm classifications and frequency evidence"
"${python_command[@]}" -m adapt_eeg.exploration.stress_rhythm_evidence

echo "[2/5] Auditing ICA-cleaned EEG annotations"
"${python_command[@]}" \
    -m adapt_eeg.exploration.inspect_ica_final_annotations \
    --allow-errors

echo "[3/5] Computing trigger-aligned four-poem cycle-aligned ITPC"
"${python_command[@]}" -m adapt_eeg.poem_itpc --allow-errors

echo "[4/5] Computing fixed-0.3s paper-window ITPC"
"${python_command[@]}" \
    -m adapt_eeg.poem_itpc_acf \
    --allow-errors

echo "[5/5] Building four-method comparison tables"
"${python_command[@]}" -m adapt_eeg.method_comparison

echo "Pipeline complete."
echo "ITPC outputs: results/poem_itpc"
echo "ACF-windowed ITPC outputs: results/poem_itpc_acf"
echo "Four-method comparison: results/method_comparison/four_method_summary.xlsx"
echo "Exploration outputs: results/exploration"

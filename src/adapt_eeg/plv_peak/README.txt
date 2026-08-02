Fixed EEG-audio PLV analysis

Place these files in adapt_eeg/plv_peak/, replacing the original modules.

Main changes:
- EEG phase is compared against the phase of the 2 Hz audio envelope.
- Invalid/missing audio alignment is excluded rather than anchored at zero.
- Reflected padding reduces filter/Hilbert edge artefacts.
- Default windows are non-overlapping beat intervals; fixed ±250 ms windows remain available.
- Five expected peaks are retained by default.
- Peak position is preserved in summaries.
- A mixed-effects regularity × peak-position model is exported.
- PLV is no longer mislabeled as ITPC.

Run example:
python -m adapt_eeg.plv_peak.run \
  --data-root data/raw/datasets_4Hz \
  --audio-root data/raw/poem1_versuri \
  --output-dir results/plv_peak \
  --window-mode beat_interval

Dependencies include numpy, pandas, scipy, mne, and statsmodels.

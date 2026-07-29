# ADAPT EEG replication report

## Inputs and generated outputs

This report summarizes the replication outputs in `results/replicate/` using the pipeline under `src/adapt_eeg/replicate/`.

Data coverage in the current run:

- 50 participants
- 14 analysis lines, 7 irregular and 7 regular
- 700 participant-line files in `metadata.csv`
- 3,594 included participant-condition-channel ITPC rows after studentized-residual filtering
- no load errors were recorded in `load_errors.csv`

Generated focused plots:

- `results/replicate/figures/rhythm_contrast_2hz.png`
- `results/replicate/figures/rhythm_contrast_4hz.png`

The contrast plotted is `regular mean ITPC - irregular mean ITPC`; negative values mean higher ITPC for irregular lines.

## Paper-style table outputs

The analysis file produced these paper-style tables:

- `paper_table_1_beat_distances.csv`
- `paper_table_2_mixed_anova.csv`
- `paper_table_2_channels_of_interest.csv`
- `paper_table_3_group_differences.csv`

The statistical table in this replication is an approximation to the draft paper's mixed ANOVA table: the rhythm effect is computed from paired participant-level regular-vs-irregular contrasts, and the group effect is computed from participant mean ITPC by language group. It is useful for checking direction and rough significance, but it is not a byte-for-byte recreation of the draft's ANOVA software output.

## Main rhythm result

The strongest finding in the replication is a broad rhythm effect, but its direction is opposite to the paper draft.

Significant rhythm effects at `p < .05`:

- 2 Hz: 16 of 18 channels
- 4 Hz: 14 of 18 channels

Top rhythm effects:

| channel | frequency | F | p | partial eta² |
|---|---:|---:|---:|---:|
| Fp1 | 2 Hz | 25.751 | 0.000006 | 0.349 |
| P7 | 4 Hz | 22.114 | 0.000022 | 0.315 |
| F7 | 4 Hz | 19.241 | 0.000061 | 0.282 |
| P7 | 2 Hz | 18.474 | 0.000084 | 0.278 |
| F3 | 2 Hz | 16.644 | 0.000166 | 0.254 |
| F3 | 4 Hz | 14.850 | 0.000339 | 0.233 |
| Pz | 4 Hz | 14.692 | 0.000368 | 0.234 |
| Fp2 | 2 Hz | 14.305 | 0.000431 | 0.230 |
| P4 | 2 Hz | 14.136 | 0.000454 | 0.224 |
| Pz | 2 Hz | 13.917 | 0.000497 | 0.221 |

## Channels emphasized in the paper draft

The draft paper highlighted Pz, C4, and T7 for the 2 Hz poetic rhythm effect. In our run:

| channel | frequency | irregular mean | regular mean | regular - irregular | rhythm p |
|---|---:|---:|---:|---:|---:|
| C4 | 2 Hz | 0.042077 | 0.038889 | -0.003188 | 0.072004 |
| Pz | 2 Hz | 0.042517 | 0.036409 | -0.006108 | 0.000497 |
| T7 | 2 Hz | 0.041398 | 0.037088 | -0.004310 | 0.010818 |
| C4 | 4 Hz | 0.041983 | 0.037620 | -0.004363 | 0.015030 |
| Pz | 4 Hz | 0.041806 | 0.035616 | -0.006190 | 0.000368 |
| T7 | 4 Hz | 0.041429 | 0.036643 | -0.004786 | 0.005907 |

Important differences from the paper draft:

1. **Direction is reversed.** The paper reports higher entrainment for regular poetic rhythm than irregular poetic rhythm. Our current output has higher ITPC for irregular rhythm at all three highlighted channels.
2. **C4 at 2 Hz is not significant here.** The paper reports C4 as significant at 2 Hz (`p = .024`); our approximate table gives C4 2 Hz `p = .072`.
3. **4 Hz rhythm effects appear significant here.** The paper draft states there was no significant rhythm-type difference at 4 Hz. Our current output has significant 4 Hz rhythm effects at C4, Pz, T7, and many other channels.

These are large substantive differences, not just rounding differences.

## Group effects

The paper draft reports some group effects where Romanian monolinguals/non-English participants entrained more than English-competent participants, especially P8 at 2 Hz and several 4 Hz frontal/central channels.

Our run found **no significant group effects at p < .05** in the approximate table.

Smallest group-comparison p-values:

| channel | frequency | non-English - English mean ITPC | 95% CI low | 95% CI high | p |
|---|---:|---:|---:|---:|---:|
| P8 | 4 Hz | 0.003376 | -0.000673 | 0.007425 | 0.099828 |
| F7 | 4 Hz | 0.002450 | -0.001037 | 0.005937 | 0.162941 |
| T7 | 4 Hz | 0.002087 | -0.001058 | 0.005232 | 0.187831 |
| O1 | 2 Hz | 0.002198 | -0.001144 | 0.005541 | 0.192194 |
| T7 | 2 Hz | 0.002132 | -0.001133 | 0.005398 | 0.194166 |
| C4 | 4 Hz | 0.002475 | -0.001387 | 0.006337 | 0.200853 |

The direction is often consistent with the draft for group differences (non-English > English), but the effects are not statistically significant in this run.

## Interpretation

The replication currently does **not** reproduce the main paper-draft pattern.

The most important mismatch is the rhythm contrast: the draft claims regular > irregular at 2 Hz, while our result is consistently irregular > regular. The second major mismatch is that the draft claims no rhythm-type effect at 4 Hz, while our result shows many significant 4 Hz rhythm effects. The third major mismatch is that draft group effects are not recovered.

Likely things to check next:

1. Confirm whether `regular` and `irregular` line labels match the intended files and whether any label inversion occurred.
2. Confirm that the ITPC computation matches the original extraction code/windowing exactly.
3. Check whether the `.set` files in `data/raw/datasets_4Hz` are already frequency-specific/processed in a way that changes interpretation of the 2 Hz vs 4 Hz analysis.
4. Compare a few participant-channel-line ITPC values against the original/reference outputs, if available.
5. Replace the approximate table code with a true mixed ANOVA implementation if exact statistical reproduction is required.

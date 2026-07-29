# ADAPT EEG replication report

## Experiment summaries

The paper tests whether neural entrainment to poetic rhythm is a biological universal. The core design compares EEG entrainment while participants listen to selected lines from Yeats's *Sailing to Byzantium*.

Experiment design summarized from the paper draft:

- Participants: 50 usable EEG participants after artifact exclusion.
- Groups: Romanian-English speakers vs Romanian monolingual/non-English participants.
- EEG: 19-channel Enobio 10-20 recording, sampled at 500 Hz, line-noise filtered at 50 Hz, cleaned with ICA in EEGLAB.
- Stimulus: recorded recitation of *Sailing to Byzantium*.
- Lines analyzed:
  - regular rhythm: lines 3, 4, 10, 15, 23, 27, 30
  - irregular rhythm: lines 1, 2, 5, 6, 7, 8, 9
- Main EEG measure: inter-trial phase coherence / ITPC at 2 Hz and 4 Hz.
- Main statistical design: mixed ANOVA with rhythm condition as a within-subject factor and language group as a between-subject factor.

The paper draft's key claims are:

1. Regular poetic rhythm should produce stronger 2 Hz entrainment than irregular poetic rhythm.
2. This regular > irregular rhythm effect is reported most clearly at Pz, C4, and T7.
3. Syllabic 4 Hz entrainment should not differ by poetic rhythm condition.
4. Group effects are reported for some channels, with non-English/Romanian monolingual participants sometimes showing stronger entrainment than English-competent participants.


## Replication attempt

This report summarizes the latest replication outputs in `results/replicate/` after aligning ITPC extraction more closely with the paper method: each line-level Raw segment is converted to MNE fixed-length epochs, then ITPC is computed with `epochs.compute_tfr(method="morlet", return_itc=True)`.

### Inputs and generated outputs

Data coverage in the current run:

- 50 participants
- 14 analysis lines, 7 irregular and 7 regular
- 700 participant-line files in `metadata.csv`
- 3,593 included participant-condition-channel ITPC rows after studentized-residual filtering
- no load errors were recorded in `load_errors.csv`

Generated focused plots:

- `results/replicate/figures/aligned_mne_rhythm_contrast_2hz.png`
- `results/replicate/figures/aligned_mne_rhythm_contrast_4hz.png`

The plotted contrast is `regular mean ITPC - irregular mean ITPC`; negative values mean higher ITPC for irregular lines.

### Paper-style table outputs

The analysis file produced these paper-style tables:

- `paper_table_1_beat_distances.csv`
- `paper_table_2_mixed_anova.csv`
- `paper_table_2_channels_of_interest.csv`
- `paper_table_3_group_differences.csv`

The statistical table in this replication is still an approximation to the draft paper's mixed ANOVA table: the rhythm effect is computed from paired participant-level regular-vs-irregular contrasts, and the group effect is computed from participant mean ITPC by language group. It is useful for checking direction and rough significance, but it is not a byte-for-byte recreation of the draft's ANOVA software output.

### Main rhythm result

After aligning extraction to MNE fixed-length epochs + `compute_tfr(..., return_itc=True)`, the main result is materially similar to the previous MNE run: broad rhythm effects are present at both frequencies, but the direction remains opposite to the paper draft.

Significant rhythm effects at `p < .05`:

- 2 Hz: 18 of 18 channels
- 4 Hz: 18 of 18 channels

Top rhythm effects:

| channel | frequency | F | p | partial eta² |
|---|---:|---:|---:|---:|
| F7 | 2 Hz | 35.811 | <0.000001 | 0.422 |
| P7 | 2 Hz | 40.749 | <0.000001 | 0.454 |
| O2 | 2 Hz | 42.797 | <0.000001 | 0.471 |
| O1 | 2 Hz | 37.113 | <0.000001 | 0.436 |
| P4 | 2 Hz | 40.004 | <0.000001 | 0.449 |
| P7 | 4 Hz | 37.241 | <0.000001 | 0.432 |
| P4 | 4 Hz | 34.256 | <0.000001 | 0.411 |
| O2 | 4 Hz | 39.648 | <0.000001 | 0.452 |
| O1 | 4 Hz | 33.852 | <0.000001 | 0.414 |
| Fp1 | 2 Hz | 31.465 | 0.000001 | 0.391 |
| F7 | 4 Hz | 32.600 | 0.000001 | 0.400 |
| T7 | 2 Hz | 29.720 | 0.000002 | 0.378 |

### Channels emphasized in the paper draft

The draft paper highlighted Pz, C4, and T7 for the 2 Hz poetic rhythm effect. In the latest aligned MNE run:

| channel | frequency | irregular mean | regular mean | regular - irregular |
|---|---:|---:|---:|---:|
| C4 | 2 Hz | 0.062921 | 0.056639 | -0.006282 |
| Pz | 2 Hz | 0.063340 | 0.058103 | -0.005237 |
| T7 | 2 Hz | 0.063198 | 0.056301 | -0.006897 |
| C4 | 4 Hz | 0.059710 | 0.053270 | -0.006440 |
| Pz | 4 Hz | 0.060060 | 0.054980 | -0.005080 |
| T7 | 4 Hz | 0.060011 | 0.053299 | -0.006712 |

### Group effects

The paper draft reports group effects where Romanian monolinguals/non-English participants entrained more than English-competent participants, especially P8 at 2 Hz and several 4 Hz frontal/central channels.

After the aligned MNE rerun, there are **no significant group effects at `p < .05`** in the approximate table.

Smallest group-comparison p-values:

| channel | frequency | non-English - English mean ITPC | 95% CI low | 95% CI high | p |
|---|---:|---:|---:|---:|---:|
| F3 | 4 Hz | 0.002161 | -0.000314 | 0.004636 | 0.084889 |
| O2 | 2 Hz | -0.002180 | -0.004763 | 0.000402 | 0.095904 |
| O2 | 4 Hz | -0.002086 | -0.004667 | 0.000494 | 0.110482 |
| F3 | 2 Hz | 0.001841 | -0.000614 | 0.004295 | 0.136482 |
| P4 | 2 Hz | -0.001578 | -0.003920 | 0.000763 | 0.181251 |
| O1 | 2 Hz | 0.001525 | -0.000807 | 0.003857 | 0.194801 |
| P4 | 4 Hz | -0.001421 | -0.003767 | 0.000924 | 0.228736 |
| O1 | 4 Hz | 0.001368 | -0.001027 | 0.003764 | 0.256453 |
| P8 | 4 Hz | 0.001226 | -0.001158 | 0.003611 | 0.305994 |
| F7 | 2 Hz | -0.001147 | -0.003388 | 0.001094 | 0.307434 |

### Change from the previous MNE run

The latest method is closer to the paper implementation style, and it changed some details:

- Included rows changed from 3,596 to 3,593 after outlier filtering.
- Highlighted-channel ITPC values are similar but not identical.
- Rhythm effects remain significant at all 18 channels for both 2 Hz and 4 Hz.
- Group effects changed materially: the earlier MNE run had significant P8 group effects at both 2 Hz and 4 Hz; the aligned run has no significant group effects.

### Summary of differences in our reproduction attempt

The aligned MNE-based replication still does **not** reproduce the main paper-draft pattern.

The major differences are:

1. **Rhythm direction is reversed.** The draft claims regular > irregular at 2 Hz, while the aligned replication is consistently irregular > regular, including at Pz, C4, and T7.
2. **4 Hz rhythm effects are present in the replication.** The draft claims no rhythm-type effect at 4 Hz, but the aligned replication shows significant rhythm effects at every channel for 4 Hz.
3. **Group effects are not recovered.** The draft reports several group effects; the aligned replication has no significant group effects at `p < .05`.
4. **The mismatch is substantive.** The differences are not small rounding or formatting differences. They affect the sign, frequency specificity, and group-effect conclusions.

Likely things to check next:

1. Confirm whether `regular` and `irregular` line labels match the intended files and whether any label inversion occurred.
2. Confirm that the `.set` files in `data/raw/datasets_4Hz` are organized by poem line exactly as assumed by `v*/4hz_datasets/p*.set`.
3. Check whether the `.set` files are already frequency-specific or transformed in a way that changes interpretation of 2 Hz vs 4 Hz MNE ITC.
4. Compare a few participant-channel-line ITPC values against original/reference outputs, if available.
5. Replace the approximate table code with a true mixed ANOVA implementation if exact statistical reproduction is required.

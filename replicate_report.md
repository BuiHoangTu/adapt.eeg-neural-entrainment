# ADAPT EEG replication report

## Experiment summaries

The paper investigates whether neural entrainment to poetic rhythm is a biological universal. The central question is whether listeners show stronger neural synchronization to metrically regular poetic rhythm than to irregular poetic rhythm, even when they do not understand the language of the poem.

The experimental design can be summarized as follows:

- **Participants:** 50 usable EEG participants after artifact exclusion.
- **Groups:** Romanian-English speakers and Romanian monolingual/non-English participants.
- **EEG acquisition:** 19-channel Enobio 10-20 system, sampled at 500 Hz, line-noise filtered at 50 Hz, and cleaned with ICA in EEGLAB.
- **Stimulus:** a recorded recitation of W. B. Yeats's *Sailing to Byzantium*.
- **Lines analyzed:**
  - regular rhythm: lines 3, 4, 10, 15, 23, 27, 30
  - irregular rhythm: lines 1, 2, 5, 6, 7, 8, 9
- **Primary EEG measure:** inter-trial phase coherence (ITPC) at 2 Hz (Poem rhythm) and 4 Hz (speaking rhythm).
- **Statistical design:** rhythm condition as a within-subject factor and language group as a between-subject factor.

The paper's central claims are:

1. Regular poetic rhythm elicits stronger 2 Hz neural entrainment than irregular poetic rhythm.
2. This regular > irregular effect is reported most clearly at Pz, C4, and T7.
3. Syllabic-rate entrainment at 4 Hz should not differ by poetic rhythm condition.
4. Some group effects are reported, with Romanian monolingual/non-English participants sometimes showing stronger entrainment than English-competent participants.

## Replication attempt

The reproduction results are stored in [`results/replicate/`](results/replicate/). The current run contains complete coverage for the selected line set: 50 participants, 14 analysis lines, and 700 participant-line observations. After outlier filtering, 3,593 participant-condition-channel ITPC rows are included in the reproduced analysis.

The principal output files are:

- [Paper-style rhythm/group table](results/replicate/paper_table_2_mixed_anova.csv)
- [Focused Pz/C4/T7 table](results/replicate/paper_table_2_channels_of_interest.csv)
- [Group-difference table](results/replicate/paper_table_3_group_differences.csv)
- [2 Hz rhythm-contrast plot](results/replicate/figures/aligned_mne_rhythm_contrast_2hz.png)
- [4 Hz rhythm-contrast plot](results/replicate/figures/aligned_mne_rhythm_contrast_4hz.png)

In the rhythm-contrast plots and tables, the contrast is defined as **regular mean ITPC - irregular mean ITPC**. Negative values therefore indicate stronger ITPC for irregular lines.

### Difference in rhythm effects

The paper draft reports a rhythm-specific effect at the poetic rhythm rate: regular poetic rhythm produces stronger 2 Hz ITPC than irregular poetic rhythm, especially at Pz, C4, and T7. It also reports that this effect should not appear at the syllabic rhythm rate of 4 Hz.

The reproduction differs on both points. It finds a strong rhythm-related effect, but the effect is reversed: irregular lines show higher mean ITPC than regular lines. The effect is also not limited to 2 Hz; it appears broadly at both tested frequencies.

Significant rhythm effects at `p < .05` in the reproduction are widespread:

- 2 Hz: 18 of 18 channels
- 4 Hz: 18 of 18 channels

Selected strongest reproduced rhythm effects are shown below.

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

Thus, the reproduction gives strong evidence that rhythm condition matters, but it does not support the paper's more specific claim that regular poetic rhythm increases 2 Hz entrainment. Instead, it indicates stronger entrainment for irregular lines and a broad effect at both 2 Hz and 4 Hz.

### Difference at Pz, C4, and T7

The paper emphasizes Pz, C4, and T7 as the main channels supporting the 2 Hz regular > irregular poetic rhythm effect. These channels are therefore the most important direct comparison points.

| channel | frequency | irregular mean | regular mean | regular - irregular |
|---|---:|---:|---:|---:|
| C4 | 2 Hz | 0.062921 | 0.056639 | -0.006282 |
| Pz | 2 Hz | 0.063340 | 0.058103 | -0.005237 |
| T7 | 2 Hz | 0.063198 | 0.056301 | -0.006897 |
| C4 | 4 Hz | 0.059710 | 0.053270 | -0.006440 |
| Pz | 4 Hz | 0.060060 | 0.054980 | -0.005080 |
| T7 | 4 Hz | 0.060011 | 0.053299 | -0.006712 |

At all three highlighted channels, the reproduced direction is opposite to the paper. The paper reports regular > irregular at 2 Hz; the reproduced values show irregular > regular. The same irregular > regular pattern also appears at 4 Hz, where the paper reports no rhythm-condition effect.

### Difference in group effects

The paper reports group differences in some channels, generally in the direction of stronger entrainment for Romanian monolingual/non-English participants. This supports the paper's claim that neural entrainment to poetic rhythm does not depend on semantic understanding of the poem.

The reproduction gives weaker support for this claim. The reproduced group-difference table uses the same contrast direction: non-English/Romanian monolingual participants minus English-competent participants. Positive values therefore indicate stronger mean ITPC in the non-English group. Some reproduced contrasts point in the same direction as the paper, but they are smaller and non-significant; several paper-positive channels reverse sign.

Examples of direct numerical comparison:

| channel | frequency | paper difference | reproduced difference | reproduced p | interpretation |
|---|---:|---:|---:|---:|---|
| P8 | 2 Hz | 0.006 | 0.001049 | 0.379970 | same direction, much weaker evidence |
| C4 | 4 Hz | 0.012 | 0.000211 | 0.873969 | same direction, much weaker evidence |
| F4 | 4 Hz | 0.021 | -0.000149 | 0.898883 | opposite direction, no evidence |
| Fp2 | 4 Hz | 0.013 | -0.000276 | 0.819731 | opposite direction, no evidence |
| Fz | 4 Hz | 0.012 | 0.000360 | 0.825538 | same direction, much weaker evidence |
| C3 | 4 Hz | 0.011 | -0.000167 | 0.914513 | opposite direction, no evidence |
| F3 | 4 Hz | 0.016 | 0.002161 | 0.084889 | same direction, weaker and not significant |

The smallest reproduced group-comparison p-values are:

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

Thus, relative to the paper, the reproduction provides weaker evidence for the claim that non-English participants gain more neural entrainment. It does not reproduce statistically reliable group differences.

### Summary of differences in our reproduction attempt

The reproduction does **not** recover the main inferential pattern described in the paper draft.

The principal differences are:

1. **The rhythm-effect direction is reversed.** The paper reports regular > irregular at 2 Hz, whereas the reproduction shows irregular > regular, including at Pz, C4, and T7.
2. **The 4 Hz rhythm effect is present in the reproduction.** The paper states that rhythm condition should not significantly affect 4 Hz syllabic-rate entrainment. The reproduction instead shows significant rhythm effects at every channel for 4 Hz.
3. **The group effects are not reproduced.** The paper reports several group effects, but the reproduction finds no significant group effects at `p < .05`.
4. **The discrepancy concerns the main claims.** The mismatch is not limited to minor numerical variation; it affects the direction of the main rhythm effect, the frequency specificity of that effect, and the reported group differences.

Overall, the reproduction supports the presence of rhythm-related ITPC differences, but it does not support the paper's specific conclusion that regular poetic rhythm produces stronger 2 Hz entrainment than irregular poetic rhythm. The current results instead suggest stronger entrainment for irregular lines and a broad effect at both 2 Hz and 4 Hz.

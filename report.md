# Regular versus irregular rhythm results


The analysis used CV `< 0.40` for regular lines and CV `> 0.55` for
irregular lines. Values between the thresholds were classified as ambiguous.
The Wilcoxon test is one-sided for the hypothesis that regular ITPC is greater
than irregular ITPC. The Mann–Whitney test comparing the language groups is
two-sided. Overall results pool all eligible lines across poems, giving each
line equal weight.

## Line-type distribution

| Poem | Regular | Ambiguous | Irregular | Total selected lines |
| ----: | ------: | --------: | --------: | -------------------: |
| 1 | 14 | 7 | 11 | 32 |
| 2 | 7 | 3 | 6 | 16 |
| 3 | 26 | 22 | 16 | 64 |
| 4 | 26 | 18 | 20 | 64 |
| 7 | 4 | 1 | 2 | 7 |
| 11 | 3 | 4 | 7 | 14 |
| 13 | 21 | 23 | 20 | 64 |
| 20 | 53 | 59 | 81 | 193 |
| **Overall** | **154** | **137** | **163** | **454** |

Five selected lines had fewer than five scoreable syllables and were marked
ambiguous with warnings: Poem 3 lines 2, 107, and 113, and Poem 4 lines 2 and
41. Ambiguous lines were excluded from the regular-versus-irregular tests.

![Pooled and per-poem CV distributions](results/exploration/cv_distribution.png)

Across the 449 lines with valid CV values, the median was 0.473. The empirical
33rd and 67th percentiles were 0.397 and 0.565, respectively, close to the
classification thresholds of 0.40 and 0.55.

## Fixed-size windows

The frequency is poem-specific; only the window duration is fixed.

### 0.1 s

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | 0.001062 | 627 | 0.347757 | 0.007242 | 371 | 0.070592 |
| 2 | 2.424 | -0.014209 | 98 | 1.000000 | 0.007199 | 351 | 0.282200 |
| 3 | 1.970 | 0.002181 | 661 | 0.095864 | -0.000206 | 292 | 0.485182 |
| 4 | 1.900 | 0.004180 | 861 | 0.002250 | -0.003890 | 176 | 0.026166 |
| 7 | 1.940 | -0.011005 | 512 | 0.781943 | -0.003914 | 230 | 0.270687 |
| 11 | 1.454 | 0.011769 | 885 | 0.003358 | 0.009665 | 345 | 0.307596 |
| 13 | 1.928 | -0.004938 | 256 | 0.997783 | 0.000211 | 260 | 0.645789 |
| 20 | 1.790 | -0.000548 | 451 | 0.835648 | -0.001264 | 249 | 0.816022 |
| **Overall** | **Varies** | **-0.000472** | **539** | **0.828859** | **0.000274** | **335** | **0.604515** |

### 0.2 s

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | -0.008742 | 5 | 1.000000 | 0.000147 | 298 | 0.771084 |
| 2 | 2.424 | -0.002725 | 470 | 0.921924 | 0.008376 | 373 | 0.129113 |
| 3 | 1.970 | 0.002605 | 884 | 0.000046 | -0.000578 | 258 | 0.973485 |
| 4 | 1.900 | -0.005145 | 43 | 1.000000 | -0.000823 | 259 | 0.617932 |
| 7 | 1.940 | -0.011599 | 229 | 0.999937 | -0.009136 | 208 | 0.119068 |
| 11 | 1.454 | -0.002116 | 463 | 0.931510 | 0.001656 | 287 | 0.895520 |
| 13 | 1.928 | -0.006101 | 35 | 1.000000 | -0.000426 | 260 | 0.645789 |
| 20 | 1.790 | -0.006660 | 0 | 1.000000 | 0.001236 | 367 | 0.018281 |
| **Overall** | **Varies** | **-0.005269** | **3** | **1.000000** | **0.000824** | **367** | **0.252902** |

### 0.3 s

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | -0.001014 | 403 | 0.971700 | -0.001468 | 205 | 0.105008 |
| 2 | 2.424 | 0.006075 | 956 | 0.000219 | 0.005960 | 340 | 0.392949 |
| 3 | 1.970 | -0.001044 | 431 | 0.884078 | -0.000552 | 241 | 0.681856 |
| 4 | 1.900 | 0.000689 | 757 | 0.041982 | 0.000268 | 288 | 0.933748 |
| 7 | 1.940 | -0.002084 | 292 | 0.999056 | 0.000161 | 311 | 0.574708 |
| 11 | 1.454 | -0.000342 | 540 | 0.764602 | -0.001315 | 226 | 0.172649 |
| 13 | 1.928 | 0.000233 | 547 | 0.275715 | 0.000078 | 259 | 0.662801 |
| 20 | 1.790 | -0.002228 | 93 | 1.000000 | 0.000709 | 320 | 0.187360 |
| **Overall** | **Varies** | **-0.000899** | **214** | **0.999992** | **0.000110** | **291** | **0.747091** |

### 0.4 s

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | 0.003029 | 944 | 0.000077 | -0.002338 | 179 | 0.030665 |
| 2 | 2.424 | 0.011149 | 1083 | 0.000000 | 0.007546 | 346 | 0.329616 |
| 3 | 1.970 | -0.002431 | 320 | 0.992621 | -0.000261 | 273 | 0.781795 |
| 4 | 1.900 | 0.001290 | 808 | 0.011706 | 0.000644 | 271 | 0.803058 |
| 7 | 1.940 | 0.001265 | 728 | 0.076870 | 0.002457 | 363 | 0.100625 |
| 11 | 1.454 | 0.000424 | 585 | 0.607786 | 0.001483 | 251 | 0.390533 |
| 13 | 1.928 | 0.002883 | 879 | 0.000001 | -0.000183 | 230 | 0.822822 |
| 20 | 1.790 | 0.000464 | 595 | 0.279314 | 0.001623 | 330 | 0.123557 |
| **Overall** | **Varies** | **0.001270** | **1007** | **0.000115** | **0.000458** | **311** | **0.961031** |

## Cycle-aligned windows

Configurations are written as cycles per window–cycle slide.
The `2–1` configuration is the primary analysis; the other configurations are
sensitivity checks.

### 1–1

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | 0.008484 | 633 | 0.325473 | 0.009516 | 316 | 0.506022 |
| 2 | 2.424 | -0.005775 | 557 | 0.709688 | 0.004903 | 304 | 0.896048 |
| 3 | 1.970 | 0.011153 | 706 | 0.035601 | 0.021580 | 339 | 0.081957 |
| 4 | 1.900 | -0.004143 | 496 | 0.827025 | 0.011190 | 365 | 0.092299 |
| 7 | 1.940 | -0.008377 | 520 | 0.757121 | 0.009639 | 299 | 0.755238 |
| 11 | 1.454 | -0.011528 | 498 | 0.872643 | 0.037247 | 352 | 0.245355 |
| 13 | 1.928 | -0.005802 | 407 | 0.847544 | -0.000912 | 238 | 0.971796 |
| 20 | 1.790 | 0.001647 | 567 | 0.389213 | -0.001798 | 255 | 0.920572 |
| **Overall** | **Varies** | **0.000639** | **661** | **0.412908** | **0.005237** | **388** | **0.120243** |

### 2–1

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | 0.039006 | 875 | 0.001369 | 0.016938 | 300 | 0.739495 |
| 2 | 2.424 | 0.002072 | 620 | 0.472496 | 0.033470 | 357 | 0.231699 |
| 3 | 1.970 | 0.004000 | 591 | 0.294092 | -0.019649 | 218 | 0.357794 |
| 4 | 1.900 | -0.018309 | 362 | 0.990397 | 0.008325 | 312 | 0.560624 |
| 7 | 1.940 | 0.024892 | 725 | 0.081450 | -0.001181 | 269 | 0.771084 |
| 11 | 1.454 | -0.021715 | 460 | 0.935363 | 0.029317 | 336 | 0.401779 |
| 13 | 1.928 | 0.012857 | 638 | 0.048295 | -0.007357 | 207 | 0.443657 |
| 20 | 1.790 | 0.012733 | 722 | 0.023651 | -0.026997 | 170 | 0.047347 |
| **Overall** | **Varies** | **0.007895** | **855** | **0.017696** | **-0.007774** | **257** | **0.323654** |

### 1–0.5

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | -0.020998 | 0 | 1.000000 | -0.003408 | 233 | 0.298743 |
| 2 | 2.424 | -0.021039 | 115 | 1.000000 | 0.012870 | 351 | 0.282200 |
| 3 | 1.970 | 0.009891 | 961 | 0.000000 | 0.004904 | 330 | 0.123557 |
| 4 | 1.900 | -0.009297 | 126 | 1.000000 | 0.001621 | 313 | 0.546710 |
| 7 | 1.940 | -0.011363 | 183 | 0.999994 | 0.005781 | 356 | 0.134562 |
| 11 | 1.454 | -0.011350 | 194 | 0.999984 | 0.004330 | 329 | 0.485788 |
| 13 | 1.928 | -0.012763 | 19 | 1.000000 | 0.002662 | 276 | 0.402737 |
| 20 | 1.790 | -0.014109 | 8 | 1.000000 | 0.002973 | 322 | 0.172962 |
| **Overall** | **Varies** | **-0.009448** | **4** | **1.000000** | **0.003172** | **413** | **0.041116** |

### 2–0.5

| Poem | Freq. | Regular minus Irregular | Wilcoxon W | Wilcoxon p | Bilingual minus monolingual | Mann–Whitney U | MW p |
| ---- | ----- | -----------------------: | ---------: | ----------: | ---------------------------: | -------------: | ---: |
| 1 | 2.209 | -0.044693 | 0 | 1.000000 | -0.005769 | 229 | 0.261750 |
| 2 | 2.424 | -0.076068 | 27 | 1.000000 | 0.004110 | 288 | 0.864337 |
| 3 | 1.970 | 0.030573 | 1015 | 0.000000 | 0.002911 | 278 | 0.698184 |
| 4 | 1.900 | -0.017962 | 4 | 1.000000 | 0.005146 | 361 | 0.109540 |
| 7 | 1.940 | -0.020778 | 2 | 1.000000 | 0.002968 | 310 | 0.588956 |
| 11 | 1.454 | -0.017382 | 50 | 1.000000 | -0.002163 | 264 | 0.551172 |
| 13 | 1.928 | -0.030121 | 0 | 1.000000 | -0.001932 | 220 | 0.645789 |
| 20 | 1.790 | -0.035563 | 0 | 1.000000 | -0.001261 | 236 | 0.602559 |
| **Overall** | **Varies** | **-0.022538** | **0** | **1.000000** | **0.000386** | **322** | **0.791899** |

## Chosen configuration: cycle-aligned 2–1

These results aggregate all eligible poems and lines within participant and
EEG channel; poem-wise results are not used here. `f` is each poem's implied
stress frequency and `2f` is its doubled frequency (the expected syllable-rate
proxy). Channel-wise p-values are uncorrected for multiple comparisons.

### Regular minus irregular

The Wilcoxon test is one-sided for the planned hypothesis that regular ITPC is
greater than irregular ITPC.

| Channel | Freq. | Regular minus Irregular | Wilcoxon W (one-sided) | Wilcoxon p (one-sided) |
| --- | --- | ---: | ---: | ---: |
| C3 | f | 0.007507 | 789 | 0.073009 |
| C4 | f | 0.008761 | 775 | 0.093842 |
| F3 | f | 0.002875 | 667 | 0.390697 |
| F4 | f | 0.012132 | 786 | 0.077151 |
| F7 | f | 0.008767 | 818 | **0.041134** |
| F8 | f | 0.020109 | 911 | **0.003805** |
| Fp1 | f | 0.007657 | 773 | 0.097138 |
| Fp2 | f | 0.003964 | 674 | 0.365224 |
| Fz | f | 0.003441 | 723 | 0.207596 |
| O1 | f | 0.007043 | 730 | 0.188837 |
| O2 | f | 0.008771 | 795 | 0.065229 |
| P3 | f | 0.000829 | 677 | 0.354476 |
| P4 | f | 0.003366 | 776 | 0.092224 |
| P7 | f | 0.002266 | 676 | 0.358047 |
| P8 | f | 0.010436 | 822 | **0.037781** |
| Pz | f | 0.002025 | 663 | 0.405470 |
| T7 | f | 0.017304 | 910 | **0.003922** |
| T8 | f | 0.014862 | 811 | **0.047567** |
| **All EEG channels** | **f** | **0.007895** | **855** | **0.017696** |

![One-sided p-values by EEG channel for regular minus irregular ITPC](results/frequency_comparison/chosen_2-1_regular_minus_irregular_one_sided_p.png)

### Regular lines: poetic frequency minus doubled poetic frequency

The one-sided test uses the hypothesis `f > 2f`; the two-sided test evaluates
any difference between the frequencies.

| Channel | Freq. | Regular f minus Regular 2f | W (one-sided) | p (one-sided) | W (two-sided) | p (two-sided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C3 | f minus 2f | -0.001207 | 628 | 0.538109 | 628 | 0.931384 |
| C4 | f minus 2f | 0.010150 | 821 | **0.038598** | 454 | 0.077196 |
| F3 | f minus 2f | 0.006720 | 743 | 0.156833 | 532 | 0.313666 |
| F4 | f minus 2f | 0.015673 | 836 | **0.027733** | 439 | 0.055466 |
| F7 | f minus 2f | 0.007357 | 744 | 0.154526 | 531 | 0.309051 |
| F8 | f minus 2f | 0.015939 | 859 | **0.016027** | 416 | **0.032053** |
| Fp1 | f minus 2f | 0.007703 | 721 | 0.213148 | 554 | 0.426296 |
| Fp2 | f minus 2f | 0.006136 | 710 | 0.245163 | 565 | 0.490326 |
| Fz | f minus 2f | 0.013390 | 854 | **0.018135** | 421 | **0.036270** |
| O1 | f minus 2f | 0.009107 | 777 | 0.090627 | 498 | 0.181255 |
| O2 | f minus 2f | 0.014052 | 866 | **0.013424** | 409 | **0.026849** |
| P3 | f minus 2f | 0.003287 | 695 | 0.292603 | 580 | 0.585207 |
| P4 | f minus 2f | 0.009032 | 798 | 0.061585 | 477 | 0.123169 |
| P7 | f minus 2f | 0.001052 | 637 | 0.503817 | 637 | 1.000000 |
| P8 | f minus 2f | 0.016782 | 896 | **0.005934** | 379 | **0.011867** |
| Pz | f minus 2f | 0.007284 | 734 | 0.178594 | 541 | 0.357187 |
| T7 | f minus 2f | 0.014470 | 868 | **0.012750** | 407 | **0.025500** |
| T8 | f minus 2f | 0.015735 | 876 | **0.010333** | 399 | **0.020666** |
| **All EEG channels** | **f minus 2f** | **0.009592** | **859** | **0.016027** | **416** | **0.032053** |

![One-sided p-values by EEG channel for regular-line poetic versus doubled frequency ITPC](results/frequency_comparison/chosen_2-1_regular_f_vs_2f_one_sided_p.png)

### Irregular lines: poetic frequency minus doubled poetic frequency

The one-sided test uses the hypothesis `f > 2f`; the two-sided test evaluates
any difference between the frequencies.

| Channel | Freq. | Irregular f minus Irregular 2f | W (one-sided) | p (one-sided) | W (two-sided) | p (two-sided) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C3 | f minus 2f | -0.003102 | 582 | 0.704091 | 582 | 0.598463 |
| C4 | f minus 2f | 0.002494 | 701 | 0.273133 | 574 | 0.546265 |
| F3 | f minus 2f | 0.004610 | 748 | 0.145518 | 527 | 0.291035 |
| F4 | f minus 2f | 0.005916 | 751 | 0.138994 | 524 | 0.277988 |
| F7 | f minus 2f | 0.000896 | 644 | 0.477112 | 631 | 0.954225 |
| F8 | f minus 2f | -0.000265 | 643 | 0.480924 | 632 | 0.961848 |
| Fp1 | f minus 2f | 0.003077 | 724 | 0.204852 | 551 | 0.409704 |
| Fp2 | f minus 2f | 0.002770 | 696 | 0.289315 | 579 | 0.578629 |
| Fz | f minus 2f | 0.010584 | 828 | **0.033166** | 447 | 0.066332 |
| O1 | f minus 2f | 0.006643 | 805 | 0.053688 | 470 | 0.107376 |
| O2 | f minus 2f | 0.009330 | 839 | **0.025893** | 436 | 0.051786 |
| P3 | f minus 2f | 0.005599 | 777 | 0.090627 | 498 | 0.181255 |
| P4 | f minus 2f | 0.007902 | 847 | **0.021472** | 428 | **0.042944** |
| P7 | f minus 2f | 0.004712 | 780 | 0.085957 | 495 | 0.171914 |
| P8 | f minus 2f | 0.007645 | 800 | 0.059243 | 475 | 0.118485 |
| Pz | f minus 2f | 0.009403 | 823 | **0.036978** | 452 | 0.073956 |
| T7 | f minus 2f | 0.002086 | 707 | 0.254315 | 568 | 0.508630 |
| T8 | f minus 2f | 0.004527 | 736 | 0.173603 | 539 | 0.347207 |
| **All EEG channels** | **f minus 2f** | **0.004713** | **777** | **0.090627** | **498** | **0.181255** |

### Exploratory language-group analysis at pooled-effect channels

The following channels were selected from the pooled channel analyses above,
then examined separately by language group. This is an exploratory follow-up:
the displayed within-group p-values are uncorrected and must not be treated as
independent confirmation of the pooled effects.

#### Regular minus irregular

These five channels had pooled one-sided `regular > irregular` p-values below
0.05. The same one-sided hypothesis is tested within each group.

| Channel | Bilingual Δ | Bilingual W | Bilingual p | Monolingual Δ | Monolingual W | Monolingual p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F7 | 0.006443 | 239 | 0.212376 | 0.011725 | 181 | **0.039714** |
| F8 | 0.011692 | 248 | 0.158048 | 0.030821 | 218 | **0.000959** |
| P8 | 0.007985 | 250 | 0.147299 | 0.013556 | 174 | 0.064440 |
| T7 | 0.018573 | 276 | **0.049640** | 0.015688 | 195 | **0.012569** |
| T8 | 0.006810 | 222 | 0.339089 | 0.025111 | 186 | **0.027123** |

#### Regular lines: poetic frequency minus doubled poetic frequency

These six channels had pooled two-sided `f` versus `2f` p-values below 0.05.
The group analysis therefore reports the matching two-sided tests.

| Channel | Bilingual Δ | Bilingual W | Bilingual p (two-sided) | Monolingual Δ | Monolingual W | Monolingual p (two-sided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F8 | 0.007773 | 179 | 0.597922 | 0.026333 | 44 | **0.005929** |
| Fz | 0.008533 | 156 | 0.294598 | 0.019571 | 69 | 0.063419 |
| O2 | 0.012640 | 143 | 0.178234 | 0.015849 | 72 | 0.079427 |
| P8 | 0.014516 | 139 | 0.150235 | 0.019667 | 60 | **0.030118** |
| T7 | 0.013834 | 143 | 0.178234 | 0.015280 | 76 | 0.105453 |
| T8 | 0.010413 | 158 | 0.316096 | 0.022509 | 45 | **0.006649** |

## ITPC differences by language group

These within-group paired tests use the chosen cycle-aligned `2–1`
configuration and aggregate all eligible poems and lines within participant.
The bilingual group has 28 participants and the monolingual group has 22.
They do not test the difference between groups.
Channel-wise p-values are uncorrected for multiple comparisons.

### Regular minus irregular at poetic frequency

The one-sided Wilcoxon test evaluates the planned hypothesis that regular ITPC
is greater than irregular ITPC.

| Group | n | ITPC difference | Wilcoxon W | Wilcoxon p (one-sided) |
| --- | ---: | ---: | ---: | ---: |
| Bilingual | 28 | 0.004475 | 240 | 0.205867 |
| Monolingual | 22 | 0.012249 | 193 | **0.015059** |

| Channel | Bilingual Δ | Bilingual W | Bilingual p | Monolingual Δ | Monolingual W | Monolingual p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C3 | 0.008894 | 260 | 0.100784 | 0.005742 | 150 | 0.231389 |
| C4 | 0.001273 | 215 | 0.398281 | 0.018293 | 179 | **0.045866** |
| F3 | -0.006090 | 168 | 0.787624 | 0.014284 | 157 | 0.168474 |
| F4 | 0.005553 | 213 | 0.415687 | 0.020504 | 178 | **0.049205** |
| F7 | 0.006443 | 239 | 0.212376 | 0.011725 | 181 | **0.039714** |
| F8 | 0.011692 | 248 | 0.158048 | 0.030821 | 218 | **0.000959** |
| Fp1 | 0.002245 | 220 | 0.355683 | 0.014545 | 175 | 0.060339 |
| Fp2 | -0.003105 | 177 | 0.724063 | 0.012961 | 161 | 0.137813 |
| Fz | -0.006215 | 165 | 0.806803 | 0.015729 | 177 | 0.052727 |
| O1 | 0.008561 | 234 | 0.246617 | 0.005110 | 146 | 0.272295 |
| O2 | 0.005707 | 226 | 0.306819 | 0.012672 | 180 | **0.042705** |
| P3 | 0.004363 | 240 | 0.205867 | -0.003669 | 115 | 0.648808 |
| P4 | 0.004111 | 231 | 0.268454 | 0.002418 | 164 | 0.117381 |
| P7 | 0.002311 | 217 | 0.381070 | 0.002209 | 131 | 0.449368 |
| P8 | 0.007985 | 250 | 0.147299 | 0.013556 | 174 | 0.064440 |
| Pz | 0.001431 | 209 | 0.450950 | 0.002781 | 136 | 0.387264 |
| T7 | 0.018573 | 276 | **0.049640** | 0.015688 | 195 | **0.012569** |
| T8 | 0.006810 | 222 | 0.339089 | 0.025111 | 186 | **0.027123** |
| **All EEG channels** | **0.004475** | **240** | **0.205867** | **0.012249** | **193** | **0.015059** |

![Regular-minus-irregular ITPC difference by language group](results/frequency_comparison/chosen_2-1_regular_minus_irregular_by_language_group.png)

### Regular lines: poetic frequency minus doubled poetic frequency

The one-sided test evaluates `f > 2f`; the two-sided test evaluates any
difference between the frequencies.

In the table, `Bi` means bilingual, `Mono` means monolingual, and subscripts
1 and 2 indicate one-sided and two-sided Wilcoxon results, respectively.

| Group | n | ITPC difference | W (one-sided) | p (one-sided) | W (two-sided) | p (two-sided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bilingual | 28 | 0.006258 | 244 | 0.181001 | 162 | 0.362001 |
| Monolingual | 22 | 0.013836 | 194 | **0.013768** | 59 | **0.027535** |

| Channel | Bi Δ | Bi W₁ | Bi p₁ | Bi W₂ | Bi p₂ | Mono Δ | Mono W₁ | Mono p₁ | Mono W₂ | Mono p₂ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C3 | -0.002377 | 197 | 0.557912 | 197 | 0.901900 | 0.000283 | 127 | 0.500000 | 126 | 1.000000 |
| C4 | 0.002996 | 232 | 0.261072 | 174 | 0.522144 | 0.019255 | 185 | **0.029345** | 68 | 0.058690 |
| F3 | 0.004178 | 224 | 0.322793 | 182 | 0.645585 | 0.009956 | 163 | 0.123950 | 90 | 0.247899 |
| F4 | 0.006881 | 225 | 0.314764 | 181 | 0.629528 | 0.026861 | 196 | **0.011458** | 57 | **0.022916** |
| F7 | 0.006410 | 224 | 0.322793 | 182 | 0.645585 | 0.008564 | 164 | 0.117381 | 89 | 0.234762 |
| F8 | 0.007773 | 227 | 0.298961 | 179 | 0.597922 | 0.026333 | 209 | **0.002964** | 44 | **0.005929** |
| Fp1 | 0.005215 | 217 | 0.381070 | 189 | 0.762139 | 0.010869 | 153 | 0.203017 | 100 | 0.406033 |
| Fp2 | 0.003071 | 210 | 0.442088 | 196 | 0.884176 | 0.010038 | 154 | 0.194022 | 99 | 0.388045 |
| Fz | 0.008533 | 250 | 0.147299 | 156 | 0.294598 | 0.019571 | 184 | **0.031710** | 69 | 0.063419 |
| O1 | 0.005930 | 232 | 0.261072 | 174 | 0.522144 | 0.013151 | 168 | 0.093488 | 85 | 0.186976 |
| O2 | 0.012640 | 263 | 0.089117 | 143 | 0.178234 | 0.015849 | 181 | **0.039714** | 72 | 0.079427 |
| P3 | 0.003834 | 223 | 0.330902 | 183 | 0.661805 | 0.002590 | 137 | 0.375119 | 116 | 0.750238 |
| P4 | 0.007263 | 247 | 0.163605 | 159 | 0.327210 | 0.011283 | 163 | 0.123950 | 90 | 0.247899 |
| P7 | 0.000648 | 206 | 0.477661 | 200 | 0.955322 | 0.001566 | 118 | 0.612736 | 118 | 0.799034 |
| P8 | 0.014516 | 267 | 0.075118 | 139 | 0.150235 | 0.019667 | 193 | **0.015059** | 60 | **0.030118** |
| Pz | 0.000892 | 205 | 0.486592 | 201 | 0.973185 | 0.015420 | 165 | 0.111054 | 88 | 0.222107 |
| T7 | 0.013834 | 263 | 0.089117 | 143 | 0.178234 | 0.015280 | 177 | 0.052727 | 76 | 0.105453 |
| T8 | 0.010413 | 248 | 0.158048 | 158 | 0.316096 | 0.022509 | 208 | **0.003325** | 45 | **0.006649** |
| **All EEG channels** | **0.006258** | **244** | **0.181001** | **162** | **0.362001** | **0.013836** | **194** | **0.013768** | **59** | **0.027535** |

![Regular-line poetic-minus-doubled-frequency ITPC difference by language group](results/frequency_comparison/chosen_2-1_regular_f_vs_2f_by_language_group.png)

## Continuous line regularity (CV) and ITPC

This analysis does not use the regular, ambiguous, or irregular categories.
Each line has equal weight: ITPC is first averaged across participants and EEG
channels for that poem-line, then correlated with line CV. ITPC uses the chosen
cycle-aligned `2–1` configuration at each poem's implied stress frequency.
Five lines without a valid CV were excluded. Poem 3 line 105 has a valid CV but
is excluded because its duration supports only one window; debiased ITPC
requires at least two windows.

The primary test is a two-sided Spearman rank correlation between line CV and
line-mean ITPC. A negative correlation supports the hypothesis that lower CV
(more regular timing) is associated with higher ITPC.

| Analysis | Spearman rho | Two-sided p | Lines |
| --- | ---: | ---: | ---: |
| CV versus line-mean ITPC | -0.103825 | **0.027994** | 448 |

![Line CV versus ITPC](results/cv_itpc/cv_vs_itpc.png)

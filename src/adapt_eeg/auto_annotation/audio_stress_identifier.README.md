# Our version vs DE JONG AND WEMPE 
<https://link.springer.com/article/10.3758/BRM.41.2.385

| Part                                  | de Jong & Wempe method                                  | Current code                                | Match?                                 |
| ------------------------------------- | ------------------------------------------------------- | ------------------------------------------- | -------------------------------------- |
| Intensity-based nucleus detection     | Yes                                                     | Yes, uses Praat intensity                   | **Yes**                                |
| [Detect local intensity peaks](#detect-local-intensity-peaks)         | Yes                                                     | `find_peaks(intensity_db)`                  | **Yes, conceptually**                  |
| Require dips before/after peak        | Yes                                                     | Explicit left/right dip check               | **Yes**                                |
| Reject unvoiced peaks                 | Yes                                                     | Checks whether nearby Praat F0 is finite    | **Yes**                                |
| No transcript needed                  | Yes                                                     | Yes                                         | **Yes**                                |
| Minimum spacing between peaks         | Paper method has its own script logic/parameters        | We imposed `0.20 s`                         | **Our addition**                       |
| Silence threshold                     | Their script uses parameterized intensity/silence logic | We use `max_intensity - 25 dB`              | **Not necessarily equivalent**         |
| Dip definition                        | Their Praat script has specific settings                | We search ±`0.20 s` and require `2 dB` dips | **Our approximation**                  |
| Voicing criterion                     | Their script tests whether candidate is voiced          | We require valid F0 at nearest pitch frame  | **Similar, not necessarily identical** |
| Pitch range                           | Used for voicing in their approach                      | Fixed `165–255 Hz` from your poetry paper   | **Different source**                   |
| Estimated syllable boundaries         | Not required for their core speech-rate detector        | Midpoints between nuclei                    | **Our addition**                       |
| Pitch/intensity/duration stress score | Not part of de Jong & Wempe                             | Added for your poetry stress problem        | **Entirely our extension**             |


## Detect local intensity peaks

- 0.2s min dist between 2 peaks (4Hz syllabic freq -> 0.25s -> round down 0.2s for some flexibility).

    In their version, only dips are required

- Use scipy (dont see their code)

- Time steps of 0.01s (they dont mention)
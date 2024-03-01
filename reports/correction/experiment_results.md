### Experiment XXX (2024-03-01) { #sec-experiment-correction-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Detection model from experiment [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Max token length: 22
* Model: XXX
* Decoder: XXX
* Loss
    * Train: 0.6192
    * Val: 0.4882390201091766
    * Test: 0.5294567942619324

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   16 |  -18 |   56 |    2 |    7 |   38 |   11 |    6 |    9 |  -14 |

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   10 |  -21 |   52 |   -9 |    1 |   34 |    2 |   -0 |    5 |  -24 |

### Remarks
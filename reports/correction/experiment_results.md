### Experiment byt5-small 1 (2024-03-01) { #sec-experiment-correction-byt5-small-1 }

* ocrpostcorrection-notebooks commit: [b677b6b](https://github.com/jvdzwaan/ocrpostcorrection-notebooks/commit/b677b6b4f2097a1eae7f4f374948da3635f5ceba)
* Detection model from experiment [9099e78](https://github.com/jvdzwaan/ocrpostcorrection-notebooks/commit/9099e785177a5c5207d01d80422e68d30f39636d)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Max token length: 22
* Model: byt5-small
    * Number of epochs: 1
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
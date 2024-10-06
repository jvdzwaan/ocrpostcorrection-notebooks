### Experiment XXX (2024-10-06) { #sec-experiment-correction-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Detection model from experiment [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Max token length: 22
* Model: google/byt5-small
    * Number of epochs: 1
* Loss
    * Train: 0.2877
    * Val: 0.2515017390251159
    * Test: 0.277634710073471

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   22 |    1 |   72 |   12 |   23 |   49 |   24 |   27 |   28 |   14 |

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   13 |  -21 |   65 |   -5 |   12 |   42 |    7 |   18 |   22 |    2 |

### Remarks
### Experiment XXX (2023-07-23) { #sec-experiment-correction-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Detection model from experiment [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Max token length: 22
* Model: XXX
* Decoder: XXX
* Loss
    * Train: 23.257548828125
    * Val: 19.877145020723148
    * Test: 20.81730119857481

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement | -214 | -302 | -145 | -118 |  -75 | -283 |  nan |  -95 | -147 | -175 |

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement | -225 | -216 | -146 | -123 |  -78 | -279 | -195 | -111 | -141 | -183 |

### Remarks

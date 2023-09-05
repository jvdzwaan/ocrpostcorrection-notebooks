### Experiment XXX (2023-09-05) { #sec-experiment-correction-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Detection model from experiment [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Max token length: 22
* Model: XXX
* Decoder: XXX
* Loss
    * Train: 7.035337067127228
    * Val: 7.531168795152685
    * Test: 8.52123194542849

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement | -283 | -444 | -131 | -171 | -127 | -226 | -292 | -144 | -195 | -275 |

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement | -296 | -364 | -140 | -180 | -114 | -220 | -237 | -156 | -184 | -305 |

### Remarks
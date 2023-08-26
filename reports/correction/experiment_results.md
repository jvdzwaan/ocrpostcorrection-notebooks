### Experiment XXX (2023-08-26) { #sec-experiment-correction-XXX }

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
    * Test: 8.613492756178495

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   17 |  -67 |   25 |   -4 |   17 |   21 |   -3 |   10 |   -7 |  -32 |

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   -4 |  -48 |   28 |  -20 |   -7 |   18 |  -13 |   -7 |  -15 |  -47 |

### Remarks
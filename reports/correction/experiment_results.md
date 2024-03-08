### Experiment byt5-small 2: AdaFactor optimizer (2024-03-08) { #sec-experiment-correction-byt5-small-2 }

* ocrpostcorrection-notebooks commit: [7665d86](https://github.com/jvdzwaan/ocrpostcorrection-notebooks/commit/7665d86e2a210e503b9332d83baa5174f89ebf99)
* Detection model from experiment [9099e78](https://github.com/jvdzwaan/ocrpostcorrection-notebooks/commit/9099e785177a5c5207d01d80422e68d30f39636d)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Max token length: 22
* Model: byt5-small
    * Number of epochs: 1
* Loss
    * Train: 0.4592
    * Val: 0.3836239278316498
    * Test: 0.4266203045845032

Trained on Google Colab T4.

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   21 |  -14 |   65 |    2 |   10 |   42 |   13 |   14 |   17 |   -8 |

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

|                 |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:----------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| %ed_improvement |   13 |  -15 |   59 |   -7 |    3 |   37 |    1 |    6 |   11 |  -19 |

### Remarks
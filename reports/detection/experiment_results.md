### Experiment XXX (2023-04-01) { #sec-experiment-detection-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Normalized editdistance threshold for 'sentences': 0.3 (only for train and val)
    * Sequence (sentence) length: size: 35, step: 30
* Pretrained model: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
* Loss
    * Train: 0.2398
    * Val: 0.2871749699115753
    * Test: 0.4474944472312927

| language   |   T1_Precision |   T1_Recall |   T1_Fmesure |
|:-----------|---------------:|------------:|-------------:|
| BG         |           0.86 |        0.77 |         0.8  |
| CZ         |           0.82 |        0.62 |         0.69 |
| DE         |           0.97 |        0.97 |         0.97 |
| EN         |           0.82 |        0.64 |         0.69 |
| ES         |           0.89 |        0.65 |         0.73 |
| FI         |           0.91 |        0.78 |         0.83 |
| FR         |           0.8  |        0.66 |         0.71 |
| NL         |           0.88 |        0.72 |         0.76 |
| PL         |           0.89 |        0.78 |         0.83 |
| SL         |           0.82 |        0.7  |         0.74 |

### Summarized results

|            |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:-----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| T1_Fmesure |  0.8 | 0.69 | 0.97 | 0.69 | 0.73 | 0.83 | 0.71 | 0.76 | 0.83 | 0.74 |
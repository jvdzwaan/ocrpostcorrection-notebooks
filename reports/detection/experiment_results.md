### Experiment XXX (2023-03-30) { #sec-experiment-detection-XXX }

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
| BG         |           0.85 |        0.71 |         0.75 |
| CZ         |           0.82 |        0.6  |         0.68 |
| DE         |           0.97 |        0.96 |         0.96 |
| EN         |           0.81 |        0.6  |         0.66 |
| ES         |           0.87 |        0.54 |         0.64 |
| FI         |           0.91 |        0.78 |         0.83 |
| FR         |           0.8  |        0.63 |         0.68 |
| NL         |           0.86 |        0.63 |         0.69 |
| PL         |           0.89 |        0.75 |         0.81 |
| SL         |           0.81 |        0.62 |         0.67 |

### Summarized results

|            |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:-----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| T1_Fmesure | 0.75 | 0.68 | 0.96 | 0.66 | 0.64 | 0.83 | 0.68 | 0.69 | 0.81 | 0.67 |
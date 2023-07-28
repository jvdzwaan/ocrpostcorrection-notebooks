### Experiment XXX (2023-07-28) { #sec-experiment-detection-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Normalized editdistance threshold for 'sentences': 0.3 (only for train and val)
    * Sequence (sentence) length: size: 35, step: 30
* Pretrained model: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
* Loss
    * Train: 0.2625
    * Val: 0.2949527204036712
    * Test: 0.4553228318691253

| language   |   T1_Precision |   T1_Recall |   T1_Fmesure |
|:-----------|---------------:|------------:|-------------:|
| BG         |           0.86 |        0.69 |         0.74 |
| CZ         |           0.85 |        0.59 |         0.68 |
| DE         |           0.97 |        0.95 |         0.96 |
| EN         |           0.83 |        0.59 |         0.66 |
| ES         |           0.88 |        0.53 |         0.63 |
| FI         |           0.89 |        0.79 |         0.83 |
| FR         |           0.8  |        0.61 |         0.67 |
| NL         |           0.86 |        0.64 |         0.69 |
| PL         |           0.89 |        0.75 |         0.8  |
| SL         |           0.82 |        0.63 |         0.68 |

### Summarized results

|            |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:-----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| T1_Fmesure | 0.74 | 0.68 | 0.96 | 0.66 | 0.63 | 0.83 | 0.67 | 0.69 |  0.8 | 0.68 |
### Experiment XXX (2023-04-22) { #sec-experiment-detection-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Dataset
    * Split seed: 8232
    * Validation set: 10.0%
    * Normalized editdistance threshold for 'sentences': 0.3 (only for train and val)
    * Sequence (sentence) length: size: 35, step: 30
* Pretrained model: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
* Loss
    * Train: 0.2439
    * Val: 0.2839458584785461
    * Test: 0.4422231018543243

| language   |   T1_Precision |   T1_Recall |   T1_Fmesure |
|:-----------|---------------:|------------:|-------------:|
| BG         |           0.86 |        0.7  |         0.75 |
| CZ         |           0.85 |        0.6  |         0.69 |
| DE         |           0.97 |        0.95 |         0.96 |
| EN         |           0.82 |        0.61 |         0.67 |
| ES         |           0.89 |        0.53 |         0.63 |
| FI         |           0.89 |        0.79 |         0.83 |
| FR         |           0.81 |        0.62 |         0.69 |
| NL         |           0.87 |        0.64 |         0.69 |
| PL         |           0.89 |        0.75 |         0.81 |
| SL         |           0.81 |        0.64 |         0.68 |

### Summarized results

|            |   BG |   CZ |   DE |   EN |   ES |   FI |   FR |   NL |   PL |   SL |
|:-----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| T1_Fmesure | 0.75 | 0.69 | 0.96 | 0.67 | 0.63 | 0.83 | 0.69 | 0.69 | 0.81 | 0.68 |
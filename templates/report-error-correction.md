{#- templates/report-error-correction.md -#}
### Experiment XXX ({{today}}) { #sec-experiment-correction-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Detection model from experiment [XXX](XXX)
* Dataset
    * Split seed: {{seed}}
    * Validation set: {{val_size*100}}%
    * Max token length: {{max_len}}
* Model: XXX
* Decoder: XXX
* Loss
    * Train: {{train_loss}}
    * Val: {{val_loss}}
    * Test: {{test_loss}}

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the 'perfect' results for error detection.

{{table_perfect}}

### Summarized results (average % of improvement in edit distance between original and corrected)

The input is the errors detected by a model.

{{table_predicted}}

### Remarks

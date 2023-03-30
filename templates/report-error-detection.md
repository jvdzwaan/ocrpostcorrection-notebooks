{#- templates/report-error-detection.md -#}
### Experiment XXX ({{today}}) { #sec-experiment-detection-XXX }

* ocrpostcorrection-notebooks commit: [XXX](XXX)
* Dataset
    * Split seed: {{seed}}
    * Validation set: {{val_size*100}}%
    * Normalized editdistance threshold for 'sentences': {{max_edit_distance}} (only for train and val)
    * Sequence (sentence) length: size: {{size}}, step: {{step}}
* Pretrained model: [{{model_name}}](https://huggingface.co/{{model_name}})
* Loss
    * Train: {{train_loss}}
    * Val: {{val_loss}}
    * Test: {{test_loss}}

{{results_table}}

### Summarized results

{{summarized_results}}

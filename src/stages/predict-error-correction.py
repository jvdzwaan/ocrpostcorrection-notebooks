import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import edlib
import pandas as pd
import torch
import typer
from datasets import Dataset
from loguru import logger
from ocrpostcorrection.bert_vectors_correction_data import (
    BertVectorsCorrectionDataset,
    collate_fn,
    predict_and_convert_to_str,
    validate_model,
)
from ocrpostcorrection.error_correction import (
    SimpleCorrectionSeq2seq,
    generate_vocabs,
    get_text_transform,
)
from ocrpostcorrection.icdar_data import (
    Text,
    extract_icdar_data,
    generate_data,
    normalized_ed,
)
from ocrpostcorrection.utils import icdar_output2simple_correction_dataset_df
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from transformers import AutoTokenizer, BertModel, DataCollatorWithPadding
from typing_extensions import Annotated

from common.option_types import dir_in_option, file_in_option, file_out_option

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
    dropout: float,
    max_len: int,
    teacher_forcing_ratio: float,
    model_path: Path,
) -> Tuple[SimpleCorrectionSeq2seq, torch.optim.Optimizer]:
    model = SimpleCorrectionSeq2seq(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout=dropout,
        max_length=max_len,
        teacher_forcing_ratio=teacher_forcing_ratio,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


def create_test_dataloader(
    test_df: pd.DataFrame,
    max_len: int,
    hidden_size: int,
    batch_size: int,
    text_transform: Dict[str, Callable[[Any], Any]],
    look_up_bert_vectors: bool,
    bert_vectors_file: Optional[Path] = None,
) -> DataLoader:
    test_dataset = BertVectorsCorrectionDataset(
        data=test_df,
        bert_vectors_file=bert_vectors_file,
        split_name="test",
        max_len=max_len,
        hidden_size=hidden_size,
        look_up_bert_vectors=look_up_bert_vectors,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn(text_transform)
    )
    logger.info(f"# test samples: {len(test_dataset)}")

    return test_dataloader


def create_predictions_csv(
    test: pd.DataFrame, max_len: int, predictions: pd.DataFrame
) -> pd.DataFrame:
    test_results = (
        test.query(f"len_ocr <= {max_len}").query(f"len_gs <= {max_len}").copy()
    )
    test_results["pred"] = predictions

    test_results["ed"] = test_results.apply(
        lambda row: edlib.align(row.ocr, row.gs)["editDistance"], axis=1
    )
    test_results["ed_norm"] = test_results.apply(
        lambda row: normalized_ed(row.ed, row.ocr, row.gs), axis=1
    )

    test_results["ed_pred"] = test_results.apply(
        lambda row: edlib.align(row.pred, row.gs)["editDistance"], axis=1
    )
    test_results["ed_norm_pred"] = test_results.apply(
        lambda row: normalized_ed(row.ed_pred, row.pred, row.gs), axis=1
    )

    return test_results


def predict_and_save(
    in_file: Path,
    data_test: Dict[str, Text],
    max_len: int,
    hidden_size: int,
    tokenizer: AutoTokenizer,
    bert_model: BertModel,
    batch_size: int,
    text_transform: Dict[str, Callable[[Any], Any]],
    vocab_transform: Dict[str, Vocab],
    model: SimpleCorrectionSeq2seq,
    out_file: Path,
) -> None:
    logger.info(f"Generating predictions for '{in_file}'")
    if in_file and out_file:
        with open(in_file) as f:
            output = json.load(f)
        test = icdar_output2simple_correction_dataset_df(output, data_test)
        test_dataloader = create_test_dataloader(
            test,
            bert_vectors_file=None,
            max_len=max_len,
            hidden_size=hidden_size,
            batch_size=batch_size,
            text_transform=text_transform,
            look_up_bert_vectors=False,
        )

        dataset = Dataset.from_pandas(test_dataloader.dataset.ds.ocr.to_frame())
        tokenized_dataset = dataset.map(
            lambda sample: tokenizer(sample["ocr"], truncation=True),
            batched=True,
        )
        tokenized_dataset = tokenized_dataset.remove_columns(["ocr"])

        collator = DataCollatorWithPadding(tokenizer)
        test_dataloader_bert_vectors = DataLoader(
            tokenized_dataset, batch_size=batch_size, collate_fn=collator
        )

        predictions = predict_and_convert_to_str(
            model=model,
            dataloader=test_dataloader,
            bert_model=bert_model,
            dataloader_bert_vectors=test_dataloader_bert_vectors,
            tgt_vocab=vocab_transform["gs"],
            device=device,
        )

        results = create_predictions_csv(test, max_len, predictions)
        results.to_csv(out_file)
        logger.info(f"Saved predictions to '{out_file}'")
    elif in_file or out_file:
        logger.info(
            "To generate predictions for an icdar json file, please specify both --*_in"
            + " and --*_out. (Got '{in_file}' (--*_in) and '{out_file}' (--*_out))"
        )


def predict_error_correction(
    error_correction_dataset: Annotated[Path, file_in_option],
    bert_vectors_file: Annotated[Path, file_in_option],
    bert_model_name: Annotated[str, typer.Option()],
    bert_model_dir: Annotated[Path, dir_in_option],
    hidden_size: Annotated[int, typer.Option()],
    dropout: Annotated[float, typer.Option()],
    max_len: Annotated[int, typer.Option()],
    teacher_forcing_ratio: Annotated[float, typer.Option()],
    model_path: Annotated[Path, file_in_option],
    batch_size: Annotated[int, typer.Option()],
    calculate_loss: Annotated[bool, typer.Option()],
    test_log_file: Annotated[Path, file_out_option],
    raw_dataset: Annotated[Path, file_in_option],
    perfect_detection_in: Annotated[Path, file_in_option],
    perfect_detection_out: Annotated[Path, file_out_option],
    predicted_detection_in: Annotated[Path, file_in_option],
    predicted_detection_out: Annotated[Path, file_out_option],
) -> None:
    # Load data
    logger.info("Loading data")
    data = pd.read_csv(error_correction_dataset, index_col=0)
    data = data.fillna("")

    train = data.query('dataset == "train"')
    test = data.query('dataset == "test"')

    logger.info(f"Train: {train.shape[0]} samples; test: {test.shape[0]} samples")

    # Load model
    logger.info("Loading model")
    vocab_transform = generate_vocabs(train)
    text_transform = get_text_transform(vocab_transform)

    model, _optimizer = load_model(
        input_size=len(vocab_transform["ocr"]),
        hidden_size=hidden_size,
        output_size=len(vocab_transform["gs"]),
        dropout=dropout,
        max_len=max_len,
        teacher_forcing_ratio=teacher_forcing_ratio,
        model_path=model_path,
    )
    model.to(device)

    if calculate_loss:
        logger.info("Calculating loss on dataset without duplicates")
        test_dataloader = create_test_dataloader(
            test_df=test,
            max_len=max_len,
            hidden_size=hidden_size,
            batch_size=batch_size,
            text_transform=text_transform,
            look_up_bert_vectors=True,
            bert_vectors_file=bert_vectors_file,
        )
        test_loss = validate_model(model, test_dataloader, device)
        test_log = pd.DataFrame({"test_loss": [test_loss]})
        test_log.to_csv(test_log_file)

    logger.info("Loading the test dataset")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _train_path, test_path = extract_icdar_data(tmp_dir, raw_dataset)
        logger.debug(f"Test data is in {test_path}")

        data_test, _md_test = generate_data(test_path)
        logger.info(f"Found {len(data_test)} texts in test data")

    # Model for calculating bert vectors on the fly
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_dir)
    bert_model.eval()
    bert_model = bert_model.to(device=device)

    predict_and_save(
        in_file=perfect_detection_in,
        data_test=data_test,
        max_len=max_len,
        hidden_size=hidden_size,
        tokenizer=tokenizer,
        bert_model=bert_model,
        batch_size=batch_size,
        text_transform=text_transform,
        vocab_transform=vocab_transform,
        model=model,
        out_file=perfect_detection_out,
    )

    predict_and_save(
        in_file=predicted_detection_in,
        data_test=data_test,
        max_len=max_len,
        hidden_size=hidden_size,
        tokenizer=tokenizer,
        bert_model=bert_model,
        batch_size=batch_size,
        text_transform=text_transform,
        vocab_transform=vocab_transform,
        model=model,
        out_file=predicted_detection_out,
    )


if __name__ == "__main__":
    typer.run(predict_error_correction)

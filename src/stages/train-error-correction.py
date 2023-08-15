from pathlib import Path

import pandas as pd
import torch
import typer
from loguru import logger
from ocrpostcorrection.bert_vectors_correction_data import (
    BertVectorsCorrectionDataset,
    collate_fn,
    train_model,
)
from ocrpostcorrection.error_correction import (
    SimpleCorrectionSeq2seq,
    generate_vocabs,
    get_text_transform,
)
from ocrpostcorrection.utils import set_seed
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_error_correction(
    seed: Annotated[int, typer.Option()],
    dataset: Annotated[Path, file_in_option],
    bert_vectors_file: Annotated[Path, file_in_option],
    max_len: Annotated[int, typer.Option()],
    batch_size: Annotated[int, typer.Option()],
    hidden_size: Annotated[int, typer.Option()],
    dropout: Annotated[float, typer.Option()],
    teacher_forcing_ratio: Annotated[float, typer.Option()],
    num_epochs: Annotated[int, typer.Option()],
    valid_niter: Annotated[int, typer.Option()],
    max_num_patience: Annotated[int, typer.Option()],
    max_num_trial: Annotated[int, typer.Option()],
    lr_decay: Annotated[float, typer.Option()],
    model_save_path: Annotated[Path, file_out_option],
    train_log: Annotated[Path, file_out_option],
) -> None:
    set_seed(seed)

    data = pd.read_csv(dataset)
    data = data.fillna("")

    train = data.query('dataset == "train"')
    val = data.query('dataset == "val"')

    vocab_transform = generate_vocabs(train)
    text_transform = get_text_transform(vocab_transform)

    train_dataset = BertVectorsCorrectionDataset(
        train, bert_vectors_file, split_name="train", max_len=max_len
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn(text_transform),
    )

    val_dataset = BertVectorsCorrectionDataset(
        val, bert_vectors_file, split_name="val", max_len=max_len
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn(text_transform)
    )

    logger.info(f"# train samples: {len(train_dataset)}")
    logger.info(f"# val samples: {len(val_dataset)}")

    out_dir = Path(model_save_path).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    out_dir = Path(train_log).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    model = SimpleCorrectionSeq2seq(
        input_size=len(vocab_transform["ocr"]),
        hidden_size=hidden_size,
        output_size=len(vocab_transform["gs"]),
        dropout=dropout,
        max_length=max_len,
        teacher_forcing_ratio=teacher_forcing_ratio,
        device=device,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    log_data = train_model(
        train_dl=train_dataloader,
        val_dl=val_dataloader,
        model=model,
        optimizer=optimizer,
        model_save_path=model_save_path,
        num_epochs=num_epochs,
        valid_niter=valid_niter,
        max_num_patience=max_num_patience,
        max_num_trial=max_num_trial,
        lr_decay=lr_decay,
        device=device,
    )
    log_data.to_csv(train_log)


if __name__ == "__main__":
    typer.run(train_error_correction)

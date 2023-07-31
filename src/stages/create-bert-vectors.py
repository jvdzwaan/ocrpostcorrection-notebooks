from pathlib import Path

import pandas as pd
import torch
import typer
from datasets import Dataset, DatasetDict
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, DataCollatorWithPadding, set_seed
from typing_extensions import Annotated

from common.option_types import dir_in_option, file_in_option, file_out_option

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_bert_vectors_for_split(
    split: str,
    model: BertModel,
    collator: DataCollatorWithPadding,
    dataset: Dataset,
    out_dir: Path,
    batch_size: int = 8,
) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    out_path = out_dir / split
    out_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving vectors for {split} in {out_path}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch.to(device=device)

            output = model(**batch)

            samples = output["pooler_output"].detach().cpu()
            out_file = out_path / f"bert_vectors_{i}.pt"
            torch.save(samples, out_file)

            del samples
            del output
            del batch
            torch.cuda.empty_cache()


def create_bert_vectors(
    seed: Annotated[int, typer.Option()],
    dataset_in: Annotated[Path, file_in_option],
    model_dir: Annotated[Path, dir_in_option],
    model_name: Annotated[str, typer.Option()],
    batch_size: Annotated[int, typer.Option()],
    out_dir: Annotated[Path, file_out_option],
) -> None:
    logger.info("Creating BERT vectors")
    set_seed(seed)

    data = pd.read_csv(dataset_in, index_col=0)
    data.fillna("", inplace=True)
    train_data = data[data.dataset == "train"].copy()
    val_data = data[data.dataset == "val"].copy()
    test_data = data[data.dataset == "test"].copy()

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_data.ocr.to_frame()),
            "val": Dataset.from_pandas(val_data.ocr.to_frame()),
            "test": Dataset.from_pandas(test_data.ocr.to_frame()),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_dataset = dataset.map(
        lambda sample: tokenizer(sample["ocr"], truncation=True),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["ocr", "__index_level_0__"])

    model = BertModel.from_pretrained(model_dir)
    model.eval()
    model = model.to(device=device)

    collator = DataCollatorWithPadding(tokenizer)

    for split_name in ("train", "val", "test"):
        logger.info(f"Creating BERT vectors for {split_name}")
        create_bert_vectors_for_split(
            split=split_name,
            model=model,
            collator=collator,
            dataset=tokenized_dataset[split_name],
            batch_size=batch_size,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    typer.run(create_bert_vectors)

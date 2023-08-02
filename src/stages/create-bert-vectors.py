from pathlib import Path

import h5py
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
    model: BertModel,
    collator: DataCollatorWithPadding,
    dataset: Dataset,
    dset: h5py.Dataset,
    batch_size: int = 8,
) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    with torch.no_grad():
        index = 0
        for batch in tqdm(dataloader):
            batch.to(device=device)

            output = model(**batch)

            samples = output["pooler_output"].detach().cpu()
            dset[index : index + samples.size(0)] = samples

            index += samples.size(0)

            del samples
            del output
            del batch
            torch.cuda.empty_cache()


def get_hidden_size(model: BertModel) -> int:
    batch = {
        "input_ids": torch.tensor([[101, 119, 102]]),
        "token_type_ids": torch.tensor([[0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    output = model(**batch)
    samples = output["pooler_output"].detach().cpu()

    return int(samples.size(-1))


def create_bert_vectors(
    seed: Annotated[int, typer.Option()],
    dataset_in: Annotated[Path, file_in_option],
    model_dir: Annotated[Path, dir_in_option],
    model_name: Annotated[str, typer.Option()],
    batch_size: Annotated[int, typer.Option()],
    out_file: Annotated[Path, file_out_option],
) -> None:
    logger.info("Creating BERT vectors")
    set_seed(seed)

    splits = ("train", "val", "test")

    data = pd.read_csv(dataset_in, index_col=0)
    data.fillna("", inplace=True)

    sizes = {}
    for name in splits:
        num = (data.dataset == name).sum()
        sizes[name] = num

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
    hidden_size = get_hidden_size(model)
    model = model.to(device=device)

    collator = DataCollatorWithPadding(tokenizer)

    out_file.parent.mkdir(exist_ok=True, parents=True)

    with h5py.File(out_file, "w") as f:
        for split_name in splits:
            logger.info(f"Creating BERT vectors for {split_name}")
            dset = f.create_dataset(
                split_name, (sizes[split_name], hidden_size), dtype="f"
            )
            create_bert_vectors_for_split(
                model=model,
                collator=collator,
                dataset=tokenized_dataset[split_name],
                batch_size=batch_size,
                dset=dset,
            )


if __name__ == "__main__":
    typer.run(create_bert_vectors)

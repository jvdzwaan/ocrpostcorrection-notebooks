from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Text

import pandas as pd
import torch
import yaml
from loguru import logger
from ocrpostcorrection.error_correction import (
    SimpleCorrectionDataset,
    SimpleCorrectionSeq2seq,
    collate_fn,
    generate_vocabs,
    get_text_transform,
    validate_model,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    train_dl: DataLoader[int],
    val_dl: DataLoader[int],
    model: SimpleCorrectionSeq2seq,
    optimizer: torch.optim.optimizer.Optimizer,
    num_epochs: int = 5,
    valid_niter: int = 5000,
    model_save_path: Text = "model.rar",
    train_log_file: Text = "train-log.csv",
    max_num_patience: int = 5,
    max_num_trial: int = 5,
    lr_decay: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> None:
    num_iter = 0
    report_loss = 0
    report_examples = 0
    val_loss_hist: List[float] = []
    train_loss_hist: List[float] = []
    num_trial = 0
    patience = 0

    model.train()

    for epoch in range(1, num_epochs + 1):
        cum_loss = 0
        cum_examples = 0

        for src, tgt in tqdm(train_dl):
            num_iter += 1

            batch_size = src.size(1)

            src = src.to(device)
            tgt = tgt.to(device)
            encoder_hidden = model.encoder.initHidden(
                batch_size=batch_size, device=device
            )

            example_losses, _ = model(src, encoder_hidden, tgt)
            example_losses = -example_losses
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            bl = batch_loss.item()
            report_loss += bl
            report_examples += batch_size

            cum_loss += bl
            cum_examples += batch_size

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if num_iter % valid_niter == 0:
                val_loss = validate_model(model, val_dl, device)
                train_loss = report_loss / report_examples
                logger.info(
                    f"Epoch {epoch}, iter {num_iter}, avg. train loss "
                    + f"{train_loss}, avg. val loss {val_loss}"
                )

                report_loss = 0
                report_examples = 0

                better_model = len(val_loss_hist) == 0 or val_loss < min(val_loss_hist)
                if better_model:
                    logger.info(f"Saving model and optimizer to {model_save_path}")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        model_save_path,
                    )
                elif patience < max_num_patience:
                    patience += 1
                    logger.info(f"Hit patience {patience}")

                    if patience == max_num_patience:
                        num_trial += 1
                        logger.info(f"Hit #{num_trial} trial")
                        if num_trial == max_num_trial:
                            logger.info("Early stop!")
                            sys.exit()

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]["lr"] * lr_decay
                        logger.info(
                            f"Load best model so far and decay learning rate to {lr}"
                        )

                        # load model
                        checkpoint = torch.load(model_save_path)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                        model = model.to(device)

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        # reset patience
                        patience = 0

                val_loss_hist.append(val_loss)
                train_loss_hist.append(train_loss)

                # Save train log
                df = pd.DataFrame(
                    {"train_loss": train_loss_hist, "val_loss": val_loss_hist}
                )
                df.to_csv(train_log_file)


def train_error_correction(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    data = pd.read_csv(
        config["create-error-correction-dataset"]["dataset"], index_col=0
    )
    data = data.fillna("")
    data = data.head(2000)

    train = data.query('dataset == "train"')
    val = data.query('dataset == "val"')

    vocab_transform = generate_vocabs(train)
    text_transform = get_text_transform(vocab_transform)

    max_len = config["train-error-correction"]["max-len"]
    batch_size = config["train-error-correction"]["batch-size"]

    train_dataset = SimpleCorrectionDataset(train, max_len=max_len)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn(text_transform),
    )

    val_dataset = SimpleCorrectionDataset(val, max_len=max_len)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn(text_transform)
    )

    logger.info(f"# train samples: {len(train_dataset)}")
    logger.info(f"# val samples: {len(val_dataset)}")

    out_dir = Path(config["train-error-correction"]["model-save-path"]).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    out_dir = Path(config["train-error-correction"]["train-log"]).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    model = SimpleCorrectionSeq2seq(
        input_size=len(vocab_transform["ocr"]),
        hidden_size=config["train-error-correction"]["hidden-size"],
        output_size=len(vocab_transform["gs"]),
        dropout=config["train-error-correction"]["dropout"],
        max_length=max_len,
        teacher_forcing_ratio=config["train-error-correction"]["teacher-forcing-ratio"],
        device=device,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_model(
        train_dl=train_dataloader,
        val_dl=val_dataloader,
        model=model,
        optimizer=optimizer,
        model_save_path=config["train-error-correction"]["model-save-path"],
        train_log_file=config["train-error-correction"]["train-log"],
        num_epochs=config["train-error-correction"]["num-epochs"],
        valid_niter=config["train-error-correction"]["valid-niter"],
        max_num_patience=config["train-error-correction"]["max-num-patience"],
        max_num_trial=config["train-error-correction"]["max-num-trial"],
        lr_decay=config["train-error-correction"]["lr-decay"],
        device=device,
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_error_correction(args.config)
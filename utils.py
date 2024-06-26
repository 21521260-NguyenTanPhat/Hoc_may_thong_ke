import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from torch.utils.tensorboard.writer import SummaryWriter

from metrics import AspectF1Score, PolarityF1Score

from typing import Literal, Callable, Any
from tqdm import tqdm

def forward(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    batch,
    device: torch.device = "cuda",
):
    text, aspects, polarities = batch["text"], batch["aspects"], batch["polarities"]

    x = tokenizer(text, padding=True, return_tensors="pt", return_token_type_ids=False)
    x = {k: v.to(device) for k, v in x.items()}
    aspects = aspects.to(device)
    polarities = polarities.to(device)

    pred = model(**x)
    return pred, (aspects, polarities)

def train(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    train_set: Dataset,
    val_set: Dataset,
    batch_size: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metrics: dict[str, Callable[[Any, Any], torch.FloatTensor]],
    num_grad_accumulate: int = 1,
    device: torch.device = "cuda",
    enable_record_loss: bool = True
):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)
    writer = SummaryWriter(log_dir="runs")

    for ith_epoch in range(1, epochs+1):
        train_loop(
            tokenizer=tokenizer,
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            ith_epoch=ith_epoch,
            num_grad_accumulate=num_grad_accumulate,
            device=device
        )
        eval_loop(
            tokenizer=tokenizer,
            model=model,
            val_loader=val_loader,
            metrics=metrics,
            writer=writer,
            ith_epoch=ith_epoch,
            device=device
        )
        if enable_record_loss:
            record_loss(
                tokenizer=tokenizer,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                writer=writer,
                ith_epoch=ith_epoch,
                device=device
            )
                    

def eval(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    val_loader: DataLoader,
    metric_name: str,
    metric_fn: Callable[[Any, Any], torch.FloatTensor],
    ith_epoch: int,
    reduction: Literal["mean", "none"] = "mean",
    device: torch.device = "cuda"
):
    model = model.to(device)
    model.eval() 
    scores = []
    iter = tqdm(
        val_loader,
        desc=f"Epoch {ith_epoch} - {metric_name} Evaluation",
        total=len(val_loader),
        colour="blue"
    )

    for batch in iter:
        pred, truth = forward(
            tokenizer=tokenizer,
            model=model,
            batch=batch,
            device=device
        )

        score = metric_fn(pred, truth)
        scores.append(score)
    
    result = torch.stack(scores)
    if reduction == "mean":
        result = result.nanmean().item()
    return result


def record_loss(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable[[Any, Any], torch.FloatTensor],
    writer: SummaryWriter,
    ith_epoch: int,
    device: torch.device = "cuda"
):
    scalars = {}
    with torch.no_grad():
        for name, dset in {"Train": train_loader, "Test": val_loader}.items():
            losses = []
            iter = tqdm(
                dset,
                desc=f"Epoch {ith_epoch} - {name} Loss",
                total=len(dset),
                colour="blue"
            )
            for batch in iter:
                pred, truth = forward(
                    tokenizer=tokenizer,
                    model=model,
                    batch=batch,
                    device=device
                )

                loss = loss_fn(pred, truth)
                losses.append(loss)
            avg_loss = torch.stack(losses).mean().item()
            scalars[name] = avg_loss
    
    writer.add_scalars(
        main_tag="Losses",
        tag_scalar_dict=scalars,
        global_step=ith_epoch
    )


def train_loop(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    ith_epoch: int,
    num_grad_accumulate: int = 1,
    device: torch.device = "cuda",
):
    model = model.to(device)
    model.train()
    n_iter = len(train_loader)
    iter = tqdm(
        enumerate(train_loader, start=1),
        desc=f"Epoch {ith_epoch}",
        total=n_iter,
        colour="green"
    )
    optimizer.zero_grad()

    for i, batch in iter:
        pred, truth = forward(
            tokenizer=tokenizer,
            model=model,
            batch=batch,
            device=device
        )

        loss = loss_fn(pred, truth)
        loss.backward()

        if i % num_grad_accumulate == 0 or i == n_iter:
            optimizer.step()
            optimizer.zero_grad()

def eval_loop(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    val_loader: DataLoader,
    metrics: dict[str, Callable[[Any, Any], torch.FloatTensor]],
    writer: SummaryWriter,
    ith_epoch: int,
    device: torch.device = "cuda"
):
    scalars = {}

    with torch.no_grad():
        for name, fn in metrics.items():
            score = eval(
                tokenizer=tokenizer,
                model=model,
                val_loader=val_loader,
                metric_name=name,
                metric_fn=fn,
                ith_epoch=ith_epoch,
                device=device
            )
            scalars[name] = score

    writer.add_scalars(
        main_tag="Metrics",
        tag_scalar_dict=scalars,
        global_step=ith_epoch
    )


def absa_eval(
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    test_set: Dataset,
    reduction: Literal["none", "mean"] = "none",
    batch_size: int = 64
):
    a_f1 = eval(
        tokenizer=tokenizer,
        model=model,
        val_loader=DataLoader(test_set, batch_size),
        metric_name="Aspect F1 Score",
        metric_fn=AspectF1Score(aspect_thresholds=0.5, reduction="none"),
        reduction="none",
        ith_epoch=1
    ).mean(dim=0)

    p_f1 = eval(
        tokenizer=tokenizer,
        model=model,
        val_loader=DataLoader(test_set, batch_size),
        metric_name="Polarity F1 Score",
        metric_fn=PolarityF1Score(reduction="none"),
        reduction="none",
        ith_epoch=1
    ).nanmean(dim=0)

    if reduction == "mean":
        a_f1 = torch.round(a_f1.nanmean(), decimals=5)
        p_f1 = torch.round(p_f1.nanmean(), decimals=5)
        result = {
            "Aspect F1": a_f1.item(),
            "Polarity F1": p_f1.item()
        }
    elif reduction == "none":
        a_f1 = torch.round(a_f1, decimals=5)
        p_f1 = torch.round(p_f1, decimals=5)
        result = {
            a: {
                "Aspect F1": a_f1[i].item(),
                "Polarity F1": p_f1[i].item() if a != "OTHERS" else torch.nan
            }
            for i, a in enumerate(["CAMERA", "FEATURES", "BATTERY", "PRICE", "GENERAL", "SER&ACC", "PERFORMANCE", "SCREEN", "DESIGN", "STORAGE", "OTHERS"])
        }
    return result
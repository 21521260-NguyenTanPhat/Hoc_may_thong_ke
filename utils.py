import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from typing import Callable, Any, Any
from tqdm import tqdm

def bold(text: str):
    return f"\033[1m{text}\033[0m"

def blue(text: str):
    return f"\033[1;34m{text}\033[0m"

def evaluate(
    model: nn.Module,
    score_fn: Callable[[Any, Any], torch.Tensor],
    data: DataLoader,
    device: torch.device = "cuda",
    metric_name: str = ""
):
    scores = []
    model.to(device=device).eval()
    with torch.no_grad():
        for batch, (x, y) in tqdm(
            enumerate(iter(data), start=1),
            total=len(data),
            desc=bold(metric_name + " Evaluation"),
            colour="blue"
        ):
            x = x.to(device=device)
            y = (y[0].to(device=device), y[1].to(device=device))

            y_hat = model(x)
            score = score_fn(y_hat, y)
            scores.append(score)
    
        result = torch.tensor(scores).mean()
    return result.item()

def train(
    model: nn.Module,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    train_set: Dataset,
    validation_set: Dataset | None = None,
    batch_size: int = 32,
    epochs: int = 5,
    metrics: dict[str, Callable[[Any, Any], torch.Tensor]] = {},
    device: torch.device = "cuda",
    log_dir: str = "runs",
    log_epoch_interval: int = 1
):
    torch.set_default_device(device)
    model.to(device=device)
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=torch.Generator(device)
    )

    val_loader = DataLoader(
        validation_set, 
        batch_size=batch_size, 
        shuffle=False,
        generator=torch.Generator(device)
    )
    _metrics = metrics.copy()
    _metrics["Loss"] = loss_fn
    writer = SummaryWriter(log_dir)

    for epoch in torch.arange(1, epochs+1):
        model.train()
        
        it = tqdm(
            enumerate(iter(train_loader), start=1),
            total=len(train_loader),
            colour="green"
        )
        for batch, (x, y) in it:
            x = x.to(device=device)
            y = (y[0].to(device=device), y[1].to(device=device))
            optimizer.zero_grad()

            y_hat = model(x)

            loss = loss_fn(y_hat, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)

            it.set_description(blue(f"Epoch {epoch}") + bold(f" | Training Loss {loss.item():.4f}"))
            optimizer.step()

        if epoch % log_epoch_interval == 0:
            for metric, fn in _metrics.items():
                scalars = {}

                # scalars["Train"] = evaluate(model, fn, train_loader, device, metric)
                if validation_set is not None:
                    scalars["Validation"] = evaluate(model, fn, val_loader, device, metric)
                
                writer.add_scalars(
                    metric, scalars,
                    global_step=epoch
                )
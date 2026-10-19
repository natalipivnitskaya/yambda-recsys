"""Training loop and full-catalog evaluation for two-tower models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from recsys.metrics import evaluate


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return batch


@torch.no_grad()
def recommend(model: torch.nn.Module, loader: DataLoader, topk: int, device: torch.device) -> Dict[int, List[int]]:
    """Top-``topk`` items by dot product for every user in ``loader``."""
    model.eval()
    recommendations = {}
    for batch in loader:
        top_items = torch.topk(model(to_device(batch, device)), k=topk, dim=1).indices.cpu().tolist()
        recommendations.update(zip(batch["uid"].tolist(), top_items))
    return recommendations


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, catalog_size: int, topk: int, device: torch.device
) -> Dict[str, float]:
    recommendations = recommend(model, loader, topk, device)
    labels = loader.dataset.labels
    targets = {uid: labels[uid] for uid in recommendations}
    return evaluate(targets, recommendations, catalog_size, topk)


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    catalog_size: int,
    topk: int = 100,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """Train for ``num_epochs``, evaluating after each one. Returns per-epoch logs."""
    device = device or next(model.parameters()).device
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}")
        for step, batch in enumerate(progress, start=1):
            loss = model(to_device(batch, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 100 == 0:
                progress.set_postfix(loss=f"{total_loss / step:.4f}")

        metrics = evaluate_model(model, eval_loader, catalog_size, topk, device)
        log = {"epoch": epoch, "loss": total_loss / len(train_loader), **metrics}
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in log.items()))
        history.append(log)
    return history

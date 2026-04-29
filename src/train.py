import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, get_linear_schedule_with_warmup

from src.data_utils import AnswersDataset
from src.model_utils import build_model

import wandb

import torch.nn.functional as F
import os
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    # ---- DDP setup ----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = local_rank == 0  # only rank 0 logs / saves

    train_path = "data/train_new_data_final.csv"
    val_path = "data/validation_new_data_final.csv"
    test_path = "data/test_new_data_final.csv"
    model_name = "allenai/longformer-base-4096"
    max_len = 4096

    config_defaults = {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "model_name": model_name,
        "max_len": max_len,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "epochs": 50,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "fusion_mode": "concat",
        "device": str(device),
        "patience": 3,
        "min_delta": 1e-4,
        "world_size": world_size,
    }

    # only rank 0 initialises wandb
    if is_main:
        wandb.init(project="research_longformer", config=config_defaults)
        config = wandb.config
    else:
        # other ranks still need the hyperparameters — read from defaults
        class _Cfg:
            pass
        config = _Cfg()
        for k, v in config_defaults.items():
            setattr(config, k, v)

    train_bs = config.train_batch_size
    val_bs = config.val_batch_size
    epochs = config.epochs
    lr = config.learning_rate
    warmup_ratio = config.warmup_ratio
    weight_decay = config.weight_decay
    patience = config.patience
    min_delta = config.min_delta

    # ----- Early stopping tracking -----
    best_val_loss = float("inf")
    best_val_acc_at_best = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_ckpt_path = "best_model.pt"

    train_df = pd.read_csv(train_path).dropna(subset=["ResponseText.x", "ground_truth"]).copy()
    val_df = pd.read_csv(val_path).dropna(subset=["ResponseText.x", "ground_truth"]).copy()
    test_df = pd.read_csv(test_path).dropna(subset=["ResponseText.x", "ground_truth"]).copy()

    train_df["ground_truth"] = train_df["ground_truth"].astype(int)
    val_df["ground_truth"] = val_df["ground_truth"].astype(int)
    test_df["ground_truth"] = test_df["ground_truth"].astype(int)

    min_label = int(train_df["ground_truth"].min())
    if min_label != 0:
        train_df["ground_truth"] -= min_label
        val_df["ground_truth"] -= min_label
        test_df["ground_truth"] -= min_label

    num_labels = int(train_df["ground_truth"].nunique())

    tok = LongformerTokenizerFast.from_pretrained(model_name)

    train_ds = AnswersDataset(
        train_df, tok,
        text_col="ResponseText.x", model_col="Model_Answer",
        label_col="ground_truth", max_len=max_len, fusion_mode="concat"
    )
    val_ds = AnswersDataset(
        val_df, tok,
        text_col="ResponseText.x", model_col="Model_Answer",
        label_col="ground_truth", max_len=max_len, fusion_mode="concat"
    )
    test_ds = AnswersDataset(
        test_df, tok,
        text_col="ResponseText.x", model_col="Model_Answer",
        label_col="ground_truth", max_len=max_len, fusion_mode="concat"
    )

    # ---- DDP samplers ----
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=train_bs, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=val_bs, sampler=val_sampler)
    test_loader = DataLoader(test_ds, batch_size=val_bs, sampler=test_sampler)

    # ---- Model ----
    model = build_model(model_name, num_labels=num_labels, fusion_mode="concat").to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if is_main:
        wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)  # shuffle differently each epoch across GPUs

        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                global_attention_mask=batch.get("global_attention_mask", None),
            )
            loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()

            if is_main:
                wandb.log({"train_batch_loss": loss.item(), "learning_rate": optim.param_groups[0]["lr"]})

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_train_loss / max(num_train_batches, 1)

        # ----- Validation -----
        model.eval()
        total = torch.tensor(0, device=device)
        correct = torch.tensor(0, device=device)
        val_loss_total = torch.tensor(0.0, device=device)
        num_val_batches = 0

        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    global_attention_mask=batch.get("global_attention_mask", None),
                )
                val_loss_total += out["loss"]
                num_val_batches += 1
                preds = out["logits"].argmax(dim=-1)
                correct += (preds == batch["labels"]).sum()
                total += batch["labels"].size(0)

        # gather metrics across all GPUs
        dist.all_reduce(val_loss_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        val_loss = (val_loss_total / max(num_val_batches * world_size, 1)).item()
        val_acc = (correct / total).item() if total.item() else 0.0

        if is_main:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss, "val_acc": val_acc})
            print(f"epoch {epoch}  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_val_acc_at_best = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.module.state_dict(), best_ckpt_path)
                print(f"  ✓ val_loss improved to {val_loss:.4f} — saved checkpoint")
            else:
                epochs_no_improve += 1
                print(f"  ✗ no improvement in val_loss ({epochs_no_improve}/{patience})")

        # broadcast early-stopping decision from rank 0 to all ranks
        stop_signal = torch.tensor(1 if (is_main and epochs_no_improve >= patience) else 0, device=device)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            if is_main:
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc_at_best:.4f}")
            break

    # ----- Test (rank 0 only) -----
    if is_main:
        print(f"\nLoading best model from epoch {best_epoch} (val_loss={best_val_loss:.4f}) for testing...")
        state_dict = torch.load(best_ckpt_path, map_location=device)
        model.module.load_state_dict(state_dict)
        model.eval()

        total = 0
        correct = 0
        rows_for_csv = []

        # use a plain (non-distributed) loader for test so rank 0 sees all samples
        plain_test_loader = DataLoader(test_ds, batch_size=val_bs, shuffle=False)

        with torch.inference_mode():
            for batch in plain_test_loader:
                batch_idx = batch["idx"].cpu().numpy()
                batch = {k: v.to(device) for k, v in batch.items()}

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    global_attention_mask=batch.get("global_attention_mask", None),
                )

                logits = out["logits"]
                probs = F.softmax(logits, dim=-1)
                conf, preds = probs.max(dim=-1)

                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

                preds_np = preds.cpu().numpy()
                labels_np = batch["labels"].cpu().numpy()
                conf_np = conf.cpu().numpy()

                for i, idx in enumerate(batch_idx):
                    rows_for_csv.append({
                        "response_id": test_df.iloc[idx]["ResponseId"],
                        "university": test_df.iloc[idx]["UNIV"],
                        "Question_type": test_df.iloc[idx]["TaskPrompt"],
                        "true_label": int(labels_np[i]),
                        "predicted_label": int(preds_np[i]),
                        "confidence": float(conf_np[i]),
                    })

        test_acc = correct / total if total else 0.0
        print(f"\nFinal TEST accuracy: {test_acc:.4f}")

        preds_out_path = "higher_bs_longformer_output_a+m_final.csv"
        pd.DataFrame(rows_for_csv).to_csv(preds_out_path, index=False)
        print("Saved predictions to:", os.path.abspath(preds_out_path))

        wandb.log({"test_acc": test_acc, "best_epoch": best_epoch, "best_val_loss": best_val_loss})
        wandb.save(preds_out_path)
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


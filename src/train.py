from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ultralytics.utils.loss import v8DetectionLoss
from tqdm import tqdm
from dataset import WeaponsDataset, collate_fn
import os
import tarfile
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics.utils.nms import non_max_suppression
import boto3
import argparse


# init DDP
def init_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, None


def get_device(local_rank, use_ddp):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}" if use_ddp else "cuda:0")
    return torch.device("cpu")


def download_data(download_path):
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    if rank == 0:
        print("Downloading data...")
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.download_file('agression-model', 'weapons_768.tar', f'{download_path}/weapons_768.tar')
        print("Extracting...")

        with tarfile.open(f'{download_path}/weapons_768.tar', 'r:*') as tar:
            tar.extractall(path=download_path)
        os.remove(f'{download_path}/weapons_768.tar')
        print("Done extracting data")
    if is_ddp:
        dist.barrier()


# Create dataloader
def get_dataloader(data_path, batch_size=32, num_workers=4, distributed=False):
    dataset = WeaponsDataset(data_path)

    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
        )
        shuffle = False

    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True,
    )
    return loader, sampler


def get_model(device):

    model = YOLO('yolo11s.pt').model
    print("YOLOv11 Loaded Successfully!")
    num_classes = 3
    detect_layer = model.model[23]
    model.model[-1].nc = 3
    model.nc = 3

    for i in range(3):
        old_conv = detect_layer.cv3[i][2]
        in_ch = old_conv.in_channels
        print(f"cv3.{i}.2 — in_ch: {in_ch}, old out_ch: {old_conv.out_channels}")

        detect_layer.cv3[i][2] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        nn.init.normal_(detect_layer.cv3[i][2].weight, std=0.01)
        nn.init.constant_(detect_layer.cv3[i][2].bias, 0)

    for i in range(3):
        conv = detect_layer.cv3[i][2]
        print(f"cv3.{i}.2 → in: {conv.in_channels}, out: {conv.out_channels}")

    for name, module in model.named_modules():
        if hasattr(module, 'nc'):
            print(name, module.nc)
    return model


def save_checkpoint(model, optimizer, epoch, path):
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    if rank == 0:
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        print("Checkpoint saved")
    if is_ddp:
        dist.barrier()

def save_model(model, file_path):
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    if rank == 0:
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, file_path)
    if is_ddp:
        dist.barrier()


def load_checkpoint(model, optimizer, path):
    # load on CPU first to avoid GPU memory issues
    checkpoint = torch.load(path, map_location='cpu')

    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def validate_loss(model, criterion, val_loader, device):
    model.eval()
    total_loss = class_loss = box_loss = dfl_loss = 0.0
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["img"].to(device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor)
                     else v for k, v in batch.items()}

            preds = model(images)
            nmp_preds = non_max_suppression(preds, 0.25, 0.45)

            formatted_preds = []
            formatted_targets = []

            image_height, image_width = images.shape[2:]
            for i in range(len(images)):
                pred = nmp_preds[i]

                if pred is None or len(pred) == 0:
                    formatted_preds.append({
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64,
                                              device=device),
                    })
                else:
                    boxes = pred[:, :4]
                    scores = pred[:, 4]
                    labels = pred[:, 5].long()

                    formatted_preds.append({
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    })

                mask = batch["batch_idx"] == i
                gt_boxes = batch["bboxes"][mask].to(device)
                gt_labels = batch["cls"][mask].long().view(-1).to(device)
                x, y, w_box, h_box = gt_boxes.unbind(-1)

                gt_boxes = torch.stack([
                    x - w_box / 2,
                    y - h_box / 2,
                    x + w_box / 2,
                    y + h_box / 2
                ], dim=-1)

                # scale to pixels
                gt_boxes[:, [0, 2]] *= image_width
                gt_boxes[:, [1, 3]] *= image_height

                formatted_targets.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                })
            for p in formatted_preds:
                print(p["labels"].shape)

            for t in formatted_targets:
                print(t["labels"].shape)

            metric.update(formatted_preds, formatted_targets)

            loss, loss_items = criterion(preds, batch)
            loss = loss.sum()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += sum(loss_items).item()
            box_loss += loss_items[0].item()
            class_loss += loss_items[1].item()
            dfl_loss += loss_items[2].item()

    n = len(val_loader)
    res = metric.compute()

    return {
        "loss": total_loss / n,
        "box": box_loss / n,
        "cls": class_loss / n,
        "dfl": dfl_loss / n,
        "mAP50": res["map_50"].item(),
        "mAP50_95": res["map"].item(),
    }


def train_weapon_yolo(model, optimizer, scheduler, criterion, train_loader, train_sampler,
                      val_loader, n_epochs, device, checkpoint_dir, model_dir):
    best_map = 0.0
    for epoch in range(n_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = class_loss = box_loss = dfl_loss = 0.0
        # test = next(iter(train_loader))
        pbar = tqdm(
            # [test],
            train_loader,
            desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            images = batch["img"]

            y_pred = model(images)
            loss, loss_items = criterion(y_pred, batch)
            loss = loss.sum()

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf detected for loss, skipping batch")
                continue

            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += sum(loss_items).item()
            box_loss += loss_items[0].item()
            class_loss += loss_items[1].item()
            dfl_loss += loss_items[2].item()

            pbar.set_postfix({
                "loss": f"{sum(loss_items).item():.4f}",
                "cls":  f"{loss_items[1].item():.4f}",
                "box":  f"{loss_items[0].item():.4f}",
                "dfl":  f"{loss_items[2].item():.4f}",
            })

        scheduler.step()

        save_checkpoint(model, optimizer, epoch,f"{checkpoint_dir}/epoch{epoch}.pt")

        n = len(train_loader)

        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"train loss: {total_loss/n:.4f} "
            f"(box:{box_loss/n:.4f} cls:{class_loss/n:.4f} dfl:{dfl_loss/n:.4f})"
        )

        val_losses = validate_loss(model, criterion, val_loader, device)

        print(
            f"Validation | "
            f"loss: {val_losses['loss']:.4f} "
            f"(box:{val_losses['box']:.4f} "
            f"cls:{val_losses['cls']:.4f} "
            f"dfl:{val_losses['dfl']:.4f})"
            f"mAP50: {val_losses['mAP50']:.4f} "
            f"mAP50_95: {val_losses['mAP50_95']:.4f}"
        )

        if val_losses["mAP50"] > best_map:
            best_map = val_losses["mAP50"]
            save_model(model, f"{model_dir}/best_mAP50.pt")
            print("Saved BEST model with mAP50")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--model-dir', type=str, default='./model')
    parser.add_argument('--data-dir', type=str, default='./data')
    return parser.parse_args()


if __name__ == '__main__':
    is_dist, local_rank = init_ddp()

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    #download_data(args.data_dir)


    device = get_device(local_rank, is_dist)

    model = get_model(device).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_loader, train_sampler = get_dataloader(
        f"{args.data_dir}/train",
        args.batch_size,
        distributed=is_dist,
    )
    val_loader, val_sampler = get_dataloader(
        f"{args.data_dir}/valid",
        args.batch_size,
        distributed=False,
    )

    criterion = v8DetectionLoss(model)
    criterion.hyp = SimpleNamespace(box=7.5, cls=1.0, dfl=1.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    train_weapon_yolo(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        train_sampler,
        val_loader,
        args.epochs,
        device,
        args.checkpoint_dir,
        args.model_dir,
    )


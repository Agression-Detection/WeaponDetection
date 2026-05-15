import tarfile
from os.path import exists

from torchvision.ops import box_iou
import torch
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils.nms import non_max_suppression
from ultralytics import YOLO
import argparse
import os
import json

from model import get_model

def run_eval(model, test_loader, device, conf_threshold=0.25, iou_threshold=0.5):
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox")
    metric.reset()

    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    all_true = []
    all_pred = []

    print("Running inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):

            images = batch["img"].to(device)
            preds = model(images)
            preds = non_max_suppression(preds, conf_threshold, iou_threshold)

            formatted_preds = []
            formatted_targets = []

            B, _, H, W = images.shape

            for i in range(B):
                pred = preds[i]

                # -----------------------
                # Predictions
                # -----------------------
                if pred is None or len(pred) == 0:
                    pred_boxes = torch.zeros((0, 4), device=device)
                    pred_scores = torch.zeros((0,), device=device)
                    pred_labels = torch.zeros((0,), dtype=torch.long, device=device)
                else:
                    pred_boxes = pred[:, :4]
                    pred_scores = pred[:, 4]
                    pred_labels = pred[:, 5].long()

                formatted_preds.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels
                })

                # -----------------------
                # Ground truth
                # -----------------------
                mask = batch["batch_idx"] == i

                gt_boxes = batch["bboxes"][mask].to(device)
                gt_labels = batch["cls"][mask].long().view(-1).to(device)

                if len(gt_boxes) > 0:
                    x, y, w, h = gt_boxes.unbind(-1)

                    gt_boxes = torch.stack([
                        x - w / 2,
                        y - h / 2,
                        x + w / 2,
                        y + h / 2
                    ], dim=-1)

                    gt_boxes[:, [0, 2]] *= W
                    gt_boxes[:, [1, 3]] *= H
                else:
                    gt_boxes = torch.zeros((0, 4), device=device)

                formatted_targets.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels
                })

                # -----------------------
                # IoU matching (CORE FIX)
                # -----------------------
                if len(pred_boxes := formatted_preds[-1]["boxes"]) > 0 and len(gt_boxes) > 0:

                    ious = box_iou(pred_boxes.cpu(), gt_boxes.cpu())

                    matched_gt = set()

                    for pi in range(len(pred_boxes)):
                        best_iou, gi = torch.max(ious[pi], dim=0)

                        if best_iou >= 0.5 and gi.item() not in matched_gt:
                            matched_gt.add(gi.item())

                            true_cls = gt_labels[gi].item()
                            pred_cls = pred_labels[pi].item()

                            if true_cls == pred_cls:
                                class_metrics[true_cls]["tp"] += 1
                            else:
                                class_metrics[true_cls]["fn"] += 1
                                class_metrics[pred_cls]["fp"] += 1

                            all_true.append(true_cls)
                            all_pred.append(pred_cls)

                        else:
                            # unmatched prediction = FP
                            pred_cls = pred_labels[pi].item()
                            class_metrics[pred_cls]["fp"] += 1

                            all_true.append(-1)
                            all_pred.append(pred_cls)

                    # unmatched GT = FN
                    for gi in range(len(gt_boxes)):
                        if gi not in matched_gt:
                            true_cls = gt_labels[gi].item()
                            class_metrics[true_cls]["fn"] += 1

                            all_true.append(true_cls)
                            all_pred.append(-1)

                # update mAP
                metric.update(formatted_preds, formatted_targets)

    # -----------------------
    # mAP
    # -----------------------
    res = metric.compute()
    mAP50 = res["map_50"].item()
    mAP50_95 = res["map"].item()

    # -----------------------
    # F1 / confusion matrix
    # -----------------------
    valid = [(t, p) for t, p in zip(all_true, all_pred) if t != -1 and p != -1]

    if len(valid) > 0:
        y_true, y_pred = zip(*valid)

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
    else:
        f1_macro = f1_weighted = f1_micro = 0
        cm = None
        report = "No valid matches"

    # -----------------------
    # per-class metrics (FIXED)
    # -----------------------
    per_class_metrics = {}

    for class_id, m in class_metrics.items():
        tp = m["tp"]
        fp = m["fp"]
        fn = m["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        per_class_metrics[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
        "confusion_matrix": cm,
        "class_report": report,
        "per_class_metrics": per_class_metrics,
    }

def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    if cm is None:
        print("No confusion matrix to plot")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")
    plt.close()

def evaluate(model_path, test_loader, output_dir="output", conf_threshold=0.25, iou_threshold= 0.45):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    model = get_model(device, False, None)
    model.load_state_dict(torch.load("./model/best_model.pt", map_location=device))
    model.to(device)

    # Run testing
    metrics = run_eval(
        model,
        test_loader,
        device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"mAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50_95: {metrics['mAP50_95']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (micro): {metrics['f1_micro']:.4f}")
    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)

    class_names = ["Gun", "Knife", "Bomb"]  # Adjust to your weapon classes
    for class_id, class_metrics in metrics['per_class_metrics'].items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"\n{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1: {class_metrics['f1']:.4f}")
        print(f"  TP: {class_metrics['tp']}, FP: {class_metrics['fp']}, FN: {class_metrics['fn']}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(metrics['class_report'])

    # Plot confusion matrix
    if metrics['confusion_matrix'] is not None:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names,
            output_path=os.path.join(output_dir, "confusion_matrix.png")
        )

    # Save results
    results_file = os.path.join(output_dir, "test_results.json")
    results_dict = {
        "mAP50": metrics['mAP50'],
        "mAP50_95": metrics['mAP50_95'],
        "f1_macro": metrics['f1_macro'],
        "f1_weighted": metrics['f1_weighted'],
        "f1_micro": metrics['f1_micro'],
        "per_class_metrics": {
            str(k): v for k, v in metrics['per_class_metrics'].items()
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print("=" * 60)

if __name__ == '__main__':
    with tarfile.open(f'./model/model.tar.gz', 'r:*') as tar:
        tar.extractall(path='./model/')
    print("extracted data")
    os.makedirs("output", exist_ok=True)
    evaluate('./model/model.tar.gz', './data/test', 16, "output")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    model = get_model()
    # # model.load_state_dict(torch.load("model/best_mAP50.pt", map_location=device))
    # print(type(model))
    # model = YOLO("model/best_mAP50.pt")

    #model.val(data="data/data.yaml")

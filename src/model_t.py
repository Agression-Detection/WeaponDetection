import tarfile
from os.path import exists

import torch
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils.nms import non_max_suppression
import argparse
import os
import json

from train import get_model


# Assuming you have these from your training code
# from train import WeaponsDataset, collate_fn, get_model, v8DetectionLoss


def test_model(model, test_loader, device, conf_threshold=0.25, iou_threshold=0.45):
    """
    Test YOLO model on test set and compute metrics.

    Returns:
        metrics: dict with mAP50, mAP50_95, F1, confusion matrix, class metrics
    """
    model.eval()

    # Initialize metrics
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.reset()

    all_pred_labels = []
    all_true_labels = []
    all_confidence = []

    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        print("Running inference on test set...", flush=True)
        for batch_idx, batch in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(test_loader)}", flush=True)

            images = batch["img"].to(device)
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

            # Model prediction
            preds = model(images)
            nmp_preds = non_max_suppression(preds, conf_threshold, iou_threshold)

            # Format predictions for mAP computation
            formatted_preds = []
            formatted_targets = []

            image_height, image_width = images.shape[2:]

            for i in range(len(images)):
                pred = nmp_preds[i]

                # Format predictions
                if pred is None or len(pred) == 0:
                    formatted_preds.append({
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    })
                    pred_labels = []
                    pred_confs = []
                else:
                    boxes = pred[:, :4]
                    scores = pred[:, 4]
                    labels = pred[:, 5].long()

                    formatted_preds.append({
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    })
                    pred_labels = labels.cpu().numpy()
                    pred_confs = scores.cpu().numpy()

                # Format ground truth
                mask = batch_data["batch_idx"] == i
                gt_boxes = batch_data["bboxes"][mask].to(device)
                gt_labels = batch_data["cls"][mask].long().view(-1).to(device)

                x, y, w_box, h_box = gt_boxes.unbind(-1)

                gt_boxes = torch.stack([
                    x - w_box / 2,
                    y - h_box / 2,
                    x + w_box / 2,
                    y + h_box / 2
                ], dim=-1)

                # Scale to pixels
                gt_boxes[:, [0, 2]] *= image_width
                gt_boxes[:, [1, 3]] *= image_height

                formatted_targets.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                })

                gt_labels_np = gt_labels.cpu().numpy()

                # Track for F1/confusion matrix
                all_pred_labels.extend(pred_labels)
                all_true_labels.extend(gt_labels_np)
                all_confidence.extend(pred_confs)

                # Per-class metrics (simple TP/FP/FN counting)
                for label in gt_labels_np:
                    if label in pred_labels:
                        class_metrics[int(label)]["tp"] += 1
                    else:
                        class_metrics[int(label)]["fn"] += 1

                for pred_label in pred_labels:
                    if pred_label not in gt_labels_np:
                        class_metrics[int(pred_label)]["fp"] += 1

            # Update mAP metric
            metric.update(formatted_preds, formatted_targets)

    # Compute mAP
    print("\nComputing mAP...", flush=True)
    res = metric.compute()
    mAP50 = res["map_50"].item()
    mAP50_95 = res["map"].item()

    # Compute F1 and confusion matrix
    print("Computing F1 and confusion matrix...", flush=True)

    if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
        # Pad shorter array
        max_len = max(len(all_true_labels), len(all_pred_labels))
        true_labels = all_true_labels + [0] * (max_len - len(all_true_labels))
        pred_labels = all_pred_labels + [0] * (max_len - len(all_pred_labels))

        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)

        cm = confusion_matrix(true_labels, pred_labels)
        class_report = classification_report(true_labels, pred_labels, zero_division=0)
    else:
        f1_macro = 0.0
        f1_weighted = 0.0
        f1_micro = 0.0
        cm = None
        class_report = "No predictions made"

    # Compute per-class metrics
    per_class_metrics = {}
    for class_id, metrics_dict in class_metrics.items():
        tp = metrics_dict["tp"]
        fp = metrics_dict["fp"]
        fn = metrics_dict["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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
        "class_report": class_report,
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


def main(model_path, test_dir, batch_size, output_dir):

    conf_threshold = 0.25
    iou_threshold = 0.45

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    from ultralytics import YOLO
    model = get_model()
    model.load_state_dict(torch.load("model/best_mAP50.pt", map_location=device))
    model.to(device)


    # Load test dataset
    print(f"Loading test data from {test_dir}")
    # Assuming you have WeaponsDataset class - adjust import as needed
    try:
        from train import WeaponsDataset, collate_fn
    except ImportError:
        print("ERROR: Could not import WeaponsDataset from train.py")
        print("Make sure train.py is in the same directory or PYTHONPATH")
        return

    test_dataset = WeaponsDataset(test_dir, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Run testing
    metrics = test_model(
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

    # Save results to JSON
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
    # with tarfile.open(f'model/model.tar.gz', 'r:*') as tar:
    #     tar.extractall(path='model/')
    # print("extracted data")
    os.makedirs("output", exist_ok=True)
    main('model/model.tar.gz', 'data/test', 16, "output")
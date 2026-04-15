from glob import glob

import cv2
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils.nms import non_max_suppression
import os
import json
import tarfile
import numpy as np
import random
from train import get_model


def compute_iou(box1, box2):
    """Compute IoU between two boxes (xyxy format)."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def test_model(model, test_loader, device, conf_threshold=0.25, iou_threshold=0.45):
    """
    Test YOLO model on test set and compute metrics with proper IoU-based matching.
    """
    model.eval()

    # Initialize metrics
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.reset()

    all_pred_labels = []
    all_true_labels = []

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
                    pred_boxes_xyxy = []
                    pred_labels = []
                else:
                    boxes = pred[:, :4]
                    scores = pred[:, 4]
                    labels = pred[:, 5].long()

                    formatted_preds.append({
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    })
                    pred_boxes_xyxy = boxes.cpu().numpy().tolist()
                    pred_labels = labels.cpu().numpy().tolist()

                # Format ground truth
                mask = batch_data["batch_idx"] == i
                gt_boxes = batch_data["bboxes"][mask].to(device)
                gt_labels = batch_data["cls"][mask].long().view(-1).to(device)

                if len(gt_boxes) > 0:
                    x, y, w_box, h_box = gt_boxes.unbind(-1)

                    gt_boxes_xyxy = torch.stack([
                        x - w_box / 2,
                        y - h_box / 2,
                        x + w_box / 2,
                        y + h_box / 2
                    ], dim=-1)

                    # Scale to pixels
                    gt_boxes_xyxy[:, [0, 2]] *= image_width
                    gt_boxes_xyxy[:, [1, 3]] *= image_height
                else:
                    gt_boxes_xyxy = torch.zeros((0, 4), device=device)

                formatted_targets.append({
                    "boxes": gt_boxes_xyxy,
                    "labels": gt_labels,
                })

                gt_boxes_xyxy_np = gt_boxes_xyxy.cpu().numpy().tolist()
                gt_labels_np = gt_labels.cpu().numpy().tolist()

                # ✅ PROPER IoU-based matching for confusion matrix
                matched_gt = set()
                for pred_box, pred_label in zip(pred_boxes_xyxy, pred_labels):
                    best_iou = 0
                    best_gt_idx = -1
                    best_gt_label = -1

                    # Find best matching GT box
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_xyxy_np, gt_labels_np)):
                        if gt_idx in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                            best_gt_label = gt_label

                    # If IoU > 0.5, count as detection (match)
                    if best_iou > 0.5:
                        matched_gt.add(best_gt_idx)
                        pred_label_int = int(pred_label)
                        gt_label_int = int(best_gt_label)

                        # Track for confusion matrix
                        all_pred_labels.append(pred_label_int)
                        all_true_labels.append(gt_label_int)

                        # Count TP/FP by class
                        if pred_label_int == gt_label_int:
                            class_metrics[gt_label_int]["tp"] += 1
                        else:
                            # Misclassified
                            class_metrics[gt_label_int]["fn"] += 1
                            class_metrics[pred_label_int]["fp"] += 1
                    else:
                        # False positive (no match)
                        class_metrics[int(pred_label)]["fp"] += 1

                # Count unmatched GTs as FN
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes_xyxy_np, gt_labels_np)):
                    if gt_idx not in matched_gt:
                        class_metrics[int(gt_label)]["fn"] += 1

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
        f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        f1_micro = f1_score(all_true_labels, all_pred_labels, average='micro', zero_division=0)

        cm = confusion_matrix(all_true_labels, all_pred_labels)
        class_report = classification_report(all_true_labels, all_pred_labels, zero_division=0)
    else:
        f1_macro = 0.0
        f1_weighted = 0.0
        f1_micro = 0.0
        cm = None
        class_report = "No predictions made"

    # Compute per-class metrics
    per_class_metrics = {}
    for class_id in range(3):  # 3 weapon classes
        metrics_dict = class_metrics.get(class_id, {"tp": 0, "fp": 0, "fn": 0})
        tp = metrics_dict["tp"]
        fp = metrics_dict["fp"]
        fn = metrics_dict["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[class_id] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return {
        "mAP50": float(mAP50),
        "mAP50_95": float(mAP50_95),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_micro": float(f1_micro),
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
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model loaded successfully")

    # Load test dataset
    print(f"Loading test data from {test_dir}")
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
        num_workers=0,  # Windows compatibility
        collate_fn=collate_fn,
        pin_memory=False if device.type == 'cpu' else True,
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

    class_names = ["Handgun", "Rifle", "Knife"]
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
        "per_class_metrics": metrics['per_class_metrics']
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print("=" * 60)

def visualize_detections_from_path(model, data_path, device, num_images=10,
                                   conf_threshold=0.25, iou_threshold=0.45):
    """
    Randomly sample images from YOLO-format dataset folder and DISPLAY predictions.

    Assumes:
    data_path/
        images/
        labels/
    """

    model.eval()
    class_names = ["Handgun", "Rifle", "Knife"]

    image_paths = glob(os.path.join(data_path, "images", "*"))
    image_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    def load_labels(label_path):
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            return boxes, labels

        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.strip().split())
                labels.append(int(cls))
                boxes.append([x, y, w, h])

        return boxes, labels

    with torch.no_grad():
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            label_path = os.path.join(
                data_path, "labels",
                os.path.splitext(img_name)[0] + ".txt"
            )

            # --- Load image ---
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Prepare model input
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

            # --- Load GT ---
            gt_boxes, gt_labels = load_labels(label_path)

            # --- Inference ---
            preds = model(img_tensor)
            preds = non_max_suppression(preds, conf_threshold, iou_threshold)[0]

            # --- Draw GT (GREEN) ---
            for (x, y, bw, bh), label in zip(gt_boxes, gt_labels):
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img,
                            f"GT: {class_names[label]}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

            # --- Draw predictions (RED) ---
            if preds is not None and len(preds) > 0:
                for pred in preds:
                    x1, y1, x2, y2, conf, cls = pred
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cls = int(cls)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img,
                                f"{class_names[cls]} {conf:.2f}",
                                (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)

            # --- Display ---
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(img_name)
            plt.axis("off")
            plt.show()

if __name__ == '__main__':
    # with tarfile.open(f'../model/model.tar.gz', 'r:*') as tar:
    #     tar.extractall(path='../model')
    # print("extracted data")
    # os.makedirs("output", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main('model/best_mAP50.pt', 'data/test', 16, "output")
    model = get_model()
    model.load_state_dict(torch.load('../model/best_mAP50.pt', map_location=device))
    visualize_detections_from_path(model, '../data/test', device, 10,
                                   conf_threshold=0.20, iou_threshold=0.3)
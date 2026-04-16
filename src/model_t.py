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

# Class index 3 is the synthetic "background / no-detection" label used when a
# predicted box has no matching GT box, or a GT box has no matching prediction.
BACKGROUND_CLASS = 3
CLASS_NAMES = ["Handgun", "Rifle", "Knife", "Background"]
NUM_WEAPON_CLASSES = 3  # real detector classes (0-2)


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

    Background handling
    -------------------
    - Unmatched prediction  → pred label  = BACKGROUND_CLASS, true label = BACKGROUND_CLASS
      (the model fired but hit nothing; counted as FP for the predicted weapon class
       AND logged in the confusion matrix as pred=Background / true=Background)
    - Unmatched GT box      → true label  = gt_label,         pred label = BACKGROUND_CLASS
      (the model missed a real object; counted as FN for that weapon class AND logged
       in the confusion matrix as pred=Background / true=<weapon>)
    """
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox")
    metric.reset()

    all_pred_labels = []
    all_true_labels = []

    # Per-class TP / FP / FN for weapon classes only (background FPs/FNs are
    # implicitly captured by the weapon-class FP/FN increments below).
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        print("Running inference on test set...", flush=True)
        for batch_idx, batch in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(test_loader)}", flush=True)

            images = batch["img"].to(device)
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

            preds = model(images)
            nmp_preds = non_max_suppression(preds, conf_threshold, iou_threshold)

            formatted_preds = []
            formatted_targets = []

            image_height, image_width = images.shape[2:]

            for i in range(len(images)):
                pred = nmp_preds[i]

                # ── Predictions ──────────────────────────────────────────────
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
                    formatted_preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                    pred_boxes_xyxy = boxes.cpu().numpy().tolist()
                    pred_labels = labels.cpu().numpy().tolist()

                # ── Ground truth ─────────────────────────────────────────────
                mask = batch_data["batch_idx"] == i
                gt_boxes = batch_data["bboxes"][mask].to(device)
                gt_labels = batch_data["cls"][mask].long().view(-1).to(device)

                if len(gt_boxes) > 0:
                    x, y, w_box, h_box = gt_boxes.unbind(-1)
                    gt_boxes_xyxy = torch.stack([
                        x - w_box / 2, y - h_box / 2,
                        x + w_box / 2, y + h_box / 2,
                    ], dim=-1)
                    gt_boxes_xyxy[:, [0, 2]] *= image_width
                    gt_boxes_xyxy[:, [1, 3]] *= image_height
                else:
                    gt_boxes_xyxy = torch.zeros((0, 4), device=device)

                formatted_targets.append({"boxes": gt_boxes_xyxy, "labels": gt_labels})

                gt_boxes_xyxy_np = gt_boxes_xyxy.cpu().numpy().tolist()
                gt_labels_np = gt_labels.cpu().numpy().tolist()

                # ── IoU-based matching ───────────────────────────────────────
                matched_gt = set()

                for pred_box, pred_label in zip(pred_boxes_xyxy, pred_labels):
                    best_iou = 0
                    best_gt_idx = -1
                    best_gt_label = -1

                    for gt_idx, (gt_box, gt_label) in enumerate(
                            zip(gt_boxes_xyxy_np, gt_labels_np)):
                        if gt_idx in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                            best_gt_label = gt_label

                    pred_label_int = int(pred_label)

                    if best_iou > 0.5:
                        # ── Matched prediction ────────────────────────────────
                        matched_gt.add(best_gt_idx)
                        gt_label_int = int(best_gt_label)

                        all_pred_labels.append(pred_label_int)
                        all_true_labels.append(gt_label_int)

                        if pred_label_int == gt_label_int:
                            class_metrics[gt_label_int]["tp"] += 1
                        else:
                            # Correct localisation, wrong class
                            class_metrics[gt_label_int]["fn"] += 1
                            class_metrics[pred_label_int]["fp"] += 1
                    else:
                        # ── Unmatched prediction (false positive) ──────────────
                        # The model fired a box that overlaps nothing real.
                        # Since there are no background images in the dataset,
                        # there is no valid GT label to pair this with, so we do
                        # NOT add it to the confusion matrix labels (that would
                        # create a spurious "true=Background" row).
                        # We only count it as FP for the predicted weapon class.
                        class_metrics[pred_label_int]["fp"] += 1

                # ── Unmatched GTs → Background ───────────────────────────────
                # The model missed these real objects.
                # In the confusion matrix: true=<weapon>, pred=Background.
                for gt_idx, (gt_box, gt_label) in enumerate(
                        zip(gt_boxes_xyxy_np, gt_labels_np)):
                    if gt_idx not in matched_gt:
                        gt_label_int = int(gt_label)
                        all_pred_labels.append(BACKGROUND_CLASS)
                        all_true_labels.append(gt_label_int)
                        class_metrics[gt_label_int]["fn"] += 1

            metric.update(formatted_preds, formatted_targets)

    # ── mAP ─────────────────────────────────────────────────────────────────
    print("\nComputing mAP...", flush=True)
    res = metric.compute()
    mAP50 = res["map_50"].item()
    mAP50_95 = res["map"].item()

    # ── F1 / confusion matrix (4-class, including Background) ────────────────
    print("Computing F1 and confusion matrix...", flush=True)
    if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
        # True labels are always weapon classes [0,1,2] — no background images exist.
        # Predicted labels can be [0,1,2,3] where 3=Background means a missed detection.
        # F1 is computed only over the weapon classes (rows that actually have GT data).
        weapon_labels = list(range(NUM_WEAPON_CLASSES))  # [0, 1, 2]
        all_labels = list(range(NUM_WEAPON_CLASSES + 1))  # [0, 1, 2, 3]

        f1_macro = f1_score(all_true_labels, all_pred_labels,
                            labels=weapon_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(all_true_labels, all_pred_labels,
                               labels=weapon_labels, average='weighted', zero_division=0)
        f1_micro = f1_score(all_true_labels, all_pred_labels,
                            labels=weapon_labels, average='micro', zero_division=0)

        # Confusion matrix rows = true weapon classes [0,1,2],
        # cols = predicted [0,1,2,3] (Background col = missed detections).
        # Drop the Background row (index 3) — it has no real GT samples.
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=all_labels)
        cm = cm[:NUM_WEAPON_CLASSES, :]  # keep rows 0-2 only

        class_report = classification_report(all_true_labels, all_pred_labels,
                                             labels=weapon_labels,
                                             target_names=CLASS_NAMES[:NUM_WEAPON_CLASSES],
                                             zero_division=0)
    else:
        f1_macro = f1_weighted = f1_micro = 0.0
        cm = None
        class_report = "No predictions made"

    # ── Per-class metrics (weapon classes only) ───────────────────────────────
    per_class_metrics = {}
    for class_id in range(NUM_WEAPON_CLASSES):
        m = class_metrics.get(class_id, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class_metrics[class_id] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    # ── Overall model precision & recall (micro across weapon classes) ────────
    # We sum TP/FP/FN over the three weapon classes only (background is not a
    # "real" detector class, so including it would distort the numbers).
    total_tp = sum(class_metrics[c]["tp"] for c in range(NUM_WEAPON_CLASSES))
    total_fp = sum(class_metrics[c]["fp"] for c in range(NUM_WEAPON_CLASSES))
    total_fn = sum(class_metrics[c]["fn"] for c in range(NUM_WEAPON_CLASSES))

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall /
                  (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0.0)

    return {
        "mAP50": float(mAP50),
        "mAP50_95": float(mAP50_95),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_micro": float(f1_micro),
        "confusion_matrix": cm,
        "class_report": class_report,
        "per_class_metrics": per_class_metrics,
        # Overall aggregated metrics
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "overall_tp": int(total_tp),
        "overall_fp": int(total_fp),
        "overall_fn": int(total_fn),
    }


def plot_confusion_matrix(cm, row_names, col_names, output_path="confusion_matrix.png"):
    """
    Plot and save confusion matrix.

    cm        : shape (num_true_classes, num_pred_classes)
    row_names : true-label names   — length must match cm.shape[0]
    col_names : predicted-label names — length must match cm.shape[1]
    """
    if cm is None:
        print("No confusion matrix to plot")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=col_names, yticklabels=row_names)
    plt.title("Confusion Matrix\n"
              "(rows = true weapon classes, Background col = missed detections)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def main(model_path, test_dir, batch_size, output_dir):
    conf_threshold = 0.25
    iou_threshold = 0.45

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}")
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model loaded successfully")

    print(f"Loading test data from {test_dir}")
    try:
        from train import WeaponsDataset, collate_fn
    except ImportError:
        print("ERROR: Could not import WeaponsDataset from train.py")
        return

    test_dataset = WeaponsDataset(test_dir, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False if device.type == 'cpu' else True,
    )
    print(f"Test dataset size: {len(test_dataset)}")

    metrics = test_model(model, test_loader, device,
                         conf_threshold=conf_threshold,
                         iou_threshold=iou_threshold)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"mAP50:      {metrics['mAP50']:.4f}")
    print(f"mAP50-95:   {metrics['mAP50_95']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (micro):    {metrics['f1_micro']:.4f}")

    print("\n" + "-" * 60)
    print("OVERALL MODEL PRECISION & RECALL  (micro, weapon classes only)")
    print("-" * 60)
    print(f"  Precision : {metrics['overall_precision']:.4f}")
    print(f"  Recall    : {metrics['overall_recall']:.4f}")
    print(f"  F1        : {metrics['overall_f1']:.4f}")
    print(f"  TP={metrics['overall_tp']}  FP={metrics['overall_fp']}  FN={metrics['overall_fn']}")

    print("\n" + "-" * 60)
    print("PER-CLASS METRICS  (weapon classes)")
    print("-" * 60)
    for class_id, cm_dict in metrics['per_class_metrics'].items():
        name = CLASS_NAMES[class_id]
        print(f"\n{name}:")
        print(f"  Precision : {cm_dict['precision']:.4f}")
        print(f"  Recall    : {cm_dict['recall']:.4f}")
        print(f"  F1        : {cm_dict['f1']:.4f}")
        print(f"  TP={cm_dict['tp']}  FP={cm_dict['fp']}  FN={cm_dict['fn']}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT  (4-class, Background included)")
    print("-" * 60)
    print(metrics['class_report'])

    # Plot 4×4 confusion matrix
    if metrics['confusion_matrix'] is not None:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            row_names=CLASS_NAMES[:NUM_WEAPON_CLASSES],  # true rows: Handgun/Rifle/Knife
            col_names=CLASS_NAMES,  # pred cols: + Background
            output_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

    # Save JSON
    results_file = os.path.join(output_dir, "test_results.json")
    results_dict = {
        "mAP50": metrics['mAP50'],
        "mAP50_95": metrics['mAP50_95'],
        "f1_macro": metrics['f1_macro'],
        "f1_weighted": metrics['f1_weighted'],
        "f1_micro": metrics['f1_micro'],
        "overall_precision": metrics['overall_precision'],
        "overall_recall": metrics['overall_recall'],
        "overall_f1": metrics['overall_f1'],
        "overall_tp": metrics['overall_tp'],
        "overall_fp": metrics['overall_fp'],
        "overall_fn": metrics['overall_fn'],
        "per_class_metrics": metrics['per_class_metrics'],
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

    image_paths = glob(os.path.join(data_path, "images", "*"))
    image_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    def load_labels(label_path):
        boxes, labels = [], []
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

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = (torch.from_numpy(img_rgb).float() / 255.0
                          ).permute(2, 0, 1).unsqueeze(0).to(device)

            gt_boxes, gt_labels = load_labels(label_path)

            preds = model(img_tensor)
            preds = non_max_suppression(preds, conf_threshold, iou_threshold)[0]

            # GT in green
            for (x, y, bw, bh), label in zip(gt_boxes, gt_labels):
                x1 = int((x - bw / 2) * w);
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w);
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"GT: {CLASS_NAMES[label]}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Predictions in red
            if preds is not None and len(preds) > 0:
                for pred in preds:
                    x1, y1, x2, y2, conf, cls = pred
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cls = int(cls)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{CLASS_NAMES[cls]} {conf:.2f}",
                                (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(img_name)
            plt.axis("off")
            plt.show()

if __name__ == '__main__':
    # with tarfile.open(f'../model/model.tar.gz', 'r:*') as tar:
    #     tar.extractall(path='../model')
    # print("extracted data")
    # TODO: Add handle no detections to metrics
    os.makedirs("../output", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main('../model/best_mAP50.pt', '../data/test', 16, "../output")
    model = get_model()
    model.load_state_dict(torch.load('../model/best_mAP50.pt', map_location=device))
    visualize_detections_from_path(model, '../data/test', device, 10,
                                   conf_threshold=0.20, iou_threshold=0.3)
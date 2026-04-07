import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from ultralytics.utils.instance import Instances
import random
import os



class WeaponsDataset(Dataset):
    def __init__(self, data_dir, img_size=768):
        self.img_dir = Path(f"{data_dir}/images")
        self.label_dir = Path(f"{data_dir}/labels")
        self.images = [str(os.path.join(self.img_dir, f)) for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        self.img_size = img_size
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.Mosaic(p=0.5, target_size=(img_size, img_size), fit_mode='contain', metadata_key="mosaic_metadata"),
        ],
            bbox_params=A.BboxParams(
                format="yolo",  # xywh norm
                label_fields=["class_labels"],
                clip=True,
                min_visibility=0.1
            )
        )


    def __len__(self):
        return len(self.images)

    def load_sample(self, idx):
        img_path = self.images[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

        bboxes = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    c, x, y, w, h = map(float, line.strip().split())
                    class_labels.append(int(c))
                    bboxes.append([x, y, w, h])

        return {
            "image": img,
            "bboxes": np.array(bboxes),
            "class_labels": class_labels,
        }

    def get_mosaic_metadata(self, idx):
        return [
            self.load_sample(i)
            for i in random.sample(range(len(self)), 5)
        ]

    def __getitem__(self, idx):

        sample = self.load_sample(idx)


        mosaic_metadata = self.get_mosaic_metadata(idx)

        out = self.transform(
            image=sample["image"],
            bboxes=sample["bboxes"],
            class_labels=sample["class_labels"],
            mosaic_metadata=mosaic_metadata,
        )

        img = out["image"]
        bboxes = torch.tensor(out["bboxes"], dtype=torch.float32)
        labels = torch.tensor(out["class_labels"], dtype=torch.int64)

        return {
            "image": img,
            "class_labels": labels,
            "bboxes": bboxes,
        }

def collate_fn(batch):
    images = []
    cls_list = []
    box_list = []
    batch_idx = []

    for i, item in enumerate(batch):
        img = item["image"]
        bboxes = item["bboxes"]
        cls = item["class_labels"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        images.append(img)

        if len(bboxes) > 0:
            for b, c in zip(bboxes, cls):
                x, y, w, h = b
                cls_list.append(c)
                box_list.append([x, y, w, h])
                batch_idx.append(i)

    return {
        "img": torch.stack(images),
        "cls": torch.tensor(cls_list, dtype=torch.float32),
        "bboxes": torch.tensor(box_list, dtype=torch.float32),
        "batch_idx": torch.tensor(batch_idx, dtype=torch.int64),
    }


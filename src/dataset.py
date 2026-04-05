import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from ultralytics.data.augment import (
    Compose, Mosaic,
    MixUp, RandomPerspective, RandomHSV,
    RandomFlip, LetterBox, Albumentations
)


class WeaponsDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir = Path(f"{data_dir}/images")
        self.label_dir = Path(f"{data_dir}/labels")
        self.images = [f for f in self.img_dir.iterdir() if f.is_file()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        img = Image.open(img_path).convert("RGB")

        labels = []
        content = label_path.read_text().strip()

        if content:
            for label in content.splitlines():
                parts = list(map(float, label.split()))
                labels.append(parts)
        if labels:
            w_img, h_img = img.size
            labels = torch.tensor(labels, dtype=torch.float32)

            classes = labels[:, 0]
            boxes = labels[:, 1:]

            # Convert bbox coords xywh → xyxy (normalized)
            x, y, w, h = boxes.unbind(-1)
            boxes = torch.stack([
                (x - w / 2) * w_img,
                (y - h / 2) * h_img,
                (x + w / 2) * w_img,
                (y + h / 2) * h_img,
            ], dim=-1)
        else:
            classes = torch.zeros((0,), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "img": img,
            "cls": classes,
            "bboxes": boxes,
            "im_file": str(img_path),
        }


class YoloCollate:
    def __init__(self, dataset, imgsz=768):
        self.mosaic = Mosaic(dataset=dataset, imgsz=imgsz, p=1.0)
        self.mixup = MixUp(dataset=dataset, p=0.2)
        self.perspective = RandomPerspective(
            degrees=15,
            translate=0.1,
            scale=0.5,
            shear=2,
            perspective=0.0,
        )
        self.hsv = RandomHSV(
            hgain=0.05,
            sgain=0.05,
            vgain=0.05,
        )

    def __call__(self, batch):
        samples = list(batch)
        new_samples = []

        for sample in samples:
            sample = self.mosaic(sample)
            sample = self.mixup(sample)
            sample = self.perspective(sample)
            sample = self.hsv(sample)

            img = sample["img"]

            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(
                    (np.array(img)).transpose(2, 0, 1)
                ).float() / 255.0

            sample["img"] = img.contiguous()

            new_samples.append(sample)

        imgs = torch.stack([s["img"] for s in new_samples])
        cls_list, box_list, batch_idx = [], [], []

        for i, s in enumerate(new_samples):
            n = len(s['cls'])
            if n == 0:
                continue
            cls_list.append(s['cls'])
            box_list.append(s['bboxes'])
            batch_idx.append(torch.full((n,), i, dtype=torch.long))

        if len(cls_list):
            cls = torch.cat(cls_list).unsqueeze(1)
            bboxes = torch.cat(box_list)
            batch_idx = torch.cat(batch_idx)
        else:
            cls = torch.zeros((0,1))
            bboxes = torch.zeros((0,4))
            batch_idx = torch.zeros((0,), dtype=torch.long)

        return {
            "img": imgs,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": batch_idx,
        }

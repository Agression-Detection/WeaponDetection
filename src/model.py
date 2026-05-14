from ultralytics import YOLO
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def get_model(device, is_dist:bool, local_rank):
    model = YOLO('yolo11s.pt').model
    print("YOLOv11 Loaded Successfully!")
    num_classes = 3
    detect_layer = model.model[23]
    model.model[-1].nc = 3
    model.nc = 3

    for i in range(3):
        old_conv = detect_layer.cv3[i][2]
        in_ch = old_conv.in_channels

        detect_layer.cv3[i][2] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        nn.init.normal_(detect_layer.cv3[i][2].weight, std=0.01)
        nn.init.constant_(detect_layer.cv3[i][2].bias, 0)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    for p in model.model[23].parameters():
        p.requires_grad = True

    model.to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return model
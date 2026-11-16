#!/usr/bin/env python
"""Для опыта с обучаемым энкодером на CUB: выходы замороженной части ResNet-18 (ImageNet) до layer3 включительно,
224x224 (Resize 256 + CenterCrop 224), и объединённые по пространству выходы исходного (предобученного) layer4 --
ориентир «вход» и цель для восстановления. Запускать окружением с torchvision
(the local Python environment).
Результат: data/cub/resnet18_layer3.npy (float16, [N, 256, 14, 14], memmap), data/cub/emb_resnet18.npy
(float16, [N, 512], единичная длина).
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "data" / "cub" / "CUB_200_2011" / "images"
OUT = HERE / "data" / "cub"


def main():
    paths = json.loads((OUT / "texts.json").read_text())
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
    stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    maps = np.lib.format.open_memmap(OUT / "resnet18_layer3.npy", mode="w+", dtype=np.float16,
                                     shape=(len(paths), 256, 14, 14))
    pooled = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = torch.stack([tf(Image.open(SRC / p).convert("RGB")) for p in paths[i:i + 64]]).cuda()
            h = stem(batch)
            maps[i:i + len(batch)] = h.half().cpu().numpy()
            pooled.append(net.layer4(h).mean((2, 3)).float().cpu().numpy())
            if i % 1280 == 0:
                print(i, flush=True)
    maps.flush()
    f = np.concatenate(pooled)
    np.save(OUT / "emb_resnet18.npy", (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float16))
    print("saved")


if __name__ == "__main__":
    main()

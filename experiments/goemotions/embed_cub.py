#!/usr/bin/env python
"""Признаки изображений CUB замороженной Inception-v3 (ImageNet, как в CBM, Koh et al., 2020): 2048-мерный выход
после глобального пулинга, 299x299, нормировка ImageNet; признаки нормируются на единичную длину, как у текстов.
Запускать окружением с torchvision
(the local Python environment). Результат: data/cub/emb_inception.npy (float16).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "data" / "cub" / "CUB_200_2011" / "images"
OUT = HERE / "data" / "cub"


def main():
    import json
    paths = json.loads((OUT / "texts.json").read_text())
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.eval().cuda()
    tf = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = torch.stack([tf(Image.open(SRC / p).convert("RGB")) for p in paths[i:i + 64]]).cuda()
            feats.append(net(batch).float().cpu().numpy())
            if i % 1280 == 0:
                print(i, flush=True)
    f = np.concatenate(feats)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)  # как у текстов: признаки единичной длины
    np.save(OUT / "emb_inception.npy", f.astype(np.float16))
    print("saved", OUT / "emb_inception.npy")


if __name__ == "__main__":
    main()

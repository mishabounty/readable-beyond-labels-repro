#!/usr/bin/env python
"""SUN Attribute Database (Patterson & Hays) -> data/sun, data/sun4, data/sun4sem -- второй набор изображений.

14 340 изображений сцен, 717 категорий, 102 атрибута с тремя голосами людей на изображение. Атрибут считается
присутствующим при >= 2 голосах из 3; берутся атрибуты с долей >= 1%, они случайно (seed 0) делятся пополам на
A (sunA, обучение) и B (sunB, отложенные). Задачи, как на CUB: sun -- 717 категорий; sun4 -- 4 случайных
суперкласса категорий (seed 0); sun4sem -- 4 суперкласса k-means по среднему профилю атрибутов A категории (fit).
Разбиение стратифицировано по категории: 60 / 20 / 20 (seed 0).
Признаки -- как embed_cub_resnet.py: выходы ResNet-18 (ImageNet) до layer3 включительно (memmap) и объединённые
выходы предобученного layer4 (вход-ориентир). Запуск окружением с torchvision:
${PYTHON:-python} sun_prepare.py <каталог с SUNAttributeDB и images>.
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent


def main():
    root = Path(sys.argv[1])
    paths = [str(p[0]) for p in sio.loadmat(root / "SUNAttributeDB" / "images.mat")["images"].ravel()]
    names = [str(a[0]) for a in sio.loadmat(root / "SUNAttributeDB" / "attributes.mat")["attributes"].ravel()]
    votes = sio.loadmat(root / "SUNAttributeDB" / "attributeLabels_continuous.mat")["labels_cv"]
    cats = sorted({p.rsplit("/", 1)[0] for p in paths})
    y = np.array([cats.index(p.rsplit("/", 1)[0]) for p in paths])
    rng = np.random.default_rng(0)
    fit, val, test = [], [], []
    for c in range(len(cats)):
        idx = rng.permutation(np.where(y == c)[0])
        k1, k2 = int(round(0.6 * len(idx))), int(round(0.8 * len(idx)))
        fit += idx[:k1].tolist(); val += idx[k1:k2].tolist(); test += idx[k2:].tolist()
    fit, val, test = np.sort(fit), np.sort(val), np.sort(test)
    Y = (votes >= 0.5).astype(np.int8)
    keep = np.where(Y.mean(0) >= 0.01)[0]
    perm = rng.permutation(keep)
    A, B = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    sup_rand = rng.permutation(np.arange(len(cats)) % 4)
    prof = np.stack([Y[fit][y[fit] == c][:, A].mean(0) for c in range(len(cats))])
    sup_sem = KMeans(4, n_init=10, random_state=0).fit_predict(prof)
    base = HERE / "data" / "sun"
    base.mkdir(parents=True, exist_ok=True)
    (base / "texts.json").write_text(json.dumps(paths))
    np.savez(base / "split.npz", fit=fit, val=val, test=test)
    for name, idx in (("sunA", A), ("sunB", B)):
        sp.save_npz(base / f"concepts_{name}.npz", sp.csr_matrix(Y[:, idx]))
        (base / f"concepts_{name}.json").write_text(json.dumps([names[i] for i in idx]))
    for d, labels in (("sun", y), ("sun4", sup_rand[y]), ("sun4sem", sup_sem[y])):
        dd = HERE / "data" / d
        dd.mkdir(exist_ok=True)
        np.save(dd / "labels.npy", labels.astype(np.int64))
        meta = {"images": len(paths), "categories": len(cats), "classes": int(labels.max()) + 1,
                "attributes_A": len(A), "attributes_B": len(B), "n_fit": len(fit), "n_val": len(val),
                "n_test": len(test), "class_sizes": np.bincount(labels).tolist()[:10]}
        (dd / "meta.json").write_text(json.dumps(meta))
        if d != "sun":
            for f in ("texts.json", "split.npz", "concepts_sunA.npz", "concepts_sunA.json", "concepts_sunB.npz",
                      "concepts_sunB.json", "resnet18_layer3.npy", "emb_resnet18.npy"):
                if not (dd / f).exists():
                    (dd / f).symlink_to(Path("..") / "sun" / f)
        print(d, json.dumps(meta), flush=True)
    # признаки ResNet-18
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
    stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    maps = np.lib.format.open_memmap(base / "resnet18_layer3.npy", mode="w+", dtype=np.float16,
                                     shape=(len(paths), 256, 14, 14))
    pooled = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = torch.stack([tf(Image.open(root / "images" / p).convert("RGB")) for p in paths[i:i + 64]]).cuda()
            h = stem(batch)
            maps[i:i + len(batch)] = h.half().cpu().numpy()
            pooled.append(net.layer4(h).mean((2, 3)).float().cpu().numpy())
    maps.flush()
    f = np.concatenate(pooled)
    np.save(base / "emb_resnet18.npy", (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float16))
    print("features saved", flush=True)


if __name__ == "__main__":
    main()

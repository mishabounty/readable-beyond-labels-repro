#!/usr/bin/env python
"""Опубликованные концептные модели как этап 1 для collapse_check.py: тот же энкодер z = normalize(MLP(x)).

  cbm -- joint Concept Bottleneck Model (Koh et al., 2020): z -> логиты концептов -> sigmoid -> классификатор;
  cem -- Concept Embedding Model (Espinosa Zarlenga et al., 2022): для концепта j два вложения c+_j, c-_j из z,
         p_j = sigmoid(s([c+_j, c-_j])), вложение концепта p_j c+_j + (1 - p_j) c-_j, классификатор по их конкатенации.
Потеря: CE + lambda * BCE концептов (та же нормированная BCE, что в train.py); ранняя остановка по val macro-F1.
Возвращается энкодер (дальше z читается так же, как у остальных методов).
"""

import torch
import torch.nn.functional as F
from torch import nn

import train as T


class CBMHead(nn.Module):
    def __init__(self, m, n_cls):
        super().__init__()
        self.concepts = T.concept_head(768, m)
        self.task = nn.Linear(m, n_cls)

    def forward(self, z):
        logits = self.concepts(z)
        return logits, self.task(torch.sigmoid(logits))


class CEMHead(nn.Module):
    def __init__(self, m, n_cls, emb=16):
        super().__init__()
        self.m, self.emb = m, emb
        self.embed = nn.Linear(768, 2 * m * emb)
        self.score = nn.Linear(2 * emb, 1)
        self.task = nn.Linear(m * emb, n_cls)

    def forward(self, z):
        c = F.leaky_relu(self.embed(z)).view(len(z), self.m, 2, self.emb)
        logits = self.score(c.flatten(2)).squeeze(-1)
        p = torch.sigmoid(logits)[..., None]
        mixed = p * c[:, :, 0] + (1 - p) * c[:, :, 1]
        return logits, self.task(mixed.flatten(1))


def stage1(x, y, split, M, kind, lam, args, gen):
    n_cls = int(y.max()) + 1
    enc = T.Encoder(x.shape[1]).to(T.DEV)
    head = (CBMHead if kind == "cbm" else CEMHead)(M.shape[1], n_cls).to(T.DEV)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": args.lr},
                             {"params": head.parameters(), "lr": 1e-3 if kind == "cem" else 1e-2}], weight_decay=0.01)
    best, stale = None, 0
    for ep in range(1, args.epochs + 1):
        enc.train()
        for b in T.batches(split["fit"], args.batch, True, gen):
            logits, out = head(enc(x[b].to(T.DEV)))
            loss = F.cross_entropy(out, y[b].to(T.DEV)) + lam * T.bce_concepts(logits, T.dense(M, b).to(T.DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
        enc.eval()
        with torch.no_grad():
            pv = torch.cat([head(enc(x[i].to(T.DEV)))[1].argmax(1).cpu() for i in torch.split(split["val"], 2048)])
        f1 = T.macro_f1(y[split["val"]], pv, n_cls)
        if best is None or f1 > best[0] + 1e-4:
            best, stale = (f1, ep, {k: v.detach().clone() for k, v in enc.state_dict().items()},
                           {k: v.detach().clone() for k, v in head.state_dict().items()}), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    enc.load_state_dict(best[2])
    head.load_state_dict(best[3])
    enc.eval()
    with torch.no_grad():
        outs = [head(enc(x[i].to(T.DEV))) for i in torch.split(split["test"], 2048)]
        pt = torch.cat([o[1].argmax(1).cpu() for o in outs])
        # качество выучивания концептов этапа 1 (AP логитов концептов на тесте)
        a_m = T.concept_metrics(torch.cat([o[0] for o in outs]), T.dense(M, split["test"]).to(T.DEV))
    del head, outs
    return enc, {"val_macro_f1": best[0], "best_epoch": best[1],
                 "test_macro_f1": T.macro_f1(y[split["test"]], pt, n_cls), "a_ap": a_m["ap"],
                 "a_macro_ap": a_m["macro_ap"]}

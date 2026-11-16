import yaml, glob, os
import numpy as np
K = {"hicem": "enhanced_clip_vitl14_hicem_discovered_concept_auc",
     "cemcs": "enhanced_clip_vitl14_cem_discovered_concept_auc",
     "unlab": "unlabelled_clip_vitl14_hicem_discovered_concept_auc",
     "coarse": "initial_clip_vitl14_cem_provided_concept_auc",
     "task": "initial_clip_vitl14_cem_task_accuracy"}
for t in ("R", "Rtrue", "C1", "P", "Rnull"):
    for rd in sorted(glob.glob(f"${REPRO_ROOT:-.}/results/{t}/*/")):
        d = {}
        for doc in yaml.safe_load_all(open(rd + "results.yaml")):
            if isinstance(doc, dict):
                d.update(doc)
        a = np.array(d.get("discovered_concept_roc_aucs", []), float)
        m = a > 0
        g = {k: d.get(v, float("nan")) for k, v in K.items()}
        ul = [k for k in d if k.startswith("unlabelled") and "auc" in k and not isinstance(d[k], list)]
        print(f"{t:6s} {os.path.basename(rd[:-1]):22s} cand {len(a):6d} matched {int(m.sum()):4d} "
              f"AUCtrain {a[m].mean() if m.any() else float('nan'):.4f} | HiCEM {g['hicem']:.4f} "
              f"CEM+CS {g['cemcs']:.4f} | coarse0 {g['coarse']:.4f} task0 {g['task']:.4f} | unlab keys {ul[:3]}")

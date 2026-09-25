#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 9: train.jsonl amb els clips sintetics i els reals, estadistiques i sessions llargues."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import numpy as np

import config as C
import corpus

DIR_DATASET = C.DIR_DATASET
MANIFEST = DIR_DATASET / "manifest.jsonl"
TRAIN = DIR_DATASET / "train.jsonl"
DIR_SESSIONS = DIR_DATASET / "sessions"
SESSIO_MAX_S = 28.0
SR = 16000


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", action="store_true")
    p.add_argument("--sense-cv-real", action="store_true",
                   help="no hi afegeix els clips reals de Common Voice (pas 6c)")
    args = p.parse_args()
    m = [x for x in corpus.llegeix_jsonl(MANIFEST) if x.get("status") == "ok"]
    if not args.sense_cv_real and C.MANIFEST_CV_REAL.exists():
        reals = [x for x in corpus.llegeix_jsonl(C.MANIFEST_CV_REAL) if x.get("status") == "ok"]
        print(f"{len(reals):,} clips reals de Common Voice (pas 6c) afegits al train")
        m += reals
    if not m:
        raise SystemExit("Cap audio ok al manifest (pas 8).")
    dev = [x for x in m if x.get("split") == "dev"]
    m = [x for x in m if x.get("split", "train") != "dev"]
    fila = lambda x: json.dumps({"audio": x["audio"], "text": x["text"], "duration": x["durada_s"],
                                 "speaker": x["veu_id"], "entorn": x["entorn_id"], "font": x["font"],
                                 "id": x["id"], "split": x.get("split", "train")}, ensure_ascii=False) + "\n"
    with TRAIN.open("w", encoding="utf-8") as fh:
        for x in m:
            fh.write(fila(x))
    with (DIR_DATASET / "dev.jsonl").open("w", encoding="utf-8") as fh:
        for x in dev:
            fh.write(fila(x))
    reals_dev = [x for x in dev if x["entorn_id"] == "real"]
    print(f"dev: {len(dev):,} clips · {sum(x['durada_s'] for x in dev)/3600:.2f} h ({len(reals_dev)} reals, "
          f"{len(dev) - len(reals_dev)} sintetics) -> dev.jsonl; fora del train")
    h = sum(x["durada_s"] for x in m) / 3600
    d = sorted(x["durada_s"] for x in m)
    print(f"{len(m):,} clips · {h:.2f} h · durada mediana {d[len(d) // 2]:.1f}s p95 {d[int(len(d) * .95)]:.1f}s max {d[-1]:.1f}s")
    for titol, clau in (("veu", "veu_id"), ("entorn", "entorn_id"), ("font", "font")):
        c = collections.Counter(x[clau] for x in m)
        hh = collections.defaultdict(float)
        for x in m: hh[x[clau]] += x["durada_s"]
        print(f"\n  per {titol}:")
        for k, n in c.most_common(): print(f"    {k:<28} {n:>7,}  {hh[k] / 3600:>6.2f} h")

    if args.sessions:
        import soundfile as sf
        DIR_SESSIONS.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        per_veu = collections.defaultdict(list)
        for x in m: per_veu[x["veu_id"]].append(x)
        n_s, h_s = 0, 0.0
        with (DIR_DATASET / "train_sessions.jsonl").open("w", encoding="utf-8") as fh:
            for veu, clips in per_veu.items():
                rng.shuffle(clips)
                grup, dur = [], 0.0
                for x in clips + [None]:
                    if x is None or (dur + x["durada_s"] + 0.5 * len(grup)) > SESSIO_MAX_S:
                        if len(grup) >= 2:
                            parts, textos = [], []
                            for g in grup:
                                y, sr = sf.read(ARREL / g["audio"], dtype="float32")
                                parts += [y, np.zeros(int(sr * rng.uniform(0.3, 0.8)), dtype="float32")]
                                textos.append(g["text"])
                            y = np.concatenate(parts[:-1])
                            ruta = DIR_SESSIONS / f"ses_{n_s:06d}.wav"
                            sf.write(ruta, y, SR, subtype="PCM_16")
                            fh.write(json.dumps({"audio": str(ruta.relative_to(ARREL)), "text": " ".join(textos),
                                                 "duration": round(len(y) / SR, 3), "speaker": veu,
                                                 "clips": [g["id"] for g in grup]}, ensure_ascii=False) + "\n")
                            n_s += 1; h_s += len(y) / SR / 3600
                        grup, dur = ([x], x["durada_s"]) if x else ([], 0.0)
                    else:
                        grup.append(x); dur += x["durada_s"]
        print(f"\n  sessions: {n_s:,} · {h_s:.2f} h -> {DIR_SESSIONS}")
    print(f"\n  {TRAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

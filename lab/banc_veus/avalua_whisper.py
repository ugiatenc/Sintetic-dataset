#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mesura objectiva del banc de veus."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline

AQUI = Path(__file__).resolve().parent
BANC = AQUI / "banc.jsonl"
MODEL = "openai/whisper-large-v3-turbo"


def norm(t: str) -> str:
    """Normalitza una transcripcio per comparar-la."""
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t)).strip()


def cer(ref: str, hyp: str) -> float:
    """Taxa d'error de caracters entre referencia i hipotesi."""
    a, b = norm(ref), norm(hyp)
    if not a:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return round(d[len(b)] / len(a), 3)


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    files = [json.loads(l) for l in BANC.open(encoding="utf-8") if l.strip()]
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=MODEL, device=dev, torch_dtype=torch.float16 if dev == 0 else torch.float32)
    for k, f in enumerate(files, 1):
        cers = []
        for j, s in enumerate(f.get("sint", []), 1):
            x, sr = sf.read(AQUI / s["audio"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                from math import gcd
                g = gcd(sr, 16000)
                x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            hyp = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "ca", "task": "transcribe"})["text"]
            c = cer(s["tts_text"], hyp)
            f[f"whisper_{j}"], f[f"cer_{j}"] = hyp.strip(), c
            cers.append(c)
        f["cer"] = round(float(np.mean(cers)), 3) if cers else None
        if k % 10 == 0:
            print(f"  {k}/{len(files)}")
    with BANC.open("w", encoding="utf-8") as fh:
        for f in files:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    amb = [f for f in files if f.get("cer") is not None]
    print(f"\nCER mitjà: {np.mean([f['cer'] for f in amb]):.3f} sobre {len(amb)} veus")
    for cat in ("aranes_declarat", "no_declarat", "test", "catala"):
        xs = [f["cer"] for f in amb if f["categoria"] == cat]
        if xs:
            print(f"  {cat:<16} n={len(xs):>3}  CER mediana {np.median(xs):.3f}  mitjana {np.mean(xs):.3f}  màx {max(xs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

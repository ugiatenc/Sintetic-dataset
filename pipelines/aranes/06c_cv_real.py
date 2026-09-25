#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 6c: clips reals d'aranes de Common Voice per al train, sense locutors ni frases del test."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import config as C
import corpus

REPO = "fsicoli/common_voice_22_0"
DIR_FONT = C.DIR_CV_CRU / "oc"
SPLITS = ("dev", "train", "other")
DUR_MIN_S, DUR_MAX_S = 1.0, 30.0


def _tsv(nom: str) -> list[dict]:
    """Baixa i llegeix un tsv de Common Voice oc."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"transcript/oc/{nom}.tsv", repo_type="dataset")
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


def normalitza_text(t: str) -> str:
    """El text tal com el vol el train: apostrof recte (com tot el corpus)."""
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nomes-comptar", action="store_true", help="informa, no copia res")
    args = p.parse_args()

    manifest_font = DIR_FONT / "manifest.jsonl"
    if not manifest_font.exists():
        raise SystemExit(f"Falta {manifest_font}: executa `06_veus.py --baixar` primer.")
    clips = [json.loads(l) for l in manifest_font.open(encoding="utf-8") if l.strip()]
    clips = [c for c in clips if c["split"] in SPLITS]
    print(f"{len(clips):,} clips als splits {SPLITS} de Common Voice oc, "
          f"{len({c['client_id'] for c in clips})} locutors, "
          f"{sum(c['durada_s'] for c in clips) / 3600:.2f} h")

    test = [json.loads(l) for l in C.MANIFEST_CV.read_text(encoding="utf-8").splitlines() if l.strip()]
    locutors_test = {t["client_id"] for t in test}
    frases_prohibides = {corpus.clau_dedup(t["text"]) for t in test}
    vots = {f["path"]: (int(f["up_votes"] or 0), int(f["down_votes"] or 0))
            for nom in SPLITS for f in _tsv(nom)}
    print(f"exclosos: {len(locutors_test)} locutors del test, {len(frases_prohibides):,} frases del test")

    motius: collections.Counter = collections.Counter()
    bons = []
    for c in clips:
        nom = Path(c["audio"]).name
        wav = DIR_FONT / "clips" / nom
        text = normalitza_text(c["text"])
        up, down = vots.get(Path(nom).with_suffix(".mp3").name, (0, 0))
        if not wav.exists():
            motius["fitxer absent"] += 1
        elif c["client_id"] in locutors_test:
            motius["locutor del test"] += 1
        elif corpus.clau_dedup(text) in frases_prohibides:
            motius["frase del test (relectura)"] += 1
        elif not (DUR_MIN_S <= c["durada_s"] <= DUR_MAX_S):
            motius[f"durada fora de {DUR_MIN_S:.0f}-{DUR_MAX_S:.0f} s"] += 1
        elif down > up:
            motius["mes vots negatius que positius"] += 1
        elif C.puntuacio_aranes(text) < C.MIN_MARQUES:
            motius["filtre dialectal (no es aranes segur)"] += 1
        else:
            fortes, foranes = C.compta_marques(text)
            bons.append({
                "id": f"cv_{Path(nom).stem}",
                "text": text, "font": f"common_voice_{c['split']}",
                "veu_id": f"cv_{c['client_id'][:12]}", "entorn_id": "real",
                "durada_s": c["durada_s"], "sample_rate": 16000, "status": "ok",
                "client_id": c["client_id"], "up_votes": up, "down_votes": down,
                "marques_fortes": fortes, "marques_foranes": foranes,
                "_origen": str(wav),
            })

    frases_train = {corpus.clau_dedup(b["text"]) for b in bons if C.split_clip(b["id"], b["client_id"]) == "train"}
    for i, b in enumerate(bons):
        split = C.split_clip(b["id"], b["client_id"])
        if split == "dev" and corpus.clau_dedup(b["text"]) in frases_train:
            split = "exclos"  # frase d'un locutor del dev que tambe llegeix un del train: ni dev ni train
        b["split"] = split
        b["status"] = "ok" if split != "exclos" else "exclos"
        b["audio"] = str(C.ruta_clip(b["id"], index=i, split="dev" if split == "dev" else "train").relative_to(ARREL))
    dev = [b for b in bons if b["split"] == "dev"]
    print(f"dev real: {len(dev)} clips · {sum(b['durada_s'] for b in dev)/60:.1f} min · {len({b['client_id'] for b in dev})} locutors -> {C.DIR_CLIPS_EVAL.relative_to(ARREL)}")
    hores = sum(b["durada_s"] for b in bons) / 3600
    print(f"\nentren {len(bons):,} clips · {hores:.2f} h · {len({b['client_id'] for b in bons})} locutors")
    print("no entren:", ", ".join(f"{m} {n:,}" for m, n in motius.most_common()))
    print("mostra:", *[f"\n   {b['text']}" for b in bons[:8]])
    if args.nomes_comptar:
        return 0

    for b in bons:
        desti = ARREL / b["audio"]
        origen = b.pop("_origen")
        if b["status"] == "exclos":
            desti.unlink(missing_ok=True); desti.with_suffix(".json").unlink(missing_ok=True)
            continue
        desti.parent.mkdir(parents=True, exist_ok=True)
        if not desti.exists():
            shutil.copy2(origen, desti)
        desti.with_suffix(".json").write_text(json.dumps({**b, "transcript": b["text"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    with C.MANIFEST_CV_REAL.open("w", encoding="utf-8") as fh:
        for b in bons:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"\n-> {C.MANIFEST_CV_REAL} ({len(bons):,} clips). Seguent: 09_dataset.py els afegeix al train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

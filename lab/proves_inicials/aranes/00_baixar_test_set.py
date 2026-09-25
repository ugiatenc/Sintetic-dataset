#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construeix un test set d'aranès amb àudio I transcripció verificada, a partir de Common Voice."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import requests

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[2]
SORTIDA_DEFECTE = ROOT / "datasets/aranes/test"

REPO = "fsicoli/common_voice_22_0"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
LOCALE = "oc"
SR = 16000

import importlib.util
_spec = importlib.util.spec_from_file_location("corpus", AQUI / "01_extreure_corpus.py")
corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus)

import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines/aranes"))
import config as C_pipeline


def sessio() -> requests.Session:
    """Sessio HTTP amb el User-Agent del projecte."""
    s = requests.Session()
    s.headers["User-Agent"] = corpus.USER_AGENT
    return s


def llegir_tsv(s: requests.Session, nom: str) -> list[dict]:
    """Baixa i parseja un tsv de Common Voice."""
    r = s.get(f"{BASE}/transcript/{LOCALE}/{nom}.tsv", timeout=120)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text), delimiter="\t"))


def es_aranes(fila: dict, filtre: str) -> bool:
    """Si una fila de Common Voice es aranesa, per accent o per text."""
    per_accent = "aran" in (fila.get("accents") or "").lower()
    per_text = corpus.puntuacio_aranes(fila["sentence"])[0] >= 1
    return {"accent": per_accent, "text": per_text, "unio": per_accent or per_text}[filtre]


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["test", "dev", "train"],
                   choices=["test", "dev", "train", "other", "invalidated"])
    p.add_argument("--filtre", choices=["accent", "text", "unio"], default="unio")
    p.add_argument("--limit", type=int, help="màxim de clips a desar")
    p.add_argument("--sortida", type=Path, default=SORTIDA_DEFECTE)
    args = p.parse_args()

    import librosa
    import soundfile as sf

    s = sessio()
    durades = {f["clip"]: int(f["duration[ms]"]) for f in llegir_tsv(s, "clip_durations")}

    clips_dir = args.sortida / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest, total_ms = [], 0

    for split in args.splits:
        files = [f for f in llegir_tsv(s, split) if es_aranes(f, args.filtre)
                 and f["client_id"] not in C_pipeline.LOCUTORS_FORA_DEL_TEST]
        volguts = {f["path"]: f for f in files}
        print(f"{split}: {len(volguts)} clips d'aranès (sense els locutors de "
              f"config.LOCUTORS_FORA_DEL_TEST)")
        if not volguts:
            continue

        url = f"{BASE}/audio/{LOCALE}/{split}/{LOCALE}_{split}_0.tar"
        r = s.get(url, timeout=300)
        r.raise_for_status()
        print(f"  tar baixat ({len(r.content) / 1e6:.1f} MB), extraient...")

        with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
            for membre in tar.getmembers():
                nom = Path(membre.name).name
                fila = volguts.get(nom)
                if not fila or (args.limit and len(manifest) >= args.limit):
                    continue
                ona, _ = librosa.load(tar.extractfile(membre), sr=SR, mono=True)
                desti = clips_dir / (Path(nom).stem + ".wav")
                sf.write(desti, ona, SR, subtype="PCM_16")
                total_ms += durades.get(nom, 0)
                manifest.append({
                    "audio": str(desti.relative_to(args.sortida)),
                    "text": fila["sentence"].strip(),
                    "accents": fila.get("accents") or "",
                    "client_id": fila["client_id"],
                    "split": split,
                    "durada_s": round(durades.get(nom, 0) / 1000, 2),
                })

    ruta = args.sortida / "manifest.jsonl"
    with ruta.open("w", encoding="utf-8") as fh:
        for entrada in manifest:
            fh.write(json.dumps(entrada, ensure_ascii=False) + "\n")

    parlants = len({m["client_id"] for m in manifest})
    print(f"\n{len(manifest)} clips | {total_ms / 3.6e6:.2f} h | {parlants} locutors")
    print(f"-> {ruta}")
    for m in manifest[:3]:
        print(f"  · [{m['durada_s']:.1f}s] {m['text'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

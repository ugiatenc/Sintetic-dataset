#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banc de proves de veus (21/09): per escoltar cada locutor de Common Voice `oc` i el que OmniVoice en fa."""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(ARREL / "pipelines/aranes"))

import build_voice_bank as bvb  # noqa: E402
import config as C  # noqa: E402
import respelling  # noqa: E402

V6 = importlib.import_module("06_veus")

DIR_FONT = C.DIR_CV_CRU / "oc"
DIR_BANC = C.DIR_VEUS
DIR_CA = C.DIR_RESERVA_CA
DIR_TEST = C.MANIFEST_CV.parent
AUDIO = AQUI / "audio"
BANC = AQUI / "banc.jsonl"
SR_TTS = 24000

FRASES = [
    "Eth Conselh Generau d'Aran a aprovat eth pressupòst entà dus mil vint-e-cinc, damb ua "
    "pujada deth dètz per cent.",
    "Deman de maitin haram ua caminada enquiath lac de Saboredo, se non plò e non hè massa hered.",
]
TTS_TEXT = [respelling.reescriu(f) for f in FRASES]
ACCENTS_CA = ["central", "nord-occidental", "valencià", "balear", "septentrional"]


def _snapshot_tsv(nom: str) -> list[dict]:
    """Llegeix un tsv de Common Voice de la cache local."""
    p = glob.glob(str(ARREL / ".hf_cache/hub/datasets--fsicoli--common_voice_22_0/snapshots/*/transcript/oc" / f"{nom}.tsv"))
    return list(csv.DictReader(open(p[0], encoding="utf-8"), delimiter="\t")) if p else []


def _ruta_existent(ruta: str, carpeta: Path) -> Path | None:
    """Les rutes dels manifests poden ser d'abans de reordenar `data/`: es resol pel nom."""
    p = Path(ruta)
    if p.exists():
        return p
    alt = carpeta / p.name
    return alt if alt.exists() else None


def locutors_oc() -> dict:
    """client_id -> dades agregades (splits, accent, gènere, edat) i clips amb àudio."""
    info: dict = collections.defaultdict(lambda: {"splits": collections.Counter(), "accents": collections.Counter(),
                                                  "gender": set(), "age": set(), "clips": []})
    for sp in ("train", "dev", "test", "other"):
        for f in _snapshot_tsv(sp):
            d = info[f["client_id"]]
            d["splits"][sp] += 1
            d["accents"][(f.get("accents") or "").strip()] += 1
            if f.get("gender"):
                d["gender"].add(f["gender"])
            if f.get("age"):
                d["age"].add(f["age"])
    for c in (json.loads(l) for l in (DIR_FONT / "manifest.jsonl").open(encoding="utf-8") if l.strip()):
        p = _ruta_existent(c["audio"], DIR_FONT / "clips")
        if p:
            info[c["client_id"]]["clips"].append({"path": str(p), "text": c["text"], "durada_s": c["durada_s"], "split": c["split"]})
    for t in (json.loads(l) for l in C.MANIFEST_CV.open(encoding="utf-8") if l.strip()):
        p = DIR_TEST / t["audio"]
        if p.exists():
            info[t["client_id"]]["clips"].append({"path": str(p), "text": t["text"], "durada_s": t["durada_s"], "split": t["split"]})
    return {k: v for k, v in info.items() if v["clips"]}


def tria_referencia(clips: list[dict], vad) -> tuple[dict, dict, np.ndarray]:
    """El millor clip segons `06_veus.avaluar_clip` entre els 10 més propers a 4 s."""
    cands = sorted(clips, key=lambda c: abs(c["durada_s"] - 4.0))[:10]
    millor = None
    for c in cands:
        s, meta = V6.avaluar_clip(c["path"], vad, None)
        if s is not None and (millor is None or meta["nota"] > millor[1]["nota"]):
            millor = (c, meta, s)
    if millor:
        c, meta, s = millor
        return c, {k: v for k, v in meta.items() if k != "embedding"}, s
    c = min(clips, key=lambda c: abs(c["durada_s"] - 5.0))
    ona, sr = sf.read(c["path"], dtype="float32")
    return c, {"nota": None, "avis": "cap clip passa els llindars del pas 6 (SNR 20 dB, parla 70 %, 2,5-8 s)"}, bvb._a_mono_16k(ona, sr)


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="només els N primers locutors oc (prova)")
    p.add_argument("--sense-tts", action="store_true", help="només referències, sense sintetitzar")
    args = p.parse_args()
    (AUDIO / "ref").mkdir(parents=True, exist_ok=True)
    (AUDIO / "sint").mkdir(parents=True, exist_ok=True)

    test_ids = {json.loads(l)["client_id"] for l in C.MANIFEST_CV.open(encoding="utf-8") if l.strip()}
    banc = {b["client_id"]: b for b in (json.loads(l) for l in (DIR_BANC / "manifest.jsonl").open(encoding="utf-8") if l.strip())}
    vad = bvb.DetectorVeu()
    files: list[dict] = []

    oc = locutors_oc()
    ordre = sorted(oc, key=lambda k: -len(oc[k]["clips"]))
    if args.limit:
        ordre = ordre[:args.limit]
    print(f"{len(oc)} locutors oc amb àudio; se'n processen {len(ordre)}")
    for i, cid in enumerate(ordre, 1):
        d = oc[cid]
        sid = f"oc_{cid[:12]}"
        accent = ", ".join(sorted(a for a in d["accents"] if a)) or "(cap)"
        b = banc.get(cid)
        ref_banc = _ruta_existent(b["ref_audio"], DIR_BANC) if b else None
        if b and ref_banc:
            ona, sr = sf.read(ref_banc, dtype="float32")
            s = bvb._a_mono_16k(ona, sr)
            ref = {"path": str(ref_banc), "text": b["ref_text"], "durada_s": b["durada_s"], "split": b.get("split", "?")}
            meta = {k: b.get(k) for k in ("nota", "snr_db", "ratio_parla", "transitoris_per_min", "constancia", "durada_s")}
            meta["origen_ref"] = "banc"
        else:
            ref, meta, s = tria_referencia(d["clips"], vad)
            meta["origen_ref"] = "triat aquí"
        ref_out = AUDIO / "ref" / f"{sid}.wav"
        sf.write(ref_out, s, bvb.FS_ANALISI, subtype="PCM_16")
        cat = ("test" if cid in test_ids else
               "aranes_declarat" if "aran" in accent.lower() else "no_declarat")
        files.append({
            "id": sid, "locale": "oc", "client_id": cid, "categoria": cat,
            "al_test": cid in test_ids, "al_banc": b is not None,
            "splits": dict(d["splits"]), "n_clips_audio": len(d["clips"]),
            "accents": accent, "gender": ", ".join(sorted(d["gender"])) or "?", "age": ", ".join(sorted(d["age"])) or "?",
            "ref_audio": str(ref_out.relative_to(AQUI)), "ref_text": ref["text"], "ref_split": ref["split"], **meta,
            "sint": [],
        })
        print(f"  [{i}/{len(ordre)}] {sid} {cat:<16} {accent[:30]:<30} clips={len(d['clips']):<4} nota={meta.get('nota')}")

    ca = [json.loads(l) for l in (DIR_CA / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    for acc in ACCENTS_CA:
        cands = sorted((v for v in ca if (v.get("accents") or "").strip().lower() == acc), key=lambda v: -(v.get("nota") or 0))
        for v in cands:
            r = _ruta_existent(v["ref_audio"], DIR_CA)
            if r:
                ona, sr = sf.read(r, dtype="float32")
                s = bvb._a_mono_16k(ona, sr)
                sid = f"ca_{v['client_id'][:12]}"
                ref_out = AUDIO / "ref" / f"{sid}.wav"
                sf.write(ref_out, s, bvb.FS_ANALISI, subtype="PCM_16")
                files.append({
                    "id": sid, "locale": "ca", "client_id": v["client_id"], "categoria": "catala",
                    "al_test": False, "al_banc": False, "splits": {v.get("split", "dev"): v.get("clips_totals", 1)},
                    "n_clips_audio": v.get("clips_totals", 1), "accents": acc, "gender": v.get("gender") or "?", "age": v.get("age") or "?",
                    "ref_audio": str(ref_out.relative_to(AQUI)), "ref_text": v["ref_text"], "ref_split": v.get("split", "dev"),
                    **{k: v.get(k) for k in ("nota", "snr_db", "ratio_parla", "transitoris_per_min", "constancia", "durada_s")},
                    "origen_ref": "banc català", "sint": [],
                })
                print(f"  [ca] {sid} {acc} nota={v.get('nota')}")
                break

    if not args.sense_tts:
        import torch
        from omnivoice import OmniVoice
        tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
        for k, f in enumerate(files, 1):
            ref_path = str(AQUI / f["ref_audio"])
            try:
                torch.manual_seed(7)
                ones = tts.generate(text=TTS_TEXT, language=C.IDIOMA_TTS,
                                    ref_audio=[ref_path] * len(TTS_TEXT), ref_text=[f["ref_text"]] * len(TTS_TEXT))
                for j, (ona, raw, tts_t) in enumerate(zip(ones, FRASES, TTS_TEXT), 1):
                    ona = np.asarray(ona, dtype="float32")
                    out = AUDIO / "sint" / f"{f['id']}_{j}.wav"
                    sf.write(out, ona, SR_TTS, subtype="PCM_16")
                    f["sint"].append({"audio": str(out.relative_to(AQUI)), "raw_text": raw, "tts_text": tts_t,
                                      "durada_s": round(ona.size / SR_TTS, 2)})
            except Exception as e:  # noqa: BLE001
                f["error_tts"] = f"{type(e).__name__}: {e}"[:200]
            if k % 10 == 0:
                print(f"  síntesi {k}/{len(files)}")

    with BANC.open("w", encoding="utf-8") as fh:
        for f in files:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"\n{len(files)} locutors -> {BANC}")
    print("frases:", *[f"\n  RAW {r}\n  TTS {t}" for r, t in zip(FRASES, TTS_TEXT)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

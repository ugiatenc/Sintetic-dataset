#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 8: pla determinista (veu, entorn, velocitat per frase) i generacio de l'audio amb OmniVoice i el simulador acustic."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import numpy as np

import acoustic_sim as ac
import config as C
import corpus
import entorns as E

DIR_DATASET = C.DIR_DATASET
PLA = DIR_DATASET / "pla.jsonl"
MANIFEST = DIR_DATASET / "manifest.jsonl"
CORPUS = C.DIR_TEXT / "corpus.jsonl"
VEUS = C.DIR_AUDIO / "veus/manifest.jsonl"
SEED = 42
SR_TTS = 24000
SR_WHISPER = 16000
DURADA_MAX_S = 29.0  # la finestra de Whisper son 30 s
TTS_MODEL_ID = "k2-fsa/OmniVoice"


def triar_dispositiu(preferit: str = "auto") -> str:
    """Primera GPU que aquesta build de torch pugui fer servir de debo (la 1080 Ti no)."""
    import torch
    if preferit != "auto":
        return preferit
    if not torch.cuda.is_available():
        return "cpu"
    for i in range(torch.cuda.device_count()):
        try:
            (torch.zeros(8, device=f"cuda:{i}") + 1).sum().item()
            return f"cuda:{i}"
        except Exception:
            continue
    return "cpu"


def planificar(args) -> int:
    """Assigna veu, entorn, velocitat i llavor a cada frase i escriu el pla."""
    frases = corpus.llegeix_jsonl(CORPUS)
    veus = corpus.llegeix_jsonl(VEUS)
    if not frases or not veus:
        raise SystemExit("Falten corpus.jsonl (pas 5) o veus/manifest.jsonl (pas 6)")
    rng = np.random.default_rng(SEED)
    frases = [frases[i] for i in rng.permutation(len(frases))]
    if args.ordre == "fiables-primer":
        frases = ordenar_fiables_primer(frases, args.ritme_fiables)
    veus_assignades = ac.mostreig_round_robin(veus, len(frases), rng)
    per_font = collections.defaultdict(list)
    for i, f in enumerate(frases):
        per_font[f["font"]].append(i)
    entorn_de = {}
    for font, idx in per_font.items():
        compatibles = ac.expandir_per_pes(ac.entorns_per_style(font, E.ENTORNS_ACTIUS))
        for i, e in zip(idx, ac.mostreig_round_robin(compatibles, len(idx), rng)):
            entorn_de[i] = e.id
    DIR_DATASET.mkdir(parents=True, exist_ok=True)
    with PLA.open("w", encoding="utf-8") as fh:
        for i, (f, v) in enumerate(zip(frases, veus_assignades)):
            fh.write(json.dumps({
                "id": f["id"], "font": f["font"], "raw_text": f["raw_text"],
                "tts_text": f["tts_text"], "durada_est_s": f["durada_est_s"],
                "veu_id": v["id"], "entorn_id": entorn_de[i],
                "speed": round(float(rng.uniform(0.94, 1.06)), 3),
                "llavor": ac.llavor_estable(SEED, f["id"]) % (2 ** 31),
            }, ensure_ascii=False) + "\n")
    print(f"Pla: {len(frases):,} frases, {len(veus)} veus, {len(E.ENTORNS_ACTIUS)} entorns -> {PLA}")
    for titol, comptador in (("veu", collections.Counter(v["id"] for v in veus_assignades)),
                             ("entorn", collections.Counter(entorn_de.values()))):
        print(f"\n  per {titol}:")
        for k, n in comptador.most_common(): print(f"    {k:<28} {n:>7,}")
    h = sum(f["durada_est_s"] for f in frases) / 3600
    print(f"\n  hores estimades del pla: {h:.1f}  (ordre: {args.ordre})")
    fiables = set(C.FONTS_FIABLES)
    acum_tot = acum_fi = 0.0
    fites = [25, 50, 100, 162, 250, 400]
    for f in frases:
        acum_tot += f["durada_est_s"] / 3600
        acum_fi += f["durada_est_s"] / 3600 if f["font"] in fiables else 0.0
        if fites and acum_tot >= fites[0]:
            print(f"  a les {fites[0]:>3} h de pla: {acum_fi:6.1f} h de fonts fiables, {acum_tot - acum_fi:6.1f} h de reserva")
            fites.pop(0)
    return 0


def ordenar_fiables_primer(frases: list[dict], ritme: float) -> list[dict]:
    """Intercala les dues capes (fonts fiables / reserva) per HORES."""
    fiables = set(C.FONTS_FIABLES)
    acum = {True: 0.0, False: 0.0}
    claus = []
    for f in frases:
        capa = f["font"] in fiables
        acum[capa] += f["durada_est_s"]
        claus.append(acum[capa] / (ritme if capa else 1.0))
    return [frases[i] for i in sorted(range(len(frases)), key=claus.__getitem__)]


def generar(args) -> int:
    """Genera l'audio de les entrades del pla que encara no en tenen."""
    import soundfile as sf
    import soxr
    import torch
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()

    pla = corpus.llegeix_jsonl(PLA)
    if not pla:
        raise SystemExit("No hi ha pla. Executa --planificar primer.")
    veus = {v["id"]: v for v in corpus.llegeix_jsonl(VEUS)}
    for v in veus.values():
        alt = VEUS.parent / Path(v["ref_audio"]).name
        if not Path(v["ref_audio"]).exists() and alt.exists():
            v["ref_audio"] = str(alt)
    fets = {m["id"] for m in corpus.llegeix_jsonl(MANIFEST) if m.get("status") in ("ok", "massa_llarg", "exclos")}
    pendents = [p for p in pla if p["id"] not in fets]
    frases_dev = C.frases_dev()
    exclosos = [p for p in pendents if C.split_clip(p["id"], text=p["raw_text"], frases_dev=frases_dev) == "exclos"]
    if exclosos:  # la frase tambe es al dev: no es genera, i al manifest queda com a `exclos`
        with MANIFEST.open("a", encoding="utf-8") as fh:
            for p in exclosos:
                fh.write(json.dumps({**p, "status": "exclos", "split": "exclos"}, ensure_ascii=False) + "\n")
        ids_exclosos = {p["id"] for p in exclosos}
        pendents = [p for p in pendents if p["id"] not in ids_exclosos]
        print(f"{len(exclosos):,} frases excloses (tambe son al dev)")
    if args.hores:
        objectiu = args.hores * 3600 - sum(m["durada_s"] for m in corpus.llegeix_jsonl(MANIFEST) if m.get("status") == "ok")
        acumulat, tall = 0.0, len(pendents)
        for k, p in enumerate(pendents):
            acumulat += p["durada_est_s"]
            if acumulat >= objectiu:
                tall = k + 1
                break
        pendents = pendents[:tall]
    if args.limit:
        pendents = pendents[:args.limit]
    print(f"pla {len(pla):,} | ja fets {len(fets):,} | ara: {len(pendents):,} "
          f"({sum(p['durada_est_s'] for p in pendents) / 3600:.1f} h est.)")
    if not pendents:
        return 0

    dispositiu = triar_dispositiu(args.dispositiu)
    print(f"dispositiu: {dispositiu}")
    from omnivoice import OmniVoice
    tts = OmniVoice.from_pretrained(TTS_MODEL_ID, device_map=dispositiu, dtype=torch.float16)
    carpetes_fetes: set[Path] = set()

    t0, n_ok, n_llarg, n_err, s_audio = time.time(), 0, 0, 0, 0.0
    n_blocs = (len(pendents) + args.bloc - 1) // args.bloc
    with MANIFEST.open("a", encoding="utf-8") as fh:
        for b in range(n_blocs):
            per_veu = collections.defaultdict(list)
            for p in pendents[b * args.bloc:(b + 1) * args.bloc]:
                per_veu[p["veu_id"]].append(p)
            for veu_id, llista in per_veu.items():
                v = veus[veu_id]
                llista.sort(key=lambda p: p["durada_est_s"])  # lots de durada semblant: menys farciment a la GPU
                for ini in range(0, len(llista), args.lot):
                    lot = llista[ini:ini + args.lot]
                    torch.manual_seed(lot[0]["llavor"])
                    try:
                        ones = tts.generate(text=[p["tts_text"] for p in lot], language=C.IDIOMA_TTS,
                                            ref_audio=[v["ref_audio"]] * len(lot),
                                            ref_text=[v["ref_text"]] * len(lot),
                                            speed=[p["speed"] for p in lot])
                    except Exception as exc:
                        for p in lot:
                            fh.write(json.dumps({**p, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]},
                                                ensure_ascii=False) + "\n")
                        fh.flush()
                        n_err += len(lot)
                        print(f"  ERROR lot {veu_id} {ini}: {exc}", flush=True)
                        continue
                    for p, ona in zip(lot, ones):
                        ona = np.asarray(ona, dtype="float32")
                        dur = ona.size / SR_TTS
                        if dur > DURADA_MAX_S or ona.size == 0:
                            fh.write(json.dumps({**p, "status": "massa_llarg", "durada_s": round(dur, 2)},
                                                ensure_ascii=False) + "\n")
                            n_llarg += 1
                            continue
                        rng = np.random.default_rng(ac.llavor_estable(SEED, p["id"], p["entorn_id"]) % (2 ** 31))
                        mescla, meta = ac.mesclar_amb_entorn(ona, SR_TTS, E.ENTORNS_ACTIUS[p["entorn_id"]], rng)
                        x16 = soxr.resample(mescla, SR_TTS, SR_WHISPER, quality="VHQ")
                        ruta = C.ruta_clip(p["id"])
                        if ruta.parent not in carpetes_fetes:
                            ruta.parent.mkdir(parents=True, exist_ok=True)
                            carpetes_fetes.add(ruta.parent)
                        sf.write(ruta, x16, SR_WHISPER, subtype="PCM_16")
                        registre = {
                            "id": p["id"], "audio": str(ruta.relative_to(ARREL)), "text": p["raw_text"],
                            "tts_text": p["tts_text"], "font": p["font"], "veu_id": veu_id,
                            "ref_text": v["ref_text"], "speed": p["speed"], "llavor": p["llavor"],
                            **{k: meta[k] for k in ("entorn_id", "sala", "rir_path", "ambient", "snr_db", "canal", "nivell_dbfs")},
                            "durada_s": round(dur, 3), "sample_rate": SR_WHISPER, "status": "ok",
                            "split": C.split_clip(p["id"]),
                        }
                        ruta.with_suffix(".json").write_text(json.dumps({**registre, "transcript": registre["text"]}, ensure_ascii=False, indent=1), encoding="utf-8")  # `transcript`: el camp que llegeix l'script d'entrenament
                        fh.write(json.dumps(registre, ensure_ascii=False) + "\n")
                        fh.flush()
                        n_ok += 1
                        s_audio += dur
                    fets_ara = n_ok + n_llarg + n_err
                    if fets_ara % (args.lot * 25) == 0 or fets_ara == len(pendents):
                        t = time.time() - t0
                        print(f"  {fets_ara:>6}/{len(pendents)}  ok {n_ok}  llargs {n_llarg}  err {n_err}  "
                              f"| {s_audio / 3600:.2f} h d'audio en {t / 60:.1f} min  "
                              f"({s_audio / max(1, t):.1f} s d'audio per s)", flush=True)
            t = time.time() - t0
            resta_s = sum(p["durada_est_s"] for p in pendents[(b + 1) * args.bloc:])
            print(f"### bloc {b + 1}/{n_blocs} acabat {time.strftime('%d/%m %H:%M')}: {s_audio / 3600:.2f} h d'audio en "
                  f"{t / 3600:.2f} h de GPU (RTF {t / max(1, s_audio):.2f}); queden {resta_s / 3600:.0f} h d'audio, "
                  f"~{resta_s * t / max(1, s_audio) / 3600:.0f} h de GPU", flush=True)
    t = time.time() - t0
    rtf = t / max(1, s_audio)
    resta = sum(p["durada_est_s"] for p in pla if p["id"] not in fets) - s_audio
    print(f"\n{n_ok} ok, {n_llarg} massa llargs, {n_err} errors | {s_audio / 3600:.2f} h en {t / 3600:.2f} h de GPU "
          f"(RTF {rtf:.2f})\n  projeccio per a la resta del pla ({resta / 3600:.0f} h d'audio): "
          f"{resta * rtf / 3600:.0f} h de GPU\n  manifest: {MANIFEST}")
    return 0


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planificar", action="store_true")
    p.add_argument("--generar", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--hores", type=float)
    p.add_argument("--lot", type=int, default=4)
    p.add_argument("--bloc", type=int, default=2000,
                   help="entrades del pla per bloc: l'ordre del pla es respecta bloc a bloc (i dins del bloc, per veu)")
    p.add_argument("--ordre", choices=("fiables-primer", "barrejat"), default="fiables-primer",
                   help="--planificar: fonts fiables al davant a ritme --ritme-fiables (per defecte), o permutacio uniforme")
    p.add_argument("--ritme-fiables", type=float, default=2.0,
                   help="hores de fonts fiables per cada hora de reserva mentre en quedin (2 = dos tercos / un terc)")
    p.add_argument("--dispositiu", default="auto")
    args = p.parse_args()
    if args.planificar:
        planificar(args)
    if args.generar:
        generar(args)
    if not (args.planificar or args.generar):
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

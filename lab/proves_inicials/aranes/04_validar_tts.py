#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 1: val OmniVoice com a font de dades sintètiques d'aranès?"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MANIFEST_DEFECTE = ROOT / "datasets/aranes/test/manifest.jsonl"
DIR_SORTIDA = ROOT / "lab/proves_inicials/aranes/audios/validacio_tts"
RESULTATS = AQUI / "resultats_validacio_tts.json"
TTS_MODEL_ID = "k2-fsa/OmniVoice"
LLAVOR = 1234

_spec = importlib.util.spec_from_file_location("whisper03", AQUI / "03_provar_whisper.py")
w03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w03)


def carregar_manifest(ruta: Path) -> list[dict]:
    """Carrega el manifest amb el lector del pas 03."""
    return w03.carregar_manifest(ruta, None)


def triar_veus(entrades: list[dict], n: int) -> list[dict]:
    """Un clip de referència per locutor, de `train`/`dev` i amb accent aranès declarat."""
    per_locutor = defaultdict(list)
    for e in entrades:
        if e["split"] in ("train", "dev") and "aran" in e["accents"].lower():
            per_locutor[e["client_id"]].append(e)
    veus = []
    for locutor, clips in sorted(per_locutor.items(), key=lambda kv: -len(kv[1])):
        bons = [c for c in clips if 4.0 <= c["durada_s"] <= 10.0]
        if bons:
            ref = min(bons, key=lambda c: abs(c["durada_s"] - 7.0))
            veus.append({"id": f"veu{len(veus):02d}", "client_id": locutor,
                         "ref_audio": ref["audio"], "ref_text": ref["text"]})
        if len(veus) == n:
            break
    return veus


def ruta_sintetic(frase: dict, i: int, veus: list[dict], idioma_tts: str) -> Path:
    """Ruta del .wav sintètic de la frase i."""
    clip = Path(frase["audio"]).stem.removeprefix("common_voice_oc_")
    return DIR_SORTIDA / idioma_tts / f"cv{clip}_tts-{idioma_tts}_{veus[i % len(veus)]['id']}.wav"


def sintetitzar(frases: list[dict], veus: list[dict], idioma_tts: str, dispositiu: str,
                mida_lot: int) -> None:
    """Genera els .wav que falten."""
    import numpy as np
    import soundfile as sf
    import torch

    desti = DIR_SORTIDA / idioma_tts
    desti.mkdir(parents=True, exist_ok=True)
    pendents = [(i, f) for i, f in enumerate(frases)
                if not ruta_sintetic(f, i, veus, idioma_tts).exists()]
    print(f"\n[TTS {idioma_tts}] {len(frases) - len(pendents)} ja fets, {len(pendents)} per generar")
    if not pendents:
        return

    from omnivoice import OmniVoice
    tts = OmniVoice.from_pretrained(TTS_MODEL_ID, device_map=dispositiu, dtype=torch.float16)

    per_veu = defaultdict(list)
    for i, f in pendents:
        per_veu[i % len(veus)].append((i, f))
    fets = 0
    for idx_veu, llista in per_veu.items():
        veu = veus[idx_veu]
        for inici in range(0, len(llista), mida_lot):
            lot = llista[inici:inici + mida_lot]
            torch.manual_seed(LLAVOR + idx_veu + inici)
            ones = tts.generate(
                text=[f["text"] for _, f in lot],
                language=idioma_tts,
                ref_audio=[veu["ref_audio"]] * len(lot),
                ref_text=[veu["ref_text"]] * len(lot),
            )
            for (i, f), ona in zip(lot, ones):
                sf.write(ruta_sintetic(f, i, veus, idioma_tts), np.asarray(ona, dtype="float32"),
                         tts.sampling_rate, subtype="PCM_16")
            fets += len(lot)
            print(f"  {fets}/{len(pendents)}")

    del tts
    torch.cuda.empty_cache()


def spearman(a: list[float], b: list[float]) -> float:
    """Correlació de rangs sense scipy."""
    import numpy as np
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=MANIFEST_DEFECTE)
    p.add_argument("--limit", type=int, help="només les N primeres frases del test")
    p.add_argument("--n-veus", type=int, default=4, help="veus de referència a clonar (defecte: 4)")
    p.add_argument("--tts-idiomes", nargs="+", default=["oc"],
                   help="`language` passat a OmniVoice, un joc d'àudios per valor (defecte: oc)")
    p.add_argument("--asr-idioma", default="oc", help="`language` de Whisper (defecte: oc)")
    p.add_argument("--lot-tts", type=int, default=4)
    p.add_argument("--mida-lot", type=int, default=8, help="lot d'ASR")
    p.add_argument("--dispositiu", default="auto")
    p.add_argument("--mostres", type=int, default=6)
    p.add_argument("--entorn", help="entorn de src/acoustic_sim.py aplicat al sintètic abans "
                                    "de l'ASR (p.ex. redaccio). Serveix per separar 'el TTS "
                                    "pronuncia més fàcil' de 'el TTS és més net que un micro "
                                    "casolà'. No toca els .wav desats")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"No hi ha {args.manifest}. Executa abans 00_baixar_test_set.py.", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore", message="(?s).*1080.*")
    from transformers.utils import logging as tlog
    tlog.disable_progress_bar()
    entrades = carregar_manifest(args.manifest)
    frases = [e for e in entrades if e["split"] == "test"][:args.limit]
    veus = triar_veus(entrades, args.n_veus)
    locutors_test = {f["client_id"] for f in frases}
    assert not locutors_test & {v["client_id"] for v in veus}, "veu de referència present al test"

    print(f"{len(frases)} frases de test ({len(locutors_test)} locutors) | {len(veus)} veus clonades:")
    for v in veus:
        print(f"  {v['id']}  {Path(v['ref_audio']).name}  «{v['ref_text'][:60]}»")
    if len(veus) < args.n_veus:
        print(f"  avís: només hi ha {len(veus)} locutors aranesos declarats amb clips de 4-10 s")

    dispositiu = w03.triar_dispositiu(args.dispositiu)

    for idioma_tts in args.tts_idiomes:
        sintetitzar(frases, veus, idioma_tts, dispositiu, args.lot_tts)

    import librosa
    import torch
    from transformers import logging as tlog
    from transformers import pipeline
    tlog.set_verbosity_error()
    print(f"\nCarregant {w03.MODEL_DEFECTE}...")
    asr = pipeline("automatic-speech-recognition", model=w03.MODEL_DEFECTE,
                   dtype=torch.float16 if dispositiu.startswith("cuda") else torch.float32,
                   device=dispositiu)

    def transcriure(rutes: list[str], degradar: bool = False) -> list[str]:
        """Transcriu una llista de wavs, opcionalment degradats amb l'entorn."""
        if degradar and args.entorn:
            ones = [degradat(r) for r in rutes]
        else:
            ones = [librosa.load(r, sr=w03.SR_ASR, mono=True)[0] for r in rutes]
        return w03.transcriure_lot(asr, ones, args.asr_idioma, args.mida_lot)

    def degradat(ruta: str):
        """El .wav sintètic (24 kHz) per l'entorn i a 16 kHz."""
        import numpy as np
        import soxr
        import acoustic_sim
        ona, sr = librosa.load(ruta, sr=None, mono=True)
        rng = np.random.default_rng(acoustic_sim.llavor_estable(args.entorn, Path(ruta).name))
        mix, _ = acoustic_sim.mesclar_amb_entorn(ona, sr, args.entorn, rng)
        return soxr.resample(mix, sr, w03.SR_ASR).astype("float32")

    refs = [f["text"] for f in frases]
    print(f"[ASR] real ({len(frases)} clips)...")
    jocs = {"real": {"rutes": [f["audio"] for f in frases]}}
    for idioma_tts in args.tts_idiomes:
        jocs[f"sintetic_{idioma_tts}"] = {
            "rutes": [str(ruta_sintetic(f, i, veus, idioma_tts)) for i, f in enumerate(frases)]}
    for nom, joc in jocs.items():
        if nom != "real":
            print(f"[ASR] {nom}...")
        joc["hipotesis"] = transcriure(joc["rutes"], degradar=nom != "real")
        joc["metriques"] = w03.metriques(refs, joc["hipotesis"])
        joc["cer_frase"] = [w03.metriques(r, h).get("CER", 0.0) for r, h in zip(refs, joc["hipotesis"])]

    real = jocs["real"]
    etiqueta = f", sintètic amb entorn '{args.entorn}'" if args.entorn else ""
    print(f"\n{'=' * 78}\n  VALIDACIÓ DEL TTS  ({len(frases)} frases, Whisper language=\"{args.asr_idioma}\"{etiqueta})\n{'=' * 78}")
    print(f"  {'':<14} {'WER':>6} {'CER':>6}   {'Spearman CER':>12}   {'acord hip. (CER)':>16}")
    print(f"  {'real':<14} {real['metriques']['WER']:>6.3f} {real['metriques']['CER']:>6.3f}")
    resum = {"n_frases": len(frases), "veus": veus, "asr_idioma": args.asr_idioma,
             "entorn": args.entorn,
             "real": real["metriques"]}
    for nom, joc in jocs.items():
        if nom == "real":
            continue
        rho = spearman(real["cer_frase"], joc["cer_frase"])
        acord = w03.metriques(real["hipotesis"], joc["hipotesis"])["CER"]
        m = joc["metriques"]
        print(f"  {nom:<14} {m['WER']:>6.3f} {m['CER']:>6.3f}   {rho:>12.2f}   {acord:>16.3f}")
        resum[nom] = m | {"spearman_cer_frase": rho, "cer_entre_hipotesis": acord}

    print("\n  Com llegir-ho: sintètic ≈ real en WER/CER, Spearman clarament positiu i acord")
    print("  baix (poca diferència entre hipòtesis) => el TTS és un substitut plausible.")
    print("  Sintètic molt per sota del real => el TTS diu una cosa més fàcil que l'aranès real.")

    print(f"\n{'-' * 78}\n  Exemples (escolta'ls: les rutes són al .json de resultats)\n{'-' * 78}")
    for i in range(min(args.mostres, len(frases))):
        print(f"\n  REF        {refs[i]}")
        for nom, joc in jocs.items():
            print(f"  {nom:<10} {joc['hipotesis'][i]}")

    resum["frases"] = [
        {"text": refs[i], **{f"{nom}_audio": joc["rutes"][i] for nom, joc in jocs.items()},
         **{f"{nom}_hipotesi": joc["hipotesis"][i] for nom, joc in jocs.items()}}
        for i in range(len(frases))]
    sortida = RESULTATS.with_name(f"{RESULTATS.stem}_{args.entorn}.json") if args.entorn else RESULTATS
    sortida.write_text(json.dumps(resum, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDesat a {sortida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

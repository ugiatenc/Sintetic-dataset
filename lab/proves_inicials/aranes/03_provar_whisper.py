#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Línia base d'ASR per a l'aranès: Whisper large-v3-turbo sobre el mateix àudio."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[2]

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

AUDIO_DEFECTE = AQUI / "referencia_aranes.wav"
DATASET_DEFECTE = ROOT / "datasets/aranes/test/manifest.jsonl"
MODEL_DEFECTE = "openai/whisper-large-v3-turbo"
SR_ASR = 16000


def triar_dispositiu(preferit: str = "auto") -> str:
    """Primera GPU que aquesta build de PyTorch pugui fer servir de debò."""
    import torch
    if preferit != "auto":
        return preferit
    if not torch.cuda.is_available():
        return "cpu"
    for i in range(torch.cuda.device_count()):
        try:
            (torch.zeros(8, device=f"cuda:{i}") + 1).sum().item()
            print(f"Dispositiu: cuda:{i} ({torch.cuda.get_device_name(i)})")
            return f"cuda:{i}"
        except Exception as e:
            print(f"  cuda:{i} ({torch.cuda.get_device_name(i)}) descartada: {type(e).__name__}")
    return "cpu"


def carregar_audio(ruta: Path):
    """Mono a 16 kHz."""
    import librosa
    ona, sr = librosa.load(str(ruta), sr=SR_ASR, mono=True)
    print(f"Àudio: {ruta.name} | {len(ona) / sr:.1f} s | {sr} Hz")
    return ona


FALLBACKS = {
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_prev_tokens": False,
}


def transcriure(asr, ona, idioma: str, chunk_s: float | None) -> str:
    """Una transcripció, amb l'idioma forçat."""
    kwargs = {"generate_kwargs": {"language": idioma, "task": "transcribe", **FALLBACKS},
              "return_timestamps": True}
    if chunk_s:
        kwargs["chunk_length_s"] = chunk_s
    sortida = asr({"raw": ona, "sampling_rate": SR_ASR}, **kwargs)
    return sortida["text"].strip()


def transcriure_lot(asr, ones: list, idioma: str, mida_lot: int) -> list[str]:
    """Tot el dataset en una crida."""
    entrades = [{"raw": o, "sampling_rate": SR_ASR} for o in ones]
    sortides = asr(entrades, batch_size=mida_lot,
                   generate_kwargs={"language": idioma, "task": "transcribe"})
    return [s["text"].strip() for s in sortides]


def metriques(referencia, hipotesi) -> dict:
    """WER i CER amb la normalització mínima raonable: minúscules i sense puntuació."""
    import jiwer

    net = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    ref, hip = net(referencia), net(hipotesi)
    if not (ref if isinstance(ref, str) else "".join(ref)).strip():
        return {}
    return {
        "WER": jiwer.wer(ref, hip),
        "CER": jiwer.cer(ref, hip),
    }


def carregar_manifest(ruta: Path, limit: int | None) -> list[dict]:
    """Les línies del manifest de `00_baixar_test_set.py`."""
    import json
    entrades = []
    for linia in ruta.read_text(encoding="utf-8").splitlines():
        if linia.strip():
            e = json.loads(linia)
            e["audio"] = str((ruta.parent / e["audio"]).resolve())
            entrades.append(e)
    return entrades[:limit] if limit else entrades


def avaluar_dataset(asr, entrades: list[dict], idiomes: list[str], mida_lot: int) -> dict:
    """WER i CER de corpus per a cada idioma forçat, sobre el mateix conjunt de clips."""
    import librosa

    print(f"Carregant {len(entrades)} clips...")
    ones = [librosa.load(e["audio"], sr=SR_ASR, mono=True)[0] for e in entrades]
    durada = sum(len(o) for o in ones) / SR_ASR
    print(f"  {durada / 60:.1f} min d'àudio | {len({e['client_id'] for e in entrades})} locutors")
    refs = [e["text"] for e in entrades]

    resultats = {}
    for idioma in idiomes:
        print(f"\nTranscrivint amb language=\"{idioma}\"...")
        hips = transcriure_lot(asr, ones, idioma, mida_lot)
        resultats[idioma] = {"metriques": metriques(refs, hips), "hipotesis": hips}
    return resultats


def informe_dataset(entrades: list[dict], resultats: dict, mostres: int) -> None:
    """Imprimeix els resultats per clip i les mostres pitjors."""
    print(f"\n{'=' * 78}\n  RESULTATS ({len(entrades)} clips)\n{'=' * 78}")
    ordenats = sorted(resultats.items(), key=lambda kv: kv[1]["metriques"]["WER"])
    for idioma, r in ordenats:
        m = r["metriques"]
        print(f"  language=\"{idioma:<3}\"   WER {m['WER']:.3f}   CER {m['CER']:.3f}")
    print(f"\n  -> punt de partida: language=\"{ordenats[0][0]}\"")

    if mostres:
        print(f"\n{'-' * 78}\n  Exemples\n{'-' * 78}")
        for i in range(min(mostres, len(entrades))):
            print(f"\n  REF   {entrades[i]['text']}")
            for idioma, r in ordenats:
                print(f"  {idioma:<4}  {r['hipotesis'][i]}")


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=Path, default=AUDIO_DEFECTE)
    p.add_argument("--dataset", type=Path, nargs="?", const=DATASET_DEFECTE,
                   help="manifest.jsonl a avaluar; sense valor, el de 00_baixar_test_set.py. "
                        "Avalua tot el conjunt "
                        "i dona WER i CER de corpus en comptes d'una sola transcripció")
    p.add_argument("--limit", type=int, help="màxim de clips del dataset a avaluar")
    p.add_argument("--model", default=MODEL_DEFECTE)
    p.add_argument("--idiomes", nargs="+", default=["oc", "ca"],
                   help="codis d'idioma a forçar, un per transcripció (defecte: oc ca)")
    p.add_argument("--referencia", type=Path,
                   help="fitxer de text amb la transcripció real d'--audio; calcula WER i CER")
    p.add_argument("--dispositiu", default="auto", help="'auto', 'cuda:1', 'cpu'...")
    p.add_argument("--chunk", type=float, default=None,
                   help="força el mode per finestres solapades de N segons en comptes de la "
                        "long-form seqüencial de Whisper. Més ràpid, però menys fiable")
    p.add_argument("--mida-lot", type=int, default=8,
                   help="clips per lot d'ASR amb --dataset (defecte: 8). Si peta per OOM, "
                        "baixa'l: Whisper omple cada àudio a 30 s encara que duri 5")
    p.add_argument("--mostres", type=int, default=5,
                   help="parells referència/hipòtesi a imprimir amb --dataset (defecte: 5)")
    p.add_argument("--sortida", type=Path, help="desa els resultats en un .json")
    args = p.parse_args()

    objectiu = args.dataset or args.audio
    if not objectiu.exists():
        quin = "00_baixar_test_set.py" if args.dataset else "02_baixar_referencia.py"
        print(f"No hi ha {objectiu}. Executa abans {quin}.", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore", message="(?s).*1080.*")
    import torch
    from transformers import logging as tlog
    from transformers import pipeline
    tlog.set_verbosity_error()

    dispositiu = triar_dispositiu(args.dispositiu)
    print(f"Carregant {args.model}...")
    asr = pipeline("automatic-speech-recognition", model=args.model,
                   dtype=torch.float16 if dispositiu.startswith("cuda") else torch.float32,
                   device=dispositiu)

    if args.dataset:
        entrades = carregar_manifest(args.dataset, args.limit)
        resultats = avaluar_dataset(asr, entrades, args.idiomes, args.mida_lot)
        informe_dataset(entrades, resultats, args.mostres)
        desar = {i: r["metriques"] | {"hipotesis": r["hipotesis"]} for i, r in resultats.items()}
    else:
        ona = carregar_audio(args.audio)
        referencia = args.referencia.read_text(encoding="utf-8").strip() if args.referencia else None
        desar = {}
        for idioma in args.idiomes:
            print(f"\n{'=' * 78}\n  language=\"{idioma}\"\n{'=' * 78}")
            text = transcriure(asr, ona, idioma, args.chunk)
            desar[idioma] = {"text": text}
            print(text or "(res)")
            if referencia:
                m = metriques(referencia, text)
                desar[idioma].update(m)
                print("\n  " + "  ".join(f"{k}: {v:.3f}" for k, v in m.items()))

        if referencia and len(desar) > 1:
            print(f"\n{'=' * 78}")
            for idioma, r in sorted(desar.items(), key=lambda kv: kv[1]["WER"]):
                print(f"  {idioma:>4}  WER {r['WER']:.3f}  CER {r['CER']:.3f}")
            print(f"  -> punt de partida: language=\"{min(desar, key=lambda k: desar[k]['WER'])}\"")
        elif not referencia:
            print(f"\n{'=' * 78}\nAixò només es compara a ull, i amb un sol tall el WER depèn "
                  "més del tall que del model.\nPer a un número de debò: "
                  "python3 00_baixar_test_set.py && python3 03_provar_whisper.py \\\n"
                  "    --dataset")

    if args.sortida:
        import json
        args.sortida.write_text(json.dumps(desar, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nDesat a {args.sortida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

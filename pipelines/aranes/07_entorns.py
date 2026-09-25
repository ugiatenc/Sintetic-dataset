#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 7: bancs de soroll (DEMAND) i de RIRs de les sales d'Aran, i una demo per escoltar-los."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import numpy as np

import acoustic_sim as ac
import build_noise_bank as bnb
import config as C
import entorns as E

DIR_DEMO = C.DIR_AUDIO / "entorns_demo"


def rt60_real(ruta: Path) -> float:
    """RT60 mesurat d'una RIR."""
    import pyroomacoustics as pra
    import soundfile as sf
    rir, sr = sf.read(ruta, dtype="float32")
    try:
        return float(pra.experimental.measure_rt60(rir, fs=sr, decay_db=30))
    except Exception:
        return float("nan")


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sense-demo", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    print("=== 1. Sorolls DEMAND ===")
    for nom, spec in E.AMBIENTS_ARANES.items():
        bnb.construir(nom, spec, ac.DIR_SOROLL, ac.SAMPLE_RATE, args.overwrite)
    bnb.escriure_atribucio(ac.DIR_SOROLL, {**bnb.AMBIENTS, **E.AMBIENTS_ARANES})

    print("\n=== 2. RIRs de les sales d'Aran ===")
    banc = ac.construir_banc_rir(E.SALES_ARANES, sr=ac.SAMPLE_RATE, overwrite=args.overwrite,
                                 verbose=False)
    mesures = {}
    for nom, rutes in banc.items():
        vals = [rt60_real(r) for r in rutes]
        vals = [v for v in vals if np.isfinite(v)]
        m = float(np.median(vals)) if vals else float("nan")
        mesures[nom] = round(m, 2)
        E.SALES_ARANES[nom].rt60_real = mesures[nom]
        s = E.SALES_ARANES[nom]
        print(f"  {nom:<13} {len(rutes)} RIR  entrada rt60={s.rt60:.2f}s  "
              f"MESURAT={m:.2f}s  ({s.dimensions[0]:.0f}x{s.dimensions[1]:.0f}x{s.dimensions[2]:.1f} m, "
              f"micro a {s.distancia_micro} m)")

    print(f"\n=== 3. Entorns actius: {len(E.ENTORNS_ACTIUS)} ===")
    total = sum(e.pes for e in E.ENTORNS_ACTIUS.values())
    for k, e in sorted(E.ENTORNS_ACTIUS.items(), key=lambda kv: -kv[1].pes):
        print(f"  {k:<16} pes {e.pes:.1f} ({100 * e.pes / total:>4.1f}%)  sala={e.sala or '-':<12} "
              f"soroll={e.ambient or '-':<15} snr={e.snr_db or '-'}  canal={e.canal}")

    if not args.sense_demo:
        print("\n=== 4. Demo ===")
        import soundfile as sf
        veus = sorted((C.DIR_AUDIO / "veus").glob("veu_*.wav"))
        if not veus:
            veus = sorted((ARREL / "lab/proves_inicials/aranes/audios/validacio_tts/ca").glob("*.wav"))[:1]
        if not veus:
            print("  cap veu disponible per a la demo")
        else:
            x, sr = sf.read(veus[0], dtype="float32")
            DIR_DEMO.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(0)
            metas = {}
            sf.write(DIR_DEMO / "00_sec.wav", x, sr, subtype="PCM_16")
            for k, e in E.ENTORNS_ACTIUS.items():
                y, meta = ac.mesclar_amb_entorn(x, sr, e, rng)
                sf.write(DIR_DEMO / f"{k}.wav", y, sr, subtype="PCM_16")
                metas[k] = {kk: vv for kk, vv in meta.items() if kk != "rir_path"} | {"rir_path": meta["rir_path"]}
            (DIR_DEMO / "meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {len(metas)} entorns sobre {veus[0].name} -> {DIR_DEMO}")

    (ac.DIR_SOROLL.parent / "README_aranes.md").write_text(
        "# Entorns d'Aran\n\n" +
        "RT60 mesurat sobre les RIRs generades (Schroeder, -30 dB):\n\n| Sala | Dimensions | RT60 entrada | RT60 mesurat |\n|---|---|---|---|\n" +
        "\n".join(f"| `{n}` | {s.dimensions[0]:.0f}x{s.dimensions[1]:.0f}x{s.dimensions[2]:.1f} m | {s.rt60:.2f} s | {mesures.get(n, float('nan')):.2f} s |"
                  for n, s in E.SALES_ARANES.items()) +
        "\n\nSorolls nous (DEMAND, CC BY 4.0): " + ", ".join(f"`{k}` ({v['demand']})" for k, v in E.AMBIENTS_ARANES.items()) +
        "\n\nEls entorns i els seus pesos son a `pipelines/aranes/entorns.py`.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

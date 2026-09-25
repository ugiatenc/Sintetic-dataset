#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revisio objectiva d'una mostra de l'audio generat (pas 8), per VEU i per ENTORN."""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import shutil
import statistics as st
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(ARREL / "pipelines/aranes"))

import numpy as np  # noqa: E402

import build_voice_bank as bvb  # noqa: E402
import config as C  # noqa: E402

V6 = importlib.import_module("06_veus")

MANIFEST = C.DIR_DATASET / "manifest.jsonl"
PLA = C.DIR_DATASET / "pla.jsonl"
VEUS = C.DIR_VEUS / "manifest.jsonl"


def llegeix(p):
    """Llegeix un jsonl."""
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def mostreja(ok, per_veu, min_entorn, rng):
    """Fins a `per_veu` clips per veu, repartits entre entorns."""
    per_v = collections.defaultdict(list)
    for r in ok:
        per_v[r["veu_id"]].append(r)
    triats, ids = [], set()
    for veu, clips in per_v.items():
        per_e = collections.defaultdict(list)
        for r in clips:
            per_e[r["entorn_id"]].append(r)
        entorns = list(per_e)
        rng.shuffle(entorns)
        k = 0
        while len([t for t in triats if t["veu_id"] == veu]) < per_veu and any(per_e.values()):
            e = entorns[k % len(entorns)]
            k += 1
            if per_e[e]:
                r = per_e[e].pop(rng.integers(len(per_e[e])))
                if r["id"] not in ids:
                    triats.append(r); ids.add(r["id"])
    per_e_tot = collections.defaultdict(list)
    for r in ok:
        if r["id"] not in ids:
            per_e_tot[r["entorn_id"]].append(r)
    n_e = collections.Counter(t["entorn_id"] for t in triats)
    for e, clips in per_e_tot.items():
        falten = min_entorn - n_e[e]
        if falten > 0 and clips:
            for i in rng.permutation(len(clips))[:falten]:
                triats.append(clips[i]); ids.add(clips[i]["id"])
    return triats


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-veu", type=int, default=12)
    p.add_argument("--min-entorn", type=int, default=30)
    p.add_argument("--llavor", type=int, default=7)
    args = p.parse_args()

    import soundfile as sf
    import torch
    from transformers import pipeline

    ok = [r for r in llegeix(MANIFEST) if r.get("status") == "ok"]
    est = {r["id"]: r["durada_est_s"] for r in llegeix(PLA)}
    veus = {v["id"]: v for v in llegeix(VEUS)}
    rng = np.random.default_rng(args.llavor)
    mostra = mostreja(ok, args.per_veu, args.min_entorn, rng)
    print(f"{len(ok):,} clips ok al manifest; mostra de {len(mostra):,} "
          f"({len({m['veu_id'] for m in mostra})} veus, {len({m['entorn_id'] for m in mostra})} entorns)", flush=True)

    emb = V6.Embedder()
    ref_emb, ref_ids = [], []
    for vid, v in veus.items():
        ona, sr = sf.read(v["ref_audio"], dtype="float32")
        ref_emb.append(emb(bvb._a_mono_16k(ona, sr))); ref_ids.append(vid)
    ref_emb = np.stack(ref_emb)

    vad = bvb.DetectorVeu()
    senyals = []
    for i, r in enumerate(mostra, 1):
        ona, sr = sf.read(ARREL / r["audio"], dtype="float32")
        s = bvb._a_mono_16k(ona, sr)
        segs = vad.segments(s)
        met = bvb.mesurar_qualitat(s, segs) if segs else {"ratio_parla": 0.0, "snr_db": -999.0, "pic_dbfs": bvb._dbfs(s), "ratio_clipping": 0.0, "transitoris_per_min": 999.0}
        e = emb(s)
        sims = ref_emb @ e
        j = int(np.argmax(sims))
        r.update({
            "durada_est_s": est.get(r["id"]), "ratio_durada": round(r["durada_s"] / est[r["id"]], 3) if est.get(r["id"]) else None,
            "ratio_parla": met["ratio_parla"], "snr_mesurat_db": met["snr_db"], "pic_dbfs": round(float(met["pic_dbfs"]), 1),
            "ratio_clipping": met["ratio_clipping"], "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9)), 1),
            "sim_veu": round(float(sims[ref_ids.index(r["veu_id"])]), 3),
            "veu_identificada": ref_ids[j], "id_correcta": ref_ids[j] == r["veu_id"],
        })
        senyals.append(s)
        if i % 200 == 0:
            print(f"  senyal {i}/{len(mostra)}", flush=True)

    gpu = torch.cuda.is_available()
    asr = pipeline("automatic-speech-recognition", model=V6.MODEL_ASR_PROVA, device=0 if gpu else -1,
                   torch_dtype=torch.float16 if gpu else torch.float32)
    for i in range(0, len(mostra), 8):
        lot = mostra[i:i + 8]
        outs = asr([{"raw": s, "sampling_rate": 16000} for s in senyals[i:i + 8]],
                   generate_kwargs={"language": "ca", "task": "transcribe"}, batch_size=8)
        for r, o in zip(lot, outs):
            hyp = o["text"].strip()
            ref = V6._norm_asr(r["tts_text"])
            r["whisper"] = hyp
            r["cer"] = round(V6._cer(ref, V6._norm_asr(hyp)), 3)
            nw = max(1, len(ref.split()))
            r["ratio_mots"] = round(len(V6._norm_asr(hyp).split()) / nw, 2)
        if (i // 8) % 25 == 0:
            print(f"  whisper {min(i + 8, len(mostra))}/{len(mostra)}", flush=True)
    del asr
    if gpu:
        torch.cuda.empty_cache()

    with (AQUI / "mostra.jsonl").open("w", encoding="utf-8") as fh:
        for r in mostra:
            fh.write(json.dumps({k: v for k, v in r.items() if k not in ("ref_text", "rir_path")}, ensure_ascii=False) + "\n")

    def med(xs):
        """Mediana ignorant None."""
        xs = [x for x in xs if x is not None]
        return st.median(xs) if xs else float("nan")

    def p90(xs):
        """Percentil 90 ignorant None."""
        xs = sorted(x for x in xs if x is not None)
        return xs[int(0.9 * (len(xs) - 1))] if xs else float("nan")

    cer_glob = med([r["cer"] for r in mostra])
    files_v = []
    for vid in sorted(veus):
        g = [r for r in mostra if r["veu_id"] == vid]
        if not g:
            continue
        net = [r for r in g if r["entorn_id"] == "estudi_radio"]
        files_v.append({
            "veu": vid, "n": len(g), "cer_med": med([r["cer"] for r in g]), "cer_p90": p90([r["cer"] for r in g]),
            "sim": med([r["sim_veu"] for r in g]), "sim_net": med([r["sim_veu"] for r in net]),
            "id_ok": sum(r["id_correcta"] for r in g) / len(g), "ratio_dur": med([r["ratio_durada"] for r in g]),
            "parla": med([r["ratio_parla"] for r in g]), "pic": max(r["pic_dbfs"] for r in g),
            "galimaties": sum(r["cer"] > 0.5 for r in g),
        })
    files_e = []
    for eid in sorted({r["entorn_id"] for r in mostra}):
        g = [r for r in mostra if r["entorn_id"] == eid]
        files_e.append({
            "entorn": eid, "n": len(g), "cer_med": med([r["cer"] for r in g]), "cer_p90": p90([r["cer"] for r in g]),
            "sim": med([r["sim_veu"] for r in g]), "id_ok": sum(r["id_correcta"] for r in g) / len(g),
            "snr_manifest": med([r["snr_db"] for r in g if r.get("snr_db") is not None]),
            "snr_mesurat": med([r["snr_mesurat_db"] for r in g]), "rms": med([r["rms_dbfs"] for r in g]),
            "pic": max(r["pic_dbfs"] for r in g), "clipping": max(r["ratio_clipping"] for r in g),
            "parla": med([r["ratio_parla"] for r in g]), "galimaties": sum(r["cer"] > 0.5 for r in g),
        })

    for carpeta in ("per_veu", "per_entorn"):
        shutil.rmtree(AQUI / carpeta, ignore_errors=True)

    def desa(carpeta, clau, valor, g):
        """Copia el millor i el pitjor clip d'un grup al kit d'escolta, amb el seu .txt."""
        d = AQUI / carpeta / valor
        d.mkdir(parents=True, exist_ok=True)
        g = sorted(g, key=lambda r: r["cer"])
        for etiqueta, r in (("millor", g[0]), ("pitjor", g[-1])):
            shutil.copyfile(ARREL / r["audio"], d / f"{etiqueta}_{r['id']}.wav")
            (d / f"{etiqueta}_{r['id']}.txt").write_text(
                f"veu: {r['veu_id']}   entorn: {r['entorn_id']}   font: {r['font']}   speed: {r['speed']}\n"
                f"CER: {r['cer']}   ratio mots: {r['ratio_mots']}   sim veu: {r['sim_veu']} (identificada: {r['veu_identificada']})\n"
                f"durada: {r['durada_s']} s (estimada {r['durada_est_s']} s)   parla: {r['ratio_parla']}   "
                f"SNR manifest/mesurat: {r.get('snr_db')}/{r['snr_mesurat_db']} dB   pic: {r['pic_dbfs']} dBFS\n\n"
                f"TEXT REAL : {r['text']}\nTTS_TEXT  : {r['tts_text']}\nWHISPER   : {r['whisper']}\n", encoding="utf-8")

    for f in files_v:
        desa("per_veu", "veu_id", f["veu"], [r for r in mostra if r["veu_id"] == f["veu"]])
    for f in files_e:
        desa("per_entorn", "entorn_id", f["entorn"], [r for r in mostra if r["entorn_id"] == f["entorn"]])

    def marca_v(f):
        """Marques d'alerta d'una veu."""
        m = []
        if f["cer_med"] > 2 * cer_glob: m.append("CER alt")
        if f["galimaties"]: m.append(f"{f['galimaties']} galimaties")
        if f["sim_net"] == f["sim_net"] and f["sim_net"] < 0.5: m.append("clon poc fidel")
        if f["id_ok"] < 0.5: m.append("no s'identifica")
        if not 0.6 <= f["ratio_dur"] <= 1.5: m.append("durada rara")
        if f["pic"] > -0.5: m.append("satura")
        return ", ".join(m)

    def marca_e(f):
        """Marques d'alerta d'un entorn."""
        m = []
        if f["cer_med"] > 2 * cer_glob: m.append("CER alt")
        if f["galimaties"]: m.append(f"{f['galimaties']} galimaties")
        if f["clipping"] > 0.001: m.append("clipping")
        if f["pic"] > -0.5: m.append("satura")
        return ", ".join(m)

    L = ["# Revisió de l'àudio generat (pas 8), per veu i per entorn", "",
         f"Mostra de **{len(mostra):,} clips** de {len(ok):,} generats fins ara (fins a {args.per_veu} per veu, "
         f"≥ {args.min_entorn} per entorn). CER = Whisper large-v3-turbo (ca) contra el `tts_text`; el valor absolut "
         f"té un sòl (Whisper catalanitza), el que compta és la comparació. Mediana global del CER: **{cer_glob:.3f}**. "
         f"«sim» = cosinus ECAPA amb la referència de la veu (sim_net: només clips de l'estudi de ràdio, sense soroll); "
         f"«id» = fracció de clips en què la referència més propera de les {len(veus)} és la seva.", "",
         "Kit d'escolta: `per_veu/<veu>/` i `per_entorn/<entorn>/`, amb el millor i el pitjor clip (per CER) i un `.txt` amb "
         "text, transcripció i mètriques. Totes les mètriques per clip: `mostra.jsonl`.", "",
         "## Per entorn", "",
         "| Entorn | n | CER med | CER p90 | sim | id | SNR manifest | SNR mesurat | RMS dBFS | pic dBFS | parla | Marques |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in files_e:
        snr_m = "—" if f["snr_manifest"] != f["snr_manifest"] else f"{f['snr_manifest']:.0f}"
        L.append(f"| `{f['entorn']}` | {f['n']} | {f['cer_med']:.3f} | {f['cer_p90']:.3f} | {f['sim']:.2f} | {f['id_ok']:.0%} | "
                 f"{snr_m} | {f['snr_mesurat']:.0f} | {f['rms']:.0f} | {f['pic']:.1f} | {f['parla']:.2f} | {marca_e(f)} |")
    L += ["", "## Per veu", "",
          "| Veu | n | CER med | CER p90 | sim | sim_net | id | durada real/est. | parla | pic dBFS | Marques |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(files_v, key=lambda f: -f["cer_med"]):
        sn = "—" if f["sim_net"] != f["sim_net"] else f"{f['sim_net']:.2f}"
        L.append(f"| `{f['veu']}` | {f['n']} | {f['cer_med']:.3f} | {f['cer_p90']:.3f} | {f['sim']:.2f} | {sn} | {f['id_ok']:.0%} | "
                 f"{f['ratio_dur']:.2f} | {f['parla']:.2f} | {f['pic']:.1f} | {marca_v(f)} |")
    dolents = sorted(mostra, key=lambda r: -r["cer"])[:15]
    L += ["", "## Els 15 clips amb més CER de la mostra", "", "| CER | ratio mots | veu | entorn | tts_text | Whisper |", "|---|---|---|---|---|---|"]
    for r in dolents:
        L.append(f"| {r['cer']:.2f} | {r['ratio_mots']} | `{r['veu_id'][4:]}` | `{r['entorn_id']}` | {r['tts_text'][:70]} | {r['whisper'][:70]} |")
    (AQUI / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nREADME i kit a {AQUI}")
    print("entorns:", [(f["entorn"], round(f["cer_med"], 3), marca_e(f)) for f in files_e])
    print("veus marcades:", [(f["veu"], round(f["cer_med"], 3), marca_v(f)) for f in files_v if marca_v(f)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

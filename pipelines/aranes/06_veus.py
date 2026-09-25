#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 6: banc de veus de referencia de Common Voice per al clonatge amb OmniVoice, amb prova de sintesi."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import numpy as np

import build_voice_bank as bvb
import config as C

REPO = "fsicoli/common_voice_22_0"
DIR_CV = C.DIR_CV_CRU
DIR_VEUS = C.DIR_VEUS
DIR_CANDIDATS = DIR_VEUS / "candidats"
MANIFEST_TEST = C.MANIFEST_CV
SPLITS = {"oc": ["dev", "train", "other"], "ca": ["dev"]}

DUR_MIN, DUR_IDEAL_MIN, DUR_IDEAL_MAX, DUR_MAX = 2.5, 3.0, 5.0, 8.0
SILENCI_VORA_S = 0.15
MIN_SNR_DB, MIN_RATIO_PARLA = 20.0, 0.70  # per acceptar un clip com a mostra del locutor
MIN_SIM_LOCUTOR = 0.55  # cosinus ECAPA amb el centroide del locutor
NOTA_MINIMA, MARGE_CLAR = 0.70, 0.05  # per sota, el locutor va a candidats/
SR_SORTIDA = 24000
MIN_SNR_REFERENCIA = 30.0  # per ser LA referencia: el TTS clona tambe el soroll de fons
MIN_SNR_CA = 45.0  # catalanes: es trien entre milers, es pot exigir mes

FRASE_PROVA = ("Eth Conselh Generau d'Aran a aprovat eth pressupòst entà er an que ven, "  # sense numerals: Whisper els escriu en xifres o en lletres i falsejava el CER
               "damb ua pujada des ajudes entàs pòbles dera Val.")
N_LECTURES = 3  # lectures per veu a la prova de sintesi; compta la mediana (OmniVoice no es determinista)
MAX_CER_SINTESI = 0.25  # mediana del CER contra el consens per sobre de la qual la veu cau
MODEL_ASR_PROVA = "openai/whisper-large-v3-turbo"
ACCENTS_CA = ("central", "nord-occidental")
MARGE_CA = 3  # catalanes de reserva per genere, per si la prova de sintesi en treu alguna


def _shards(locale: str, split: str) -> list[str]:
    """Fitxers tar d'un split de Common Voice al repositori."""
    from huggingface_hub import HfApi
    fitxers = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in fitxers
                  if re.match(rf"audio/{locale}/{split}/{locale}_{split}_\d+\.tar$", f))


def _tsv(locale: str, nom: str) -> list[dict]:
    """Baixa i llegeix un tsv de Common Voice."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"transcript/{locale}/{nom}.tsv", repo_type="dataset")
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


def baixar(locale: str, splits: list[str]) -> Path:
    """Tots els clips dels splits demanats -> wav 16 kHz mono + manifest.jsonl."""
    import librosa
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    desti = DIR_CV / locale
    (desti / "clips").mkdir(parents=True, exist_ok=True)
    durades = {f["clip"]: int(f["duration[ms]"]) for f in _tsv(locale, "clip_durations")}
    manifest = []
    for split in splits:
        files = {f["path"]: f for f in _tsv(locale, split)}
        shards = _shards(locale, split)
        print(f"[{locale}/{split}] {len(files):,} clips en {len(shards)} shard(s)")
        for shard in shards:
            ruta = hf_hub_download(REPO, shard, repo_type="dataset")
            n = 0
            with tarfile.open(ruta) as tar:
                for membre in tar.getmembers():
                    nom = Path(membre.name).name
                    fila = files.get(nom)
                    if not fila:
                        continue
                    wav = desti / "clips" / (Path(nom).stem + ".wav")
                    if not wav.exists():
                        ona, _ = librosa.load(tar.extractfile(membre), sr=bvb.FS_ANALISI, mono=True)
                        sf.write(wav, ona, bvb.FS_ANALISI, subtype="PCM_16")
                    manifest.append({
                        "audio": str(wav), "text": fila["sentence"].strip(),
                        "client_id": fila["client_id"], "split": split, "locale": locale,
                        "accents": fila.get("accents") or "", "gender": fila.get("gender") or "",
                        "age": fila.get("age") or "",
                        "durada_s": round(durades.get(nom, 0) / 1000, 2),
                    })
                    n += 1
            print(f"   {Path(shard).name}: {n:,} clips")
    ruta = desti / "manifest.jsonl"
    with ruta.open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"   -> {len(manifest):,} clips, {len({m['client_id'] for m in manifest})} locutors, "
          f"{sum(m['durada_s'] for m in manifest) / 3600:.1f} h  ({ruta})")
    return ruta


class Embedder:
    """ECAPA-TDNN a la GPU si n'hi ha."""

    def __init__(self):
        """Carrega l'ECAPA-TDNN de speechbrain (GPU si n'hi ha)."""
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": self.device})

    def __call__(self, senyal16k: np.ndarray) -> np.ndarray:
        """Embedding normalitzat d'un senyal a 16 kHz."""
        import torch
        with torch.no_grad():
            v = self.model.encode_batch(torch.from_numpy(senyal16k).unsqueeze(0).to(self.device))
        v = v.squeeze().cpu().numpy()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def constancia(senyal16k: np.ndarray, segments) -> float:
    """0..1: com d'estable es el nivell de la parla."""
    fin = int(0.05 * bvb.FS_ANALISI)
    masc = bvb._mascara_parla(senyal16k.size, segments)
    n = senyal16k.size // fin
    if n < 4:
        return 0.0
    talls = senyal16k[:n * fin].reshape(-1, fin)
    es_parla = masc[:n * fin].reshape(-1, fin).mean(axis=1) > 0.5
    rms = np.sqrt((talls ** 2).mean(axis=1))[es_parla]
    if rms.size < 4 or rms.mean() <= 0:
        return 0.0
    cv = float(rms.std() / rms.mean())
    return float(np.clip(1.0 - cv / 1.2, 0.0, 1.0))


def p_durada(d: float) -> float:
    """1 entre 3 i 5 s; baixa linealment fins a 0 als extrems permesos."""
    if DUR_IDEAL_MIN <= d <= DUR_IDEAL_MAX:
        return 1.0
    if d < DUR_IDEAL_MIN:
        return max(0.0, (d - DUR_MIN) / (DUR_IDEAL_MIN - DUR_MIN))
    return max(0.0, (DUR_MAX - d) / (DUR_MAX - DUR_IDEAL_MAX))


def nota(met: dict, dur: float, const: float) -> float:
    """Puntuacio 0..1 d'un clip: SNR, parla, transitoris, durada i constancia."""
    p_snr = float(np.clip((met["snr_db"] - MIN_SNR_DB) / (bvb.SNR_BO_DB - MIN_SNR_DB), 0, 1))
    p_parla = float(np.clip((met["ratio_parla"] - MIN_RATIO_PARLA) / (0.98 - MIN_RATIO_PARLA), 0, 1))
    p_net = 1.0 - float(np.clip(met["transitoris_per_min"] / bvb.MAX_TRANSITORIS_PER_MIN, 0, 1))
    return round(0.35 * p_snr + 0.20 * p_parla + 0.15 * p_net + 0.15 * p_durada(dur) + 0.15 * const, 3)


def avaluar_clip(ruta: str, vad, emb) -> tuple[np.ndarray | None, dict]:
    """(senyal net a 16 kHz, metadades) o (None, {"motiu": ...})."""
    import soundfile as sf
    ona, sr = sf.read(ruta, dtype="float32")
    s = bvb._a_mono_16k(ona, sr)
    segs = vad.segments(s)
    if not segs:
        return None, {"motiu": "sense parla"}
    s, segs = bvb._retallar_vores(s, segs, SILENCI_VORA_S, SILENCI_VORA_S)
    dur = s.size / bvb.FS_ANALISI
    if dur < DUR_MIN:
        return None, {"motiu": f"massa curt ({dur:.1f}s)"}
    if dur > DUR_MAX:
        return None, {"motiu": f"massa llarg ({dur:.1f}s)"}
    met = bvb.mesurar_qualitat(s, segs)
    motiu = bvb.clip_acceptable(met, MIN_SNR_DB, MIN_RATIO_PARLA)
    if motiu:
        return None, {"motiu": motiu}
    const = constancia(s, segs)
    return s, {"durada_s": round(dur, 2), "snr_db": round(met["snr_db"], 1),
               "ratio_parla": round(met["ratio_parla"], 3),
               "transitoris_per_min": round(met["transitoris_per_min"], 1),
               "constancia": round(const, 3), "nota": nota(met, dur, const),
               "embedding": emb(s) if emb else None}


def _norm_asr(t: str) -> str:
    """Normalitza una transcripcio (minuscules, sense accents ni xifres)."""
    import unicodedata
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t)).strip()


def _cer(a: str, b: str) -> float:
    """Taxa d'error de caracters (distancia d'edicio / llargada de la referencia)."""
    if not a:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / len(a)


def prova_sintesi(banc: list[dict], desa: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Cada veu llegeix FRASE_PROVA N_LECTURES vegades damb OmniVoice, Whisper transcriu cada lectura."""
    import librosa
    import respelling
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice
    from transformers import pipeline
    text = respelling.reescriu(FRASE_PROVA)
    if desa is not None:
        desa.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0" if gpu else "cpu",
                                    dtype=torch.float16 if gpu else torch.float32)
    asr = pipeline("automatic-speech-recognition", model=MODEL_ASR_PROVA, device=0 if gpu else -1,
                   torch_dtype=torch.float16 if gpu else torch.float32)
    lectures: dict[str, list[str]] = {}
    for i, b in enumerate(banc, 1):
        lectures[b["id"]] = []
        rutes = []
        for k in range(N_LECTURES):
            torch.manual_seed(7 + k)
            ona = np.asarray(tts.generate(text=[text], language=C.IDIOMA_TTS, ref_audio=[b["ref_audio"]],
                                          ref_text=[b["ref_text"]])[0], dtype="float32")
            if desa is not None:
                ruta = desa / f"{b['id']}_{k + 1}.wav"
                sf.write(ruta, ona, SR_SORTIDA)
                rutes.append(str(ruta))
            x16 = librosa.resample(ona, orig_sr=SR_SORTIDA, target_sr=16000)
            hyp = asr({"raw": x16, "sampling_rate": 16000},
                      generate_kwargs={"language": "ca", "task": "transcribe"})["text"]
            lectures[b["id"]].append(hyp.strip())
        if rutes:
            b["audio_sintesi"] = rutes
        if i % 10 == 0:
            print(f"  prova de sintesi {i}/{len(banc)}")
    del tts, asr
    if gpu:
        torch.cuda.empty_cache()
    totes = [_norm_asr(h) for hs in lectures.values() for h in hs]
    if len(totes) >= 3:
        recompte: dict[str, int] = {}
        for t in totes:
            recompte[t] = recompte.get(t, 0) + 1
        uniques = list(recompte)
        mitjanes = [sum(recompte[u] * _cer(t, u) for u in uniques) / len(totes) for t in uniques]
        consens = uniques[int(np.argmin(mitjanes))]
    else:
        consens = _norm_asr(text)
    for b in banc:
        cers = [round(_cer(consens, _norm_asr(h)), 3) for h in lectures[b["id"]]]
        ordre = sorted(range(len(cers)), key=lambda k: cers[k])
        k_med = ordre[len(ordre) // 2]
        b["cer_lectures"] = cers
        b["cer_sintesi"] = cers[k_med]
        b["whisper_sintesi"] = lectures[b["id"]][k_med]
    print(f"  consens: «{consens}»")
    return ([b for b in banc if b["cer_sintesi"] <= MAX_CER_SINTESI],
            [b for b in banc if b["cer_sintesi"] > MAX_CER_SINTESI])


def candidates_catalanes(n: int, marge: int = MARGE_CA) -> list[dict]:
    """N veus de la reserva catalana (`reserva_catalanes/manifest.jsonl`."""
    ruta = C.DIR_RESERVA_CA / "manifest.jsonl"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}: `06_veus.py --locales ca --sortida {C.DIR_RESERVA_CA}`")
    ca = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    ca = [v for v in ca if (v.get("accents") or "").strip().lower() in ACCENTS_CA and Path(v["ref_audio"]).exists()
          and (v.get("snr_db") or 0) >= MIN_SNR_CA and (v.get("transitoris_per_min") or 0) == 0
          and (v.get("ratio_parla") or 0) >= 0.85]
    ca.sort(key=lambda v: -(v.get("nota") or 0))
    dones = [v for v in ca if "fem" in (v.get("gender") or "")][:n // 2 + marge]
    homes = [v for v in ca if "masc" in (v.get("gender") or "")][:n - n // 2 + marge]
    return dones + homes


def retalla_catalanes(banc: list[dict], n: int) -> list[dict]:
    """Despres de la prova de sintesi deixa nomes N veus catalanes (meitat dones, meitat homes."""
    quota = {"f": n // 2, "m": n - n // 2}
    triades = set()
    for b in sorted((b for b in banc if b.get("locale") == "ca"), key=lambda v: -(v.get("nota") or 0)):
        g = "f" if "fem" in (b.get("gender") or "") else "m"
        if quota[g] > 0:
            quota[g] -= 1
            triades.add(b["id"])
    sobrants = [b for b in banc if b.get("locale") == "ca" and b["id"] not in triades]
    for b in sobrants:
        Path(b["ref_audio"]).unlink(missing_ok=True)
    if any(quota.values()):
        print(f"  AVIS: no arriben a {n} catalanes; en falten {quota} (dones/homes)")
    print(f"  catalanes: {len(triades)} triades, {len(sobrants)} de reserva esborrades")
    return [b for b in banc if b.get("locale") != "ca" or b["id"] in triades]


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    global DIR_VEUS, DIR_CANDIDATS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baixar", action="store_true")
    p.add_argument("--sense-prova-sintesi", action="store_true",
                   help="no sintetitza cap frase de prova (vegeu FRASE_PROVA)")
    p.add_argument("--afegir-catalanes", type=int, default=0, metavar="N",
                   help="hi afegeix N veus de la reserva catalana (reserva_catalanes/), per diversitat de timbre")
    p.add_argument("--locales", nargs="+", default=["oc"], choices=["oc", "ca"])
    p.add_argument("--max-locutors", type=int, default=None)
    p.add_argument("--nota-minima", type=float, default=NOTA_MINIMA)
    p.add_argument("--sortida", type=Path, default=DIR_VEUS,
                   help="carpeta del banc (per defecte data/aranes/audio/veus). Per fer la reserva "
                        "catalana sense trepitjar el banc occita: --locales ca --sortida "
                        "data/aranes/audio/reserva_catalanes")
    args = p.parse_args()
    DIR_VEUS = args.sortida.resolve()
    DIR_CANDIDATS = DIR_VEUS / "candidats"

    if args.baixar:
        for loc in args.locales:
            baixar(loc, SPLITS[loc])

    exclosos = set()
    if MANIFEST_TEST.exists():
        exclosos |= {json.loads(l)["client_id"] for l in MANIFEST_TEST.read_text(encoding="utf-8").splitlines()
                     if l.strip()}
    clips = []
    for loc in args.locales:
        m = DIR_CV / loc / "manifest.jsonl"
        if not m.exists():
            print(f"  {loc}: no hi ha clips baixats (executa --baixar)")
            continue
        clips += [json.loads(l) for l in m.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_locutor = defaultdict(list)
    for c in clips:
        if c["client_id"] not in exclosos:
            per_locutor[c["client_id"]].append(c)
    print(f"\n{len(clips):,} clips | {len(exclosos)} locutors del test exclosos | "
          f"{len(per_locutor)} locutors candidats")
    if not per_locutor:
        return 1

    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
    vad = bvb.DetectorVeu()
    try:
        emb = Embedder()
        print(f"  embeddings ECAPA a {emb.device}")
    except Exception as e:
        print(f"  avis: sense embeddings ({type(e).__name__}); no es comprova la consistencia")
        emb = None

    DIR_VEUS.mkdir(parents=True, exist_ok=True)
    DIR_CANDIDATS.mkdir(parents=True, exist_ok=True)
    banc, dubtosos, descartats = [], [], {}
    locutors = sorted(per_locutor.items(), key=lambda kv: -len(kv[1]))
    if args.max_locutors:
        locutors = locutors[:args.max_locutors]

    for i, (cid, cl) in enumerate(locutors, 1):
        bons, motius = [], defaultdict(int)
        for c in cl:
            s, meta = avaluar_clip(c["audio"], vad, emb)
            if s is None:
                motius[meta["motiu"].split(" (")[0].split("=")[0]] += 1
                continue
            bons.append((s, meta, c))
        if emb and len(bons) >= 3:
            E = np.stack([b[1]["embedding"] for b in bons])
            centre = E.mean(axis=0); centre /= np.linalg.norm(centre)
            sims = E @ centre
            for b, sim in zip(bons, sims):
                b[1]["sim_locutor"] = round(float(sim), 3)
            fora = sum(1 for sim in sims if sim < MIN_SIM_LOCUTOR)
            if fora:
                motius["una altra veu o soroll"] += fora
            bons = [b for b, sim in zip(bons, sims) if sim >= MIN_SIM_LOCUTOR]
        if not bons:
            descartats[cid] = dict(motius)
            print(f"  [{i:>3}/{len(locutors)}] {cid[:10]}  {len(cl):>3} clips -> cap de valid  {dict(motius)}")
            continue
        bons.sort(key=lambda b: -b[1]["nota"])
        millor = bons[0][1]["nota"]
        segon = bons[1][1]["nota"] if len(bons) > 1 else 0.0
        nets = [b for b in bons if b[1]["snr_db"] >= MIN_SNR_REFERENCIA]
        clar = millor >= args.nota_minima and bool(nets)
        registre = {"client_id": cid, "locale": cl[0]["locale"], "accents": cl[0]["accents"],
                    "gender": cl[0]["gender"], "age": cl[0]["age"], "clips_totals": len(cl),
                    "clips_valids": len(bons), "rebutjats": dict(motius)}
        if clar:
            s, meta, c = nets[0]
            nom = f"veu_{cl[0]['locale']}_{cid[:12]}"
            info = bvb.escriure_wav(DIR_VEUS / f"{nom}.wav", s, SR_SORTIDA, "lufs",
                                    bvb.DEFAULT_LUFS, bvb.DEFAULT_PEAK_DBFS)
            banc.append({**registre, "id": nom, "ref_audio": str(DIR_VEUS / f"{nom}.wav"),
                         "ref_text": c["text"], "origen": c["audio"], "split": c["split"],
                         **{k: v for k, v in meta.items() if k != "embedding"},
                         "alternatives": [{"origen": b[2]["audio"], "nota": b[1]["nota"],
                                           "durada_s": b[1]["durada_s"], "text": b[2]["text"]}
                                          for b in bons[1:4]]})
            print(f"  [{i:>3}/{len(locutors)}] {cid[:10]}  {len(cl):>3} clips -> VEU  nota {millor:.2f}  "
                  f"{meta['durada_s']}s snr {meta['snr_db']}  «{c['text'][:50]}»")
        else:
            d = DIR_CANDIDATS / cid[:12]
            d.mkdir(parents=True, exist_ok=True)
            files = []
            for k, (s, meta, c) in enumerate(bons[:3], 1):
                bvb.escriure_wav(d / f"cand{k}.wav", s, SR_SORTIDA, "lufs",
                                 bvb.DEFAULT_LUFS, bvb.DEFAULT_PEAK_DBFS)
                files.append({"fitxer": f"cand{k}.wav", "text": c["text"], "split": c["split"],
                              **{k2: v for k2, v in meta.items() if k2 != "embedding"}})
            with (d / "candidats.csv").open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(files[0]) + ["triat", "comentari"])
                w.writeheader(); w.writerows(files)
            per_que = ("cap clip arriba a la nota minima" if millor < args.nota_minima else
                       f"cap clip net: el millor te SNR {bons[0][1]['snr_db']} dB < {MIN_SNR_REFERENCIA:.0f}")
            dubtosos.append({**registre, "carpeta": str(d), "millor_nota": millor, "per_que": per_que})
            print(f"  [{i:>3}/{len(locutors)}] {cid[:10]}  {len(cl):>3} clips -> candidats/  "
                  f"nota {millor:.2f} vs {segon:.2f}  ({per_que})")

    if args.afegir_catalanes:
        import shutil
        afegides = candidates_catalanes(args.afegir_catalanes)
        for v in afegides:
            desti = DIR_VEUS / f"{v['id']}.wav"
            shutil.copyfile(v["ref_audio"], desti)
            banc.append({**v, "ref_audio": str(desti)})
        print(f"\n{len(afegides)} veus catalanes candidates ({sum(1 for v in afegides if 'fem' in (v.get('gender') or ''))} dones), "
              f"reserva inclosa; en quedaran {args.afegir_catalanes} despres de la prova de sintesi")

    if banc and not args.sense_prova_sintesi:
        print(f"\nProva de sintesi: {len(banc)} veus llegeixen «{FRASE_PROVA[:45]}...» "
              f"({N_LECTURES} lectures cadascuna; compta la mediana)")
        banc, dolentes = prova_sintesi(banc, DIR_VEUS / "prova_sintesi")
        for b in dolentes:
            descartats[b["client_id"]] = {**b.get("rebutjats", {}),
                                          "prova de sintesi": f"CER {b['cer_sintesi']} contra el consens: "
                                                              f"«{b['whisper_sintesi'][:80]}»"}
            Path(b["ref_audio"]).unlink(missing_ok=True)
        print(f"  {len(dolentes)} veus fora per la prova de sintesi: {[b['id'] for b in dolentes]}")
    if args.afegir_catalanes:
        banc = retalla_catalanes(banc, args.afegir_catalanes)

    with (DIR_VEUS / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for b in banc:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    (DIR_CANDIDATS / "pendents.json").write_text(json.dumps(dubtosos, ensure_ascii=False, indent=2), encoding="utf-8")
    (DIR_VEUS / "descartats.json").write_text(json.dumps(descartats, ensure_ascii=False, indent=2), encoding="utf-8")
    aran = sum(1 for b in banc if "aran" in b["accents"].lower())
    (DIR_VEUS / "README.md").write_text(f"""# Banc de veus

**{len(banc)} veus** triades automaticament (de {len(locutors)} locutors candidats), {aran} damb accent
aranes declarat. **{len(dubtosos)} locutors a `candidats/`** perque algu triï a ma, i
{len(descartats)} sense cap clip valid (`descartats.json` diu per que).

Cada veu es UN clip sencer de Common Voice, damb el seu text exacte a `manifest.jsonl`
(`ref_text`). Criteris: 3-8 s (ideal 3-5), sense silenci a les vores, SNR >= {MIN_SNR_DB:.0f} dB per
comptar com a clip del locutor i **>= {MIN_SNR_REFERENCIA:.0f} dB per ser la referencia** (el TTS clona
tambe el soroll), parla >= {MIN_RATIO_PARLA:.0%} del clip, sense saturacio ni transitoris, i que soni
com la resta de clips del mateix locutor (embedding ECAPA, cosinus >= {MIN_SIM_LOCUTOR}). Les veus
catalanes, SNR >= {MIN_SNR_CA:.0f} dB. Nivell a {bvb.DEFAULT_LUFS} LUFS, {SR_SORTIDA} Hz mono.

Els locutors del test del projecte (`datasets/aranes/test/manifest.jsonl`, vinguin del split de
Common Voice que vinguin) queden fora sempre. I cada veu ha passat la PROVA DE SINTESI: ha llegit
«{FRASE_PROVA}» {N_LECTURES} vegades damb OmniVoice, Whisper ha transcrit cada lectura, i la MEDIANA del CER
contra el consens no passa de {MAX_CER_SINTESI} (`cer_sintesi`, `cer_lectures` i `whisper_sintesi` al manifest;
l'audio de cada lectura es a `prova_sintesi/<id>_<k>.wav`, tambe el de les veus que han caigut).
Veus catalanes (`veu_ca_*`), si n'hi ha: nomes per diversitat de timbre.

## candidats/

Un locutor hi va quan el seu millor clip no arriba a {NOTA_MINIMA} de nota. A cada carpeta hi ha `cand1..3.wav` i
`candidats.csv` damb les metriques, el text i dues columnes buides (`triat`, `comentari`).
Per incorporar-ne un: copia el wav a `veus/` i afegeix la linia al `manifest.jsonl`.
""", encoding="utf-8")
    print(f"\n{len(banc)} veus -> {DIR_VEUS}   |   {len(dubtosos)} a candidats/   |   {len(descartats)} descartats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

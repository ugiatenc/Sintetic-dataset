#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banc de veus de referencia (Common Voice, VoxPopuli): VAD, metriques de qualitat, seleccio i normalitzacio."""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import tarfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

REPO_COMMON_VOICE = "fsicoli/common_voice_17_0"
REPO_VOXPOPULI = "facebook/voxpopuli"

ACCENT_KEYWORDS_ES_ES = ["espana"]

DEFAULT_OUTPUT_DIR = Path("data/voice_seeds")
MANIFEST_NAME = "voice_bank_manifest.json"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_LUFS = -23.0
DEFAULT_PEAK_DBFS = -1.0

DURADA_MIN_S = 3.0
DURADA_MAX_S = 10.0
SILENCI_CONFORT_S = 0.2
SILENCI_VORA_MAX_S = 0.5
MARGE_JUNTURA_S = 0.14

FS_ANALISI = 16000
MIN_RATIO_PARLA = 0.60
MIN_SNR_DB = 18.0
MIN_SNR_VOXPOPULI_DB = 14.0
SNR_BO_DB = 35.0
SNR_MAX_DB = 60.0
MAX_RATIO_CLIPPING = 0.001
LLINDAR_CLIPPING = 0.98
MAX_TRANSITORIS_PER_MIN = 6.0
MIN_DURADA_CLIP_S = 1.0
MIN_PUNTUACIO = 0.5

GRUPS_EDAT = {
    "teens": "jove", "twenties": "jove",
    "thirties": "adult", "fourties": "adult", "fifties": "adult",
    "sixties": "gran", "seventies": "gran", "eighties": "gran", "nineties": "gran",
}
GENERES = {
    "male_masculine": "male", "female_feminine": "female",
    "male": "male", "female": "female",
}


def _normalitzar(text):
    """Treu accents/majuscules per comparar paraules clau sense sorpreses."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return text.lower()


def _sanejar(text, per_defecte="unknown"):
    """Deixa un fragment apte per a un nom de fitxer."""
    net = re.sub(r"[^A-Za-z0-9]+", "-", _normalitzar(text)).strip("-")
    return net or per_defecte


def _genere_canonic(valor):
    """Genere normalitzat."""
    return GENERES.get(_normalitzar(valor).strip(), "unknown")


def _grup_edat(valor):
    """Grup d'edat normalitzat."""
    return GRUPS_EDAT.get(_normalitzar(valor).strip(), "desconegut")


def _id_curt(speaker_id):
    """`client_id` de Common Voice fa 128 hex: massa per a un nom de fitxer."""
    net = _sanejar(speaker_id, "")
    if net and len(net) <= 12:
        return net
    return hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()[:10]


def _a_mono_16k(senyal, sr):
    """Mono float32 a FS_ANALISI, que es on treballen VAD i metriques."""
    if senyal.ndim > 1:
        senyal = senyal.mean(axis=1)
    senyal = senyal.astype(np.float32)
    if sr != FS_ANALISI:
        senyal = soxr.resample(senyal, sr, FS_ANALISI, quality="VHQ")
    return senyal.astype(np.float32)


def _dbfs(x):
    """calcula el nivell de pic en dBFS (0 dB = 1.0) d'un senyal float32."""
    pic = float(np.abs(x).max()) if x.size else 0.0
    return 20 * np.log10(pic) if pic > 0 else -np.inf


class DetectorVeu:
    """Embolcall sobre silero-vad amb caiguda a `librosa.effects.split`."""

    def __init__(self, backend="auto"):
        """Prepara el VAD amb el backend triat (silero o energia)."""
        self.backend = backend
        self._model = None
        self._get_ts = None
        if backend in ("auto", "silero"):
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                self._model = load_silero_vad()
                self._get_ts = get_speech_timestamps
                self.backend = "silero"
            except Exception as exc:
                if backend == "silero":
                    raise
                print(f"  [avis] silero-vad no disponible ({exc.__class__.__name__}): "
                      f"es fa servir el VAD d'energia de librosa", file=sys.stderr)
                self.backend = "energia"
        else:
            self.backend = "energia"

    def segments(self, senyal16k, min_silence_ms: int = 200):
        """Retorna [(inici_s, fi_s), ...] dels trams amb parla."""
        if self.backend == "silero":
            import torch

            with torch.no_grad():
                trams = self._get_ts(
                    torch.from_numpy(senyal16k), self._model,
                    sampling_rate=FS_ANALISI, return_seconds=True,
                    min_speech_duration_ms=200, min_silence_duration_ms=min_silence_ms,
                    speech_pad_ms=30,
                )
            return [(float(t["start"]), float(t["end"])) for t in trams]

        import librosa

        if not np.any(senyal16k):
            return []
        trams = librosa.effects.split(senyal16k, top_db=35, frame_length=1024,
                                      hop_length=256)
        return [(i / FS_ANALISI, f / FS_ANALISI) for i, f in trams]


class ComparadorLocutors:
    """Rebutja un locutor candidat si la seva veu s'assembla massa a la d'un ja acceptat (al banc actual o a."""

    def __init__(self, llindar):
        """Comparador de locutors amb el llindar de similitud."""
        self.llindar = llindar
        self.model = None
        if llindar <= 0:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
            )
        except Exception as exc:
            print(f"  [avis] speechbrain no disponible ({exc.__class__.__name__}): "
                  f"sense comprovacio de locutors duplicats", file=sys.stderr)
            self.llindar = 0

    @property
    def actiu(self):
        """Si el model de comparacio esta carregat."""
        return self.model is not None

    def embedding(self, senyal16k):
        """Vector unitari (norma 1) que representa la veu del clip -- normalitzat perque comparar-lo amb un altre."""
        import torch

        with torch.no_grad():
            vec = self.model.encode_batch(
                torch.from_numpy(senyal16k).unsqueeze(0)).squeeze().numpy()
        norma = np.linalg.norm(vec)
        return vec / norma if norma > 0 else vec

    def mes_semblant(self, embedding, acceptats):
        """`acceptats`: {fitxer: embedding}."""
        millor_fitxer, millor_sim = None, 0.0
        for fitxer, altre in acceptats.items():
            sim = float(np.dot(embedding, altre))
            if sim > millor_sim:
                millor_fitxer, millor_sim = fitxer, sim
        return millor_fitxer, millor_sim


def _mascara_parla(n_mostres, segments):
    """Mascara booleana de les mostres dins de segments de parla."""
    mascara = np.zeros(n_mostres, dtype=bool)
    for ini, fi in segments:
        mascara[int(ini * FS_ANALISI):int(fi * FS_ANALISI)] = True
    return mascara


def mesurar_qualitat(senyal16k, segments):
    """Metriques d'un clip a partir del senyal i dels trams de parla del VAD."""
    n = senyal16k.size
    durada = n / FS_ANALISI
    if durada < 0.05:
        return {"durada_s": round(durada, 3), "durada_parla_s": 0.0, "ratio_parla": 0.0,
                "snr_db": -999.0, "pic_dbfs": -999.0, "ratio_clipping": 0.0,
                "transitoris_per_min": 999.0}

    mascara = _mascara_parla(n, segments)
    finestra = int(0.025 * FS_ANALISI)
    n_fin = n // finestra
    talls = senyal16k[:n_fin * finestra].reshape(-1, finestra)
    rms = np.sqrt((talls ** 2).mean(axis=1))
    es_parla = mascara[:n_fin * finestra].reshape(-1, finestra).mean(axis=1) > 0.5

    rms_parla = float(np.median(rms[es_parla])) if es_parla.any() else 0.0
    terra = float(np.percentile(rms, 10)) if rms.size else 0.0
    if rms_parla <= 0:
        snr = -999.0
    elif terra <= 0:
        snr = SNR_MAX_DB
    else:
        snr = float(np.clip(20 * np.log10(rms_parla / terra), -999.0, SNR_MAX_DB))

    n_transitoris = 0
    if rms_parla > 0 and (~es_parla).any():
        n_transitoris = int((rms[~es_parla] > rms_parla * 10 ** (-9 / 20)).sum())

    return {
        "durada_s": round(durada, 3),
        "durada_parla_s": round(float(mascara.sum()) / FS_ANALISI, 3),
        "ratio_parla": round(float(mascara.mean()), 3),
        "snr_db": round(snr, 1),
        "pic_dbfs": round(float(_dbfs(senyal16k)), 1),
        "ratio_clipping": round(float(np.mean(np.abs(senyal16k) >= LLINDAR_CLIPPING)), 5),
        "transitoris_per_min": round(n_transitoris * 60 / durada, 1),
    }


def puntuar(metriques):
    """Puntuacio 0..1 per ordenar clips i locutors (no nomes acceptar/rebutjar): amb mes candidats que places."""
    snr = metriques["snr_db"]
    p_snr = np.clip((snr - MIN_SNR_DB) / (SNR_BO_DB - MIN_SNR_DB), 0, 1)
    p_parla = np.clip((metriques["ratio_parla"] - MIN_RATIO_PARLA)
                      / (0.95 - MIN_RATIO_PARLA), 0, 1)
    p_net = 1.0 - np.clip(metriques["transitoris_per_min"] / MAX_TRANSITORIS_PER_MIN, 0, 1)
    p_clip = 0.0 if metriques["ratio_clipping"] > MAX_RATIO_CLIPPING else 1.0
    return round(float(0.45 * p_snr + 0.25 * p_parla + 0.20 * p_net + 0.10 * p_clip), 3)


def clip_acceptable(metriques, min_snr, min_ratio_parla):
    """Motiu del descart, o None si passa."""
    if metriques["durada_s"] < MIN_DURADA_CLIP_S:
        return "massa curt"
    if metriques["durada_parla_s"] < MIN_DURADA_CLIP_S * 0.5:
        return "gairebe sense parla"
    if metriques["ratio_parla"] < min_ratio_parla:
        return f"ratio_parla={metriques['ratio_parla']:.2f}"
    if metriques["snr_db"] < min_snr:
        return f"snr={metriques['snr_db']:.1f}dB"
    if metriques["ratio_clipping"] > MAX_RATIO_CLIPPING:
        return f"clipping={metriques['ratio_clipping']:.4f}"
    if metriques["transitoris_per_min"] > MAX_TRANSITORIS_PER_MIN:
        return f"transitoris={metriques['transitoris_per_min']:.1f}/min"
    return None


def _retallar_vores(senyal16k, segments, marge_ini_s, marge_fi_s):
    """Com `retallar_silencis`, pero amb un marge diferent a cada costat."""
    if not segments:
        return senyal16k, segments
    ini = max(0.0, segments[0][0] - marge_ini_s)
    fi = min(senyal16k.size / FS_ANALISI, segments[-1][1] + marge_fi_s)
    tall = senyal16k[int(ini * FS_ANALISI):int(fi * FS_ANALISI)]
    return tall, [(s - ini, e - ini) for s, e in segments]


def retallar_silencis(senyal16k, segments, marge_s=SILENCI_VORA_MAX_S):
    """Elimina el silenci de les vores que passi de `marge_s`, i reajusta els segments a la nova base de temps."""
    return _retallar_vores(senyal16k, segments, marge_s, marge_s)


def _extreure(senyal16k, segments, ini_s, fi_s, coixi_s=0.1, dur_max=None):
    """Retalla [ini_s, fi_s] (amb un coixi als extrems) i hi reajusta els segments de parla i les metriques."""
    durada = senyal16k.size / FS_ANALISI
    ini = max(0.0, ini_s - coixi_s)
    fi = min(durada, fi_s + coixi_s)
    if dur_max is not None:
        fi = min(fi, ini + dur_max)
    tros = senyal16k[int(ini * FS_ANALISI):int(fi * FS_ANALISI)]
    segs = [(max(0.0, a - ini), min(fi - ini, b - ini))
            for a, b in segments if b > ini and a < fi]
    return tros, segs, mesurar_qualitat(tros, segs)


def _tall_per_energia(senyal16k, ini_s, dur_max, marge_s=0.6):
    """Punt de tall quan no hi ha cap pausa on tallar."""
    durada = senyal16k.size / FS_ANALISI
    fi_teoric = min(durada, ini_s + dur_max)
    a = int(max(ini_s, fi_teoric - marge_s) * FS_ANALISI)
    b = int(fi_teoric * FS_ANALISI)
    finestra = int(0.02 * FS_ANALISI)
    if b - a < 2 * finestra:
        return fi_teoric
    tros = senyal16k[a:b]
    n = (tros.size // finestra) * finestra
    rms = np.sqrt((tros[:n].reshape(-1, finestra) ** 2).mean(axis=1))
    return (a + (int(rms.argmin()) + 1) * finestra) / FS_ANALISI


def millor_finestra(senyal16k, segments, dur_min, dur_max):
    """En un clip mes llarg que `dur_max`."""
    millor = None
    for i in range(len(segments)):
        for j in range(i, len(segments)):
            durada = segments[j][1] - segments[i][0]
            if durada < dur_min:
                continue
            if durada > dur_max:
                break
            tros, segs, met = _extreure(senyal16k, segments, segments[i][0],
                                        segments[j][1], dur_max=dur_max)
            clau = (puntuar(met), durada)
            if millor is None or clau > millor[0]:
                millor = (clau, tros, segs, met)
    if millor is not None:
        _, tros, segs, met = millor
        return tros, segs, met

    ini = max(segments, key=lambda t: t[1] - t[0])[0]
    fi = _tall_per_energia(senyal16k, ini, dur_max - 0.2)
    if fi - ini < dur_min:
        return None
    return _extreure(senyal16k, segments, ini, fi, dur_max=dur_max)


def muntar_clip(candidats, dur_min, dur_max, silenci_s=SILENCI_CONFORT_S,
               marge_juntura_s=MARGE_JUNTURA_S):
    """Construeix un clip de `dur_min`..`dur_max` s a partir dels clips validats d'un locutor (ja ordenats de."""
    for senyal, segments, met, text in candidats:
        if dur_min <= met["durada_s"] <= dur_max:
            return senyal, [met], [text]

    peces, usats, textos_usats, total = [], [], [], 0.0
    for senyal, segments, met, text in candidats:
        if total >= dur_min:
            break
        restant = dur_max - total - (silenci_s if peces else 0.0)
        if restant <= 0:
            break
        if met["durada_s"] > restant:
            fins = max((e for _, e in segments if e <= restant), default=0.0)
            if fins < MIN_DURADA_CLIP_S:
                fins = _tall_per_energia(senyal, 0.0, restant)
            if fins < MIN_DURADA_CLIP_S:
                continue
            senyal = senyal[:int(min(fins + 0.1, restant) * FS_ANALISI)]
            segments = [(a, min(b, fins)) for a, b in segments if a < fins]
            met = dict(met, durada_s=round(senyal.size / FS_ANALISI, 3), retallat=True)
        peces.append([senyal, segments])
        total += senyal.size / FS_ANALISI + (silenci_s if len(peces) > 1 else 0.0)
        usats.append(met)
        textos_usats.append(text)

    if total < dur_min:
        return None, usats, textos_usats

    darrer = len(peces) - 1
    for k, (senyal, segments) in enumerate(peces):
        marge_ini = SILENCI_VORA_MAX_S if k == 0 else marge_juntura_s
        marge_fi = SILENCI_VORA_MAX_S if k == darrer else marge_juntura_s
        peces[k] = list(_retallar_vores(senyal, segments, marge_ini, marge_fi))

    silenci = np.zeros(int(FS_ANALISI * silenci_s), dtype=np.float32)
    trossos = []
    for k, (senyal, _segments) in enumerate(peces):
        if k > 0:
            trossos.append(silenci)
        trossos.append(senyal)
    return np.concatenate(trossos), usats, textos_usats


def normalitzar(senyal, sr, mode, lufs_objectiu, pic_objectiu_db):
    """-23 LUFS (EBU R128) o pic a -1 dBFS."""
    info = {"mode": mode}
    if mode == "lufs":
        try:
            import pyloudnorm as pyln
        except ImportError:
            print("  [avis] pyloudnorm no instal.lat: normalitzacio de pic",
                  file=sys.stderr)
            pyln = None
        if pyln is None or senyal.size < int(0.5 * sr):
            info["mode"] = "peak"
        else:
            mesurador = pyln.Meter(sr)
            entrada = mesurador.integrated_loudness(senyal.astype(np.float64))
            if np.isfinite(entrada):
                senyal = senyal * 10 ** ((lufs_objectiu - entrada) / 20)
                info["loudness_entrada_lufs"] = round(float(entrada), 2)
            else:
                info["mode"] = "peak"

    pic = float(np.abs(senyal).max()) if senyal.size else 0.0
    sostre = 10 ** (pic_objectiu_db / 20)
    if pic > 0 and (pic > sostre or info["mode"] == "peak"):
        senyal = senyal * (sostre / pic)

    if info["mode"] == "lufs":
        import pyloudnorm as pyln

        info["loudness_lufs"] = round(
            float(pyln.Meter(sr).integrated_loudness(senyal.astype(np.float64))), 2)
    info["pic_dbfs"] = round(float(_dbfs(senyal)), 2)
    return senyal.astype(np.float32), info


def escriure_wav(ruta, senyal16k, sample_rate, mode, lufs, pic_db):
    """Resample al sample rate de sortida i escriu PCM 16 bits mono."""
    senyal = senyal16k
    if sample_rate != FS_ANALISI:
        senyal = soxr.resample(senyal16k, FS_ANALISI, sample_rate, quality="VHQ")
    senyal, info = normalitzar(senyal, sample_rate, mode, lufs, pic_db)
    sf.write(ruta, senyal, sample_rate, subtype="PCM_16")
    info["durada_s"] = round(senyal.size / sample_rate, 3)
    return info


def _cv_descarregar_tsv(idioma, split, token, cache_dir):
    """Baixa un tsv de Common Voice."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        REPO_COMMON_VOICE, f"transcript/{idioma}/{split}.tsv", repo_type="dataset",
        token=token, cache_dir=cache_dir,
    ))


def _cv_descarregar_shards(idioma, split, token, cache_dir):
    """Descarrega els .tar d'audio d'un split i retorna {nom_shard: tarfile} i l'index {path_del_tsv."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    patro = f"audio/{idioma}/{split}/"
    shards = sorted(s.rfilename for s in api.dataset_info(REPO_COMMON_VOICE).siblings
                    if s.rfilename.startswith(patro))
    if not shards:
        raise RuntimeError(f"Cap .tar trobat a {patro} -- split o idioma incorrecte?")
    if len(shards) > 1:
        print(f"  ATENCIO: '{split}' te {len(shards)} .tar (descarrega de diversos GB). "
              f"Amb --split dev n'hi ha prou per a unes desenes de locutors.")

    tars = {}
    for nom_repo in shards:
        ruta = hf_hub_download(REPO_COMMON_VOICE, nom_repo, repo_type="dataset",
                               token=token, cache_dir=cache_dir)
        print(f"  [ok] {nom_repo}")
        tars[Path(nom_repo).stem] = tarfile.open(ruta)

    index = {}
    for nom_shard, tar in tars.items():
        for membre in tar.getnames():
            if "/" in membre:
                index[membre.split("/", 1)[1]] = nom_shard
    return tars, index


def _cv_files(tsv_path):
    """Files del .tsv amb pandas si hi es (lectura per trossos, el validated.tsv fa 140 MB)."""
    columnes = ["client_id", "path", "sentence", "gender", "age", "accents"]
    try:
        import pandas as pd

        lector = pd.read_csv(tsv_path, sep="\t", usecols=lambda c: c in columnes,
                             dtype=str, quoting=csv.QUOTE_NONE, chunksize=50_000,
                             on_bad_lines="skip")
        for tros in lector:
            yield from tros.fillna("").to_dict("records")
    except ImportError:
        with open(tsv_path, encoding="utf-8") as f:
            yield from csv.DictReader(f, delimiter="\t")
DetectorVeu 

def explorar(idioma, split, token, cache_dir):
    """Distribucio de `accents`/`gender`/`age` de tot el split (nomes el .tsv)."""
    tsv_path = _cv_descarregar_tsv(idioma, split, token, cache_dir)
    accents, generes, edats, total = Counter(), Counter(), Counter(), 0
    for fila in _cv_files(tsv_path):
        total += 1
        accents[fila.get("accents") or "(buit)"] += 1
        generes[fila.get("gender") or "(buit)"] += 1
        edats[fila.get("age") or "(buit)"] += 1

    print(f"{total} files a {idioma}/{split}.tsv\n")
    for titol, comptador in [("accents", accents), ("gender", generes), ("age", edats)]:
        print(f"-- {titol} --")
        for valor, n in comptador.most_common(20):
            print(f"  {n:6d}  {valor}")
        print()


def perfilar_common_voice(idioma, split, token, cache_dir, accent_keywords,
                          clips_per_locutor, max_locutors, exclou=frozenset()):
    """Fase barata: nomes el .tsv."""
    tsv_path = _cv_descarregar_tsv(idioma, split, token, cache_dir)
    keywords = [_normalitzar(k) for k in accent_keywords]

    perfils = {}
    for fila in _cv_files(tsv_path):
        accent = _normalitzar(fila.get("accents"))
        if not accent or not any(k in accent for k in keywords):
            continue
        cid = fila["client_id"]
        if cid in exclou:
            continue
        if cid not in perfils:
            if len(perfils) >= max_locutors:
                continue
            perfils[cid] = {
                "dataset": "common_voice", "speaker_id": cid,
                "gender": _genere_canonic(fila.get("gender")),
                "age": _sanejar(fila.get("age")), "accent": fila.get("accents", ""),
                "files": [],
            }
        if len(perfils[cid]["files"]) < clips_per_locutor:
            perfils[cid]["files"].append(fila)
    return perfils


def carregar_audio_cv(fila, tars, index):
    """Extreu i decodifica un mp3 del .tar (a memoria: no s'escriu res a disc)."""
    shard = index.get(fila["path"])
    if shard is None:
        return None
    dades = tars[shard].extractfile(f"{shard}/{fila['path']}").read()
    senyal, sr = sf.read(io.BytesIO(dades), dtype="float32")
    return _a_mono_16k(senyal, sr)


def _audio_de_fila(fila):
    """Mono 16 kHz d'una fila de `datasets`, sigui quina sigui la versio."""
    audio = fila["audio"]
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return _a_mono_16k(np.asarray(audio["array"], dtype=np.float32),
                               int(audio["sampling_rate"]))
        if audio.get("bytes"):
            senyal, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            return _a_mono_16k(senyal, sr)
        if audio.get("path"):
            senyal, sr = sf.read(audio["path"], dtype="float32")
            return _a_mono_16k(senyal, sr)
        return None
    mostres = audio.get_all_samples()
    return _a_mono_16k(mostres.data.numpy().T.squeeze(), int(mostres.sample_rate))


def _vp_fitxers_parquet(idioma, split, token, cache_dir):
    """Nomes els parquet del split demanat (com `_cv_descarregar_shards` amb els .tar de Common Voice)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    patro = f"{idioma}/{split}-"
    noms = sorted(s.rfilename for s in api.dataset_info(REPO_VOXPOPULI).siblings
                 if s.rfilename.startswith(patro) and s.rfilename.endswith(".parquet"))
    if not noms:
        raise RuntimeError(f"Cap parquet trobat a {patro}* -- split incorrecte?")
    if len(noms) > 1:
        print(f"  ATENCIO: '{split}' te {len(noms)} parquet (descarrega de diversos GB).")
    rutes = []
    for nom in noms:
        ruta = hf_hub_download(REPO_VOXPOPULI, nom, repo_type="dataset",
                               token=token, cache_dir=cache_dir)
        print(f"  [ok] {nom}")
        rutes.append(ruta)
    return rutes


def perfilar_voxpopuli(idioma, split, token, cache_dir, clips_per_locutor,
                       max_locutors, max_files, exclou=frozenset()):
    """Agrupa els clips per `speaker_id` i s'atura quan ja hi ha prou locutors amb prou segons acumulats."""
    from datasets import load_dataset

    fitxers = _vp_fitxers_parquet(idioma, split, token, cache_dir)
    ds = load_dataset("parquet", data_files=fitxers, split="train")
    try:
        from datasets import Audio

        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    perfils, llegides = {}, 0
    for fila in ds:
        llegides += 1
        if llegides > max_files:
            break
        sid = str(fila.get("speaker_id") or "").strip()
        if not sid or sid.lower() == "none" or sid in exclou:
            continue
        if sid not in perfils:
            if len(perfils) >= max_locutors:
                continue
            perfils[sid] = {
                "dataset": "voxpopuli", "speaker_id": sid,
                "gender": _genere_canonic(fila.get("gender")),
                "age": "unknown", "accent": fila.get("accent") or "",
                "clips": [], "segons_crus": 0.0,
            }
        perfil = perfils[sid]
        if len(perfil["clips"]) >= clips_per_locutor:
            continue

        try:
            senyal = _audio_de_fila(fila)
        except Exception as exc:
            print(f"  [avis] fila {fila.get('audio_id')} il.legible: {exc}", file=sys.stderr)
            continue
        if senyal is None:
            continue
        text = fila.get("raw_text") or fila.get("normalized_text") or ""
        perfil["clips"].append((senyal, text))
        perfil["segons_crus"] += senyal.size / FS_ANALISI

        llestos = sum(1 for p in perfils.values() if p["segons_crus"] >= DURADA_MIN_S * 1.5)
        if llestos >= max_locutors:
            break

    if llegides:
        print(f"  {llegides} files llegides, {len(perfils)} locutors")
    return perfils


def _round_robin(grups):
    """Intercala diverses llistes: [[a1,a2],[b1]] -> [a1,b1,a2]."""
    barrejat, grups = [], [list(g) for g in grups]
    while any(grups):
        for grup in grups:
            if grup:
                barrejat.append(grup.pop(0))
    return barrejat


class SelectorEquilibrat:
    """Decideix quin locutor es prova a continuacio: ~50% de veus masculines i ~50% de femenines."""

    def __init__(self, perfils, llavor):
        """Selector equilibrat: cues per quota, barrejades amb la llavor."""
        rng = np.random.default_rng(llavor)
        per_quota = defaultdict(list)
        for clau, perfil in perfils.items():
            per_quota[(perfil["gender"], _grup_edat(perfil["age"]))].append(clau)
        for claus in per_quota.values():
            rng.shuffle(claus)

        grups_per_genere = defaultdict(list)
        for (genere, _), claus in sorted(per_quota.items()):
            grups_per_genere[genere].append(claus)
        self.cues = {g: _round_robin(grups) for g, grups in grups_per_genere.items()}
        self.emesos = Counter()

    def __len__(self):
        """Perfils que queden a les cues."""
        return sum(len(cua) for cua in self.cues.values())

    def seguent(self):
        """Seguent locutor a provar, o None si no en queda cap."""
        disponibles = [g for g in ("female", "male") if self.cues.get(g)]
        if not disponibles:
            disponibles = sorted(g for g, cua in self.cues.items() if cua)
        if not disponibles:
            return None
        genere = min(disponibles, key=lambda g: (self.emesos[g], g))
        return self.cues[genere].pop(0)

    def registrar(self, genere):
        """Compta un locutor que ha superat el control de qualitat."""
        self.emesos[genere] += 1


def processar_locutor(perfil, clips_crus, vad, args, min_snr=None):
    """Clips crus -- parelles `(senyal."""
    min_snr = args.min_snr if min_snr is None else min_snr
    candidats = []
    for senyal, text in clips_crus:
        if senyal is None or senyal.size < int(MIN_DURADA_CLIP_S * FS_ANALISI):
            continue
        segments = vad.segments(senyal)
        if not segments:
            continue
        senyal, segments = retallar_silencis(senyal, segments)
        if senyal.size / FS_ANALISI > args.durada_max:
            millor = millor_finestra(senyal, segments, args.durada_min, args.durada_max)
            if millor is None:
                continue
            senyal, segments, met = millor
            met["text_exacte"] = False
        else:
            met = mesurar_qualitat(senyal, segments)
            met["text_exacte"] = True

        motiu = clip_acceptable(met, min_snr, args.min_ratio_parla)
        if motiu:
            continue
        met["puntuacio"] = puntuar(met)
        candidats.append((senyal, segments, met, text))

    if not candidats:
        return None, "cap clip supera el control de qualitat"

    candidats.sort(key=lambda c: c[2]["puntuacio"], reverse=True)
    senyal, usats, textos_usats = muntar_clip(candidats, args.durada_min, args.durada_max)
    if senyal is None:
        acumulat = sum(m["durada_s"] for m in usats)
        return None, f"nomes {acumulat:.1f}s de parla valida (calen {args.durada_min:.0f}s)"

    met_final = mesurar_qualitat(senyal, vad.segments(senyal))
    met_final["snr_db"] = min(m["snr_db"] for m in usats)
    met_final["transitoris_per_min"] = max(m["transitoris_per_min"] for m in usats)
    met_final["puntuacio_validacio"] = puntuar(met_final)
    met_final["n_clips_font"] = len(usats)
    met_final["text"] = " / ".join(t.strip() for t in textos_usats if t and t.strip())
    met_final["text_exacte"] = all(
        m.get("text_exacte", True) and not m.get("retallat") for m in usats)
    if met_final["puntuacio_validacio"] < args.min_puntuacio:
        return None, f"puntuacio {met_final['puntuacio_validacio']:.2f} < {args.min_puntuacio}"
    return (senyal, met_final), None


def nom_fitxer(perfil):
    """es_ES_<dataset>_<speaker>_<gender>_<age>.wav."""
    return (f"es_ES_{perfil['dataset']}_{_id_curt(perfil['speaker_id'])}"
            f"_{perfil['gender']}_{_sanejar(perfil['age'])}.wav")


def _locutors_existents(output_dir):
    """`speaker_id` ja presents al manifest, agrupats per `dataset`."""
    ruta = Path(output_dir) / MANIFEST_NAME
    if not ruta.exists():
        return defaultdict(set)
    try:
        manifest = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaultdict(set)

    existents = defaultdict(set)
    for entrada in manifest.get("clips", {}).values():
        existents[entrada["dataset"]].add(entrada["speaker_id"])
    return existents


def _embeddings_existents(output_dir):
    """Embeddings de veu ja guardats al manifest (camp `embedding`), com a {fitxer: np.array}."""
    ruta = Path(output_dir) / MANIFEST_NAME
    if not ruta.exists():
        return {}
    try:
        manifest = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    return {
        fitxer: np.array(entrada["embedding"], dtype=np.float32)
        for fitxer, entrada in manifest.get("clips", {}).items()
        if "embedding" in entrada
    }


def escriure_manifest(output_dir, entrades, args):
    """Fusiona amb el manifest d'execucions anteriors (si n'hi ha)."""
    ruta = Path(output_dir) / MANIFEST_NAME
    manifest = {"generat": None, "parametres": {}, "clips": {}}
    if ruta.exists():
        try:
            manifest.update(json.loads(ruta.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  [avis] {ruta} no es JSON valid: es reescriu de zero")

    manifest["clips"].update(entrades)
    manifest["generat"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["parametres"] = {
        "sample_rate": args.sample_rate, "durada_min_s": args.durada_min,
        "durada_max_s": args.durada_max, "normalitzacio": args.normalitzacio,
        "lufs_objectiu": args.lufs, "pic_objectiu_dbfs": args.pic_dbfs,
        "min_snr_db": args.min_snr, "min_snr_voxpopuli_db": args.min_snr_voxpopuli,
        "min_ratio_parla": args.min_ratio_parla,
        "min_puntuacio": args.min_puntuacio,
        "llindar_duplicat": args.llindar_duplicat,
        "fonts": {
            "common_voice": {"repo": REPO_COMMON_VOICE, "llicencia": "CC0-1.0",
                             "origen": "https://commonvoice.mozilla.org"},
            "voxpopuli": {"repo": REPO_VOXPOPULI, "llicencia": "CC0-1.0",
                          "origen": "https://github.com/facebookresearch/voxpopuli"},
        },
    }
    resum = Counter(c["gender"] for c in manifest["clips"].values())
    manifest["resum"] = {
        "n_clips": len(manifest["clips"]),
        "per_genere": dict(resum),
        "per_dataset": dict(Counter(c["dataset"] for c in manifest["clips"].values())),
        "durada_total_s": round(sum(c["durada_s"] for c in manifest["clips"].values()), 1),
    }
    ruta.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta, manifest


def _mostrar_seleccio(selector, perfils, quota, camp_clips):
    """Que sortiria si tots els locutors passessin el control de qualitat."""
    for _ in range(quota):
        clau = selector.seguent()
        if clau is None:
            break
        perfil = perfils[clau]
        selector.registrar(perfil["gender"])
        print(f"  [dry-run] {nom_fitxer(perfil):58s} clips={len(perfil[camp_clips])}")


def _construir_cv(args, vad, quota, exclou=frozenset(), comparador=None, acceptats=None):
    """Construeix el banc a partir de Common Voice fins a la quota."""
    if quota <= 0:
        return []
    print(f"\n== Common Voice ({args.idioma}/{args.split}) ==")
    if exclou:
        print(f"  {len(exclou)} locutors ja al manifest -- es descarten com a candidats")
    perfils = perfilar_common_voice(
        args.idioma, args.split, args.token, args.cache_dir, args.accent_contains,
        args.clips_per_locutor, max_locutors=quota * args.factor_candidats, exclou=exclou,
    )
    print(f"  {len(perfils)} locutors amb accent {args.accent_contains}")
    if not perfils:
        print("  Cap locutor. Prova --explore o relaxa --accent-contains.")
        return []

    selector = SelectorEquilibrat(perfils, args.llavor)
    if args.dry_run:
        _mostrar_seleccio(selector, perfils, quota, "files")
        return []

    print("  Descarregant audio...")
    tars, index = _cv_descarregar_shards(args.idioma, args.split, args.token, args.cache_dir)
    resultats = []
    while len(resultats) < quota:
        clau = selector.seguent()
        if clau is None:
            print(f"  [avis] candidats exhaurits amb {len(resultats)}/{quota} locutors")
            break
        perfil = perfils[clau]
        crus = [(carregar_audio_cv(f, tars, index), f.get("sentence", ""))
                for f in perfil["files"]]
        nous = _emetre(perfil, crus, vad, args, comparador=comparador, acceptats=acceptats)
        if nous:
            selector.registrar(perfil["gender"])
        resultats += nous
    for tar in tars.values():
        tar.close()
    return resultats


def _construir_voxpopuli(args, vad, quota, exclou=frozenset(), comparador=None, acceptats=None):
    """Construeix el banc a partir de VoxPopuli fins a la quota."""
    if quota <= 0:
        return []
    print(f"\n== VoxPopuli ({args.idioma}/{args.split_voxpopuli}) ==")
    if exclou:
        print(f"  {len(exclou)} locutors ja al manifest -- es descarten com a candidats")
    perfils = perfilar_voxpopuli(
        args.idioma, args.split_voxpopuli, args.token, args.cache_dir,
        args.clips_per_locutor, max_locutors=quota * args.factor_candidats,
        max_files=args.max_files_voxpopuli, exclou=exclou,
    )
    if not perfils:
        print("  Cap locutor trobat.")
        return []

    selector = SelectorEquilibrat(perfils, args.llavor)
    if args.dry_run:
        _mostrar_seleccio(selector, perfils, quota, "clips")
        return []

    resultats = []
    while len(resultats) < quota:
        clau = selector.seguent()
        if clau is None:
            print(f"  [avis] candidats exhaurits amb {len(resultats)}/{quota} locutors")
            break
        perfil = perfils[clau]
        nous = _emetre(perfil, perfil["clips"], vad, args, min_snr=args.min_snr_voxpopuli,
                      comparador=comparador, acceptats=acceptats)
        if nous:
            selector.registrar(perfil["gender"])
        resultats += nous
    return resultats


def _emetre(perfil, clips_crus, vad, args, min_snr=None, comparador=None, acceptats=None):
    """Processa un locutor i, si passa, escriu el .wav."""
    resultat, motiu = processar_locutor(perfil, clips_crus, vad, args, min_snr=min_snr)
    nom = nom_fitxer(perfil)
    if resultat is None:
        print(f"  [--] {nom:60s} descartat: {motiu}")
        return []

    senyal, met = resultat

    embedding = None
    if comparador is not None and comparador.actiu:
        embedding = comparador.embedding(senyal)
        fitxer_semblant, sim = comparador.mes_semblant(embedding, acceptats or {})
        if sim > comparador.llindar:
            print(f"  [--] {nom:60s} descartat: possible duplicat de "
                  f"{fitxer_semblant} (sim={sim:.2f})")
            return []

    info = escriure_wav(args.output_dir / nom, senyal, args.sample_rate,
                        args.normalitzacio, args.lufs, args.pic_dbfs)
    entrada = {
        "fitxer": nom, "dataset": perfil["dataset"],
        "speaker_id": perfil["speaker_id"], "speaker_id_curt": _id_curt(perfil["speaker_id"]),
        "gender": perfil["gender"], "age": perfil["age"],
        "grup_edat": _grup_edat(perfil["age"]), "accent": perfil["accent"],
        "idioma": "es_ES", "sample_rate": args.sample_rate,
        "durada_s": info["durada_s"], "n_clips_font": met["n_clips_font"],
        "puntuacio_validacio": met["puntuacio_validacio"],
        "text": met["text"], "text_exacte": met["text_exacte"],
        "qualitat": {k: met[k] for k in ("snr_db", "ratio_parla", "durada_parla_s",
                                         "ratio_clipping", "transitoris_per_min")},
        "normalitzacio": info,
        "font_real": perfil["dataset"] == "voxpopuli",
    }
    if embedding is not None:
        entrada["embedding"] = [round(float(x), 4) for x in embedding]
        acceptats[nom] = embedding
    print(f"  [ok] {nom:60s} {info['durada_s']:5.1f}s  snr={met['snr_db']:5.1f}dB  "
          f"punt={met['puntuacio_validacio']:.2f}  clips={met['n_clips_font']}")
    return [(nom, entrada)]


def main():
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=["common_voice", "voxpopuli"],
                        choices=["common_voice", "voxpopuli"],
                        help="Fonts a fer servir (per defecte, totes dues)")
    parser.add_argument("--n-locutors", type=int, default=20,
                        help="Locutors objectiu en total, repartits entre fonts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--idioma", default="es")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"],
                        help="Split de Common Voice: 'dev'/'test' son un sol .tar "
                             "(~770 MB); 'train' en te 9 (diversos GB)")
    parser.add_argument("--split-voxpopuli", default="validation",
                        choices=["validation", "test", "train"],
                        help="Split de VoxPopuli (streaming; 'train' son 10 parquet)")
    parser.add_argument("--explore", action="store_true",
                        help="Nomes imprimeix la distribucio d'accents/gender/age de "
                             "Common Voice i surt (no baixa cap audio)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fa el perfilat i la seleccio i mostra que sortiria, "
                             "sense processar ni escriure audio")
    parser.add_argument("--accent-contains", nargs="*", default=ACCENT_KEYWORDS_ES_ES,
                        help="Paraules clau (sense accents, minuscules) que ha de "
                             "contenir el camp `accents` de Common Voice")
    parser.add_argument("--clips-per-locutor", type=int, default=8,
                        help="Clips crus maxims a considerar per locutor")
    parser.add_argument("--factor-candidats", type=int, default=4,
                        help="Candidats a perfilar per cada placa (el control de "
                             "qualitat en descarta una part)")
    parser.add_argument("--max-files-voxpopuli", type=int, default=1500,
                        help="Files maximes a llegir/decodificar del parquet de "
                             "VoxPopuli durant el perfilat, abans de rendir-se si "
                             "no hi ha prou candidats (el parquet ja esta baixat "
                             "sencer; aixo nomes limita CPU de decodificacio)")
    parser.add_argument("--durada-min", type=float, default=DURADA_MIN_S,
                        help="Durada minima d'un clip per publicar-lo TOT SOL "
                             "(OmniVoice recomana 3-10s). Nomes s'empalmen clips "
                             "quan cap clip solt del locutor hi arriba.")
    parser.add_argument("--durada-max", type=float, default=DURADA_MAX_S)
    parser.add_argument("--normalitzacio", default="lufs", choices=["lufs", "peak"],
                        help="'lufs' = -23 LUFS amb sostre de pic; 'peak' = pic a -1 dBFS")
    parser.add_argument("--lufs", type=float, default=DEFAULT_LUFS)
    parser.add_argument("--pic-dbfs", type=float, default=DEFAULT_PEAK_DBFS)
    parser.add_argument("--vad", default="auto", choices=["auto", "silero", "energia"],
                        help="'auto' fa servir silero-vad si esta instal.lat i cau "
                             "al VAD d'energia de librosa si no")
    parser.add_argument("--min-snr", type=float, default=MIN_SNR_DB)
    parser.add_argument("--min-snr-voxpopuli", type=float, default=MIN_SNR_VOXPOPULI_DB,
                        help="Llindar d'SNR nomes per a VoxPopuli: son gravacions "
                             "de sala reals (18-25 dB tipics) i el llindar de "
                             "Common Voice, pensat per a gravacions netes, en "
                             "descartaria la majoria")
    parser.add_argument("--min-ratio-parla", type=float, default=MIN_RATIO_PARLA)
    parser.add_argument("--min-puntuacio", type=float, default=MIN_PUNTUACIO)
    parser.add_argument("--llindar-duplicat", type=float, default=0.75,
                        help="Similitud de veu (embedding ECAPA-TDNN, 0-1) per "
                             "sobre de la qual un candidat es descarta com a "
                             "possible duplicat d'un locutor ja acceptat. <= 0 "
                             "desactiva la comprovacio (no cal speechbrain)")
    parser.add_argument("--llavor", type=int, default=42)
    parser.add_argument("--token", default=None,
                        help="Token de HF; per defecte, HF_TOKEN de l'entorn o la "
                             "sessio de `huggingface-cli login`")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    if args.explore:
        explorar(args.idioma, args.split, args.token, args.cache_dir)
        return

    if args.durada_min > args.durada_max:
        parser.error("--durada-min no pot ser mes gran que --durada-max")

    vad = None if args.dry_run else DetectorVeu(args.vad)
    if vad:
        print(f"VAD: {vad.backend}")
    comparador = None if args.dry_run else ComparadorLocutors(args.llindar_duplicat)
    if comparador and comparador.actiu:
        print(f"Comprovacio de duplicats: activa (llindar={args.llindar_duplicat})")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quotes = {f: args.n_locutors // len(args.sources) for f in args.sources}
    quotes[args.sources[-1]] += args.n_locutors - sum(quotes.values())

    existents = _locutors_existents(args.output_dir)
    acceptats = _embeddings_existents(args.output_dir) if comparador and comparador.actiu else {}

    entrades = {}
    if "common_voice" in args.sources:
        entrades.update(dict(_construir_cv(
            args, vad, quotes["common_voice"], existents["common_voice"],
            comparador=comparador, acceptats=acceptats)))
    if "voxpopuli" in args.sources:
        entrades.update(dict(_construir_voxpopuli(
            args, vad, quotes["voxpopuli"], existents["voxpopuli"],
            comparador=comparador, acceptats=acceptats)))

    if args.dry_run:
        print("\n(dry-run: no s'ha processat ni escrit cap audio)")
        return
    if not entrades:
        print("\nCap clip generat. Amb --min-snr / --min-ratio-parla mes baixos "
              "o mes --factor-candidats hi entrarien mes locutors.")
        return

    ruta, manifest = escriure_manifest(args.output_dir, entrades, args)
    resum = manifest["resum"]
    print(f"\n{len(entrades)} clips nous a {args.output_dir}/ "
          f"({resum['n_clips']} al banc, {resum['durada_total_s']:.0f}s en total)")
    print(f"Genere: {resum['per_genere']}   Font: {resum['per_dataset']}")
    print(f"Manifest: {ruta}")


if __name__ == "__main__":
    main()

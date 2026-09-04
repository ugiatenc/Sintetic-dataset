#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construeix un banc de llavors de veu (es_ES) per a clonatge zero-shot amb OmniVoice.

Dues fonts, amb rols diferents:

1. **Common Voice** (`fsicoli/common_voice_17_0`, CC0): gravacions netes de
   navegador/mobil, amb metadades autodeclarades de `gender`, `age` i `accents`.
   Es filtra estrictament per accent peninsular (el camp `accents` dels .tsv reals
   sempre comenca per "España: ..." per als locutors d'aqui), i son les veus que
   despres passen pel simulador acustic sencer.
2. **VoxPopuli** (`facebook/voxpopuli`, config `es`, CC0): parla real del Parlament
   Europeu. Porta `speaker_id` i `gender` a les anotacions, pero **no** edat. L'audio
   ja ve ambientat (sala + soroll de fons), per aixo al notebook aquestes veus es
   marquen `font_real=True` i salten el simulador.

**Per que `--split dev` per defecte a Common Voice:** `validated.tsv` es la unio de
train+dev+test (>400k files per `es`), pero l'audio nomes es descarrega en blocs
(.tar) per split -- train te 9 blocs de diversos GB, mentre que dev cap en un sol
.tar (~770 MB) amb gent d'una gran varietat d'accents i generes.

**Per que VoxPopuli va en streaming:** els parquet d'`es` pesen ~1 GB per split i
no calen sencers; es llegeixen fila a fila i s'atura quan ja hi ha prou locutors
(`--max-files-voxpopuli`).

Pipeline per locutor:

    clips crus -> VAD -> control de qualitat -> retall de silencis
              -> motor de durada (talla els llargs / empalma els curts)
              -> resample 24 kHz -> normalitzacio (-23 LUFS o pic -1 dBFS)
              -> es_ES_<dataset>_<speaker>_<gender>_<age>.wav + manifest JSON

**Integracio amb el pipeline:** `lab/generate_voices_environments.ipynb` (seccio 2)
encara llegeix les veus netes de `datasets/audios/referencia/locutors.json`, que es el
format que escrivia la versio anterior d'aquest script. El manifest d'aqui
(`voice_bank_manifest.json`) porta la mateixa informacio i mes, pero amb un altre nom i
una altra estructura: per fer-lo servir des del notebook cal adaptar-hi la carrega.

Exemples:
    # Explorar els valors de accents/gender/age de Common Voice (nomes baixa el .tsv)
    python3 src/build_voice_bank.py --explore

    # 20 locutors equilibrats 50/50, de les dues fonts
    python3 src/build_voice_bank.py --n-locutors 20

    # Nomes Common Voice, a la carpeta que llegeix el notebook
    python3 src/build_voice_bank.py --sources common_voice \
        --output-dir datasets/audios/referencia
"""

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

# --- Fonts ---------------------------------------------------------------
REPO_COMMON_VOICE = "fsicoli/common_voice_17_0"
REPO_VOXPOPULI = "facebook/voxpopuli"

# Als .tsv reals, els accents peninsulars sempre comencen per "España: ..."
ACCENT_KEYWORDS_ES_ES = ["espana"]

# --- Sortida -------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("data/voice_seeds")
MANIFEST_NAME = "voice_bank_manifest.json"
DEFAULT_SAMPLE_RATE = 24000          # master d'OmniVoice; 16 bits, mono, PCM WAV
DEFAULT_LUFS = -23.0                 # EBU R128
DEFAULT_PEAK_DBFS = -1.0             # sostre de pic, tambe en mode LUFS

# --- Motor de durada -----------------------------------------------------
DURADA_MIN_S = 8.0
DURADA_MAX_S = 10.0
SILENCI_CONFORT_S = 0.2              # entre clips empalmats
SILENCI_VORA_MAX_S = 0.5             # silenci tolerat a l'inici/final

# --- Control de qualitat -------------------------------------------------
FS_ANALISI = 16000                   # el VAD i les metriques treballen aqui
MIN_RATIO_PARLA = 0.60               # parla / durada total del clip
MIN_SNR_DB = 18.0                    # nivell de parla vs. terra de soroll
MIN_SNR_VOXPOPULI_DB = 14.0          # VoxPopuli es sala real, no cabina
SNR_BO_DB = 35.0                     # a partir d'aqui, la puntuacio d'SNR satura
SNR_MAX_DB = 60.0                    # sostre: per sobre, la mesura ja no vol dir res
MAX_RATIO_CLIPPING = 0.001           # fraccio de mostres saturades
LLINDAR_CLIPPING = 0.98
MAX_TRANSITORIS_PER_MIN = 6.0        # pics fora de parla (rialles, aplaudiments)
MIN_DURADA_CLIP_S = 1.0
MIN_PUNTUACIO = 0.5                  # puntuacio de validacio minima per publicar

GRUPS_EDAT = {  # etiquetes de Common Voice -> grup de mostreig
    "teens": "jove", "twenties": "jove",
    "thirties": "adult", "fourties": "adult", "fifties": "adult",
    "sixties": "gran", "seventies": "gran", "eighties": "gran", "nineties": "gran",
}
GENERES = {  # etiquetes de les dues fonts -> valor canonic del nom de fitxer
    "male_masculine": "male", "female_feminine": "female",
    "male": "male", "female": "female",
}


# =========================================================================
# Utilitats
# =========================================================================

def _normalitzar(text):
    """Treu accents/majuscules per comparar paraules clau sense sorpreses."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return text.lower()


def _sanejar(text, per_defecte="unknown"):
    """Deixa un fragment apte per a un nom de fitxer."""
    net = re.sub(r"[^A-Za-z0-9]+", "-", _normalitzar(text)).strip("-")
    return net or per_defecte


def _genere_canonic(valor):
    return GENERES.get(_normalitzar(valor).strip(), "unknown")


def _grup_edat(valor):
    return GRUPS_EDAT.get(_normalitzar(valor).strip(), "desconegut")


def _id_curt(speaker_id):
    """`client_id` de Common Voice fa 128 hex: massa per a un nom de fitxer.
    Un prefix del sha1 es estable entre execucions i prou curt per llegir-lo."""
    net = _sanejar(speaker_id, "")
    if net and len(net) <= 12:
        return net  # els speaker_id de VoxPopuli ja son curts (p.ex. "96678")
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
    pic = float(np.abs(x).max()) if x.size else 0.0
    return 20 * np.log10(pic) if pic > 0 else -np.inf


# =========================================================================
# 1. Deteccio d'activitat de veu (VAD)
# =========================================================================

class DetectorVeu:
    """Embolcall sobre silero-vad amb caiguda a `librosa.effects.split`.

    Silero es una xarxa entrenada per a veu i distingeix parla de soroll
    estacionari, musica o aplaudiments; `librosa.effects.split` nomes mira
    energia, aixi que amb fons sorollos marca com a "parla" qualsevol cosa
    forta. Per aixo el backend d'energia es un pla B (una instal.lacio sense
    silero-vad segueix funcionant) i no el cami per defecte.
    """

    def __init__(self, backend="auto"):
        self.backend = backend
        self._model = None
        self._get_ts = None
        if backend in ("auto", "silero"):
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad

                self._model = load_silero_vad()
                self._get_ts = get_speech_timestamps
                self.backend = "silero"
            except Exception as exc:  # sense xarxa, sense torch, sense paquet...
                if backend == "silero":
                    raise
                print(f"  [avis] silero-vad no disponible ({exc.__class__.__name__}): "
                      f"es fa servir el VAD d'energia de librosa", file=sys.stderr)
                self.backend = "energia"
        else:
            self.backend = "energia"

    def segments(self, senyal16k):
        """Retorna [(inici_s, fi_s), ...] dels trams amb parla."""
        if self.backend == "silero":
            import torch

            with torch.no_grad():
                trams = self._get_ts(
                    torch.from_numpy(senyal16k), self._model,
                    sampling_rate=FS_ANALISI, return_seconds=True,
                    min_speech_duration_ms=200, min_silence_duration_ms=200,
                    speech_pad_ms=30,
                )
            return [(float(t["start"]), float(t["end"])) for t in trams]

        import librosa

        if not np.any(senyal16k):
            return []
        trams = librosa.effects.split(senyal16k, top_db=35, frame_length=1024,
                                      hop_length=256)
        return [(i / FS_ANALISI, f / FS_ANALISI) for i, f in trams]


# =========================================================================
# 2. Control de qualitat
# =========================================================================

def _mascara_parla(n_mostres, segments):
    mascara = np.zeros(n_mostres, dtype=bool)
    for ini, fi in segments:
        mascara[int(ini * FS_ANALISI):int(fi * FS_ANALISI)] = True
    return mascara


def mesurar_qualitat(senyal16k, segments):
    """Metriques d'un clip a partir del senyal i dels trams de parla del VAD.

    `snr_db` compara el nivell de la parla amb el terra de soroll, estimat com
    el percentil 10 de l'energia per finestres de 25 ms. Comparar nomes contra
    les **pauses** del VAD no serveix aqui: un discurs seguit (el cas normal a
    VoxPopuli) no en te cap, i donaria SNR infinit encara que la sala fos
    sorollosa. El percentil, en canvi, cau als buits entre paraules, que
    existeixen sempre.

    `transitoris_per_min` compta pics d'energia **fora** de la parla, que es on
    cauen rialles, aplaudiments i cops de micro.
    """
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
        snr = SNR_MAX_DB  # silenci digital: no hi ha res a mesurar
    else:
        snr = float(np.clip(20 * np.log10(rms_parla / terra), -999.0, SNR_MAX_DB))

    # Un aplaudiment o una rialla sobresurten molt per sobre del fons, pero
    # queden fora dels trams que el VAD marca com a parla.
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
    """Puntuacio 0..1 per ordenar clips i locutors (no nomes acceptar/rebutjar):
    amb mes candidats que places, val la pena quedar-se els millors."""
    snr = metriques["snr_db"]
    p_snr = np.clip((snr - MIN_SNR_DB) / (SNR_BO_DB - MIN_SNR_DB), 0, 1)
    p_parla = np.clip((metriques["ratio_parla"] - MIN_RATIO_PARLA)
                      / (0.95 - MIN_RATIO_PARLA), 0, 1)
    p_net = 1.0 - np.clip(metriques["transitoris_per_min"] / MAX_TRANSITORIS_PER_MIN, 0, 1)
    p_clip = 0.0 if metriques["ratio_clipping"] > MAX_RATIO_CLIPPING else 1.0
    return round(float(0.45 * p_snr + 0.25 * p_parla + 0.20 * p_net + 0.10 * p_clip), 3)


def clip_acceptable(metriques, min_snr, min_ratio_parla):
    """Motiu del descart, o None si passa. Retornar el motiu (i no un bool)
    permet imprimir per que s'ha quedat sense candidats un locutor."""
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


# =========================================================================
# 3. Motor de durada: retall, tall de llargs i empalmat de curts
# =========================================================================

def retallar_silencis(senyal16k, segments, marge_s=SILENCI_VORA_MAX_S):
    """Elimina el silenci de les vores que passi de `marge_s`, i reajusta els
    segments a la nova base de temps."""
    if not segments:
        return senyal16k, segments
    ini = max(0.0, segments[0][0] - marge_s)
    fi = min(senyal16k.size / FS_ANALISI, segments[-1][1] + marge_s)
    tall = senyal16k[int(ini * FS_ANALISI):int(fi * FS_ANALISI)]
    return tall, [(s - ini, e - ini) for s, e in segments]


def _extreure(senyal16k, segments, ini_s, fi_s, coixi_s=0.1, dur_max=None):
    """Retalla [ini_s, fi_s] (amb un coixi als extrems) i hi reajusta els
    segments de parla i les metriques. `dur_max` limita el resultat **coixi
    inclos**: sense aixo, una finestra de 10.0 s en sortia de 10.2 i el motor de
    durada la rebutjava per passar-se del maxim que ell mateix havia demanat."""
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
    """Punt de tall quan no hi ha cap pausa on tallar: el minim d'energia dins
    dels ultims `marge_s` de la finestra, per caure en una respiracio i no
    enmig d'una vocal."""
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
    """En un clip mes llarg que `dur_max`, tria el millor tram que encaixi entre
    `dur_min` i `dur_max` **tallant per pauses**, mai a mitja paraula.

    Es proven totes les finestres [segment i .. segment j] i es queda la de mes
    puntuacio; a igualtat, la mes llarga (mes context per al clonador)."""
    millor = None
    for i in range(len(segments)):
        for j in range(i, len(segments)):
            durada = segments[j][1] - segments[i][0]
            if durada < dur_min:
                continue
            if durada > dur_max:
                break  # allargant j nomes creix: no cal seguir
            tros, segs, met = _extreure(senyal16k, segments, segments[i][0],
                                        segments[j][1], dur_max=dur_max)
            clau = (puntuar(met), durada)
            if millor is None or clau > millor[0]:
                millor = (clau, tros, segs, met)
    if millor is not None:
        _, tros, segs, met = millor
        return tros, segs, met

    # Pla B: cap parella de pauses encaixa a la finestra. Passa sempre que el
    # VAD marca un discurs seguit com un unic tram de mes de `dur_max` (tipic de
    # VoxPopuli), i abans aixo descartava el locutor sencer. Es talla dins del
    # tram mes llarg, pel punt de menys energia.
    ini = max(segments, key=lambda t: t[1] - t[0])[0]
    fi = _tall_per_energia(senyal16k, ini, dur_max - 0.2)  # 0.2 = coixi dels dos extrems
    if fi - ini < dur_min:
        return None
    return _extreure(senyal16k, segments, ini, fi, dur_max=dur_max)


def muntar_clip(candidats, dur_min, dur_max, silenci_s=SILENCI_CONFORT_S):
    """Construeix un clip de `dur_min`..`dur_max` s a partir dels clips validats
    d'un locutor (ja ordenats de millor a pitjor).

    - Un sol clip que ja hi encaixi: es fa servir tal qual.
    - Clips curts: s'empalmen amb un silenci de confort de 0.2 s entre frases.
      No hi va cap crossfade: son frases diferents, i encavalcar-les crearia una
      transicio que no existeix en cap gravacio real.
    - Menys de `dur_min` s de parla valida acumulada: el locutor es descarta.
    """
    for senyal, segments, met in candidats:  # cas ideal: un clip ja bo
        if dur_min <= met["durada_s"] <= dur_max:
            return senyal, [met]

    silenci = np.zeros(int(FS_ANALISI * silenci_s), dtype=np.float32)
    trossos, usats, total = [], [], 0.0
    for senyal, segments, met in candidats:
        if total >= dur_min:
            break
        restant = dur_max - total - (silenci_s if trossos else 0.0)
        if restant <= 0:
            break
        if met["durada_s"] > restant:
            # Nomes serveix un tros: es talla per la pausa mes propera per sota
            # de `restant`, per no partir cap paraula.
            fins = max((e for _, e in segments if e <= restant), default=0.0)
            if fins < MIN_DURADA_CLIP_S:
                # Cap pausa prou aviat (tipic d'una frase seguida de Common
                # Voice): es talla pel punt de menys energia. Descartar el tros
                # sencer, com es feia abans, deixava fora locutors que nomes
                # necessitaven un parell de segons per arribar al minim.
                fins = _tall_per_energia(senyal, 0.0, restant)
            if fins < MIN_DURADA_CLIP_S:
                continue
            senyal = senyal[:int(min(fins + 0.1, restant) * FS_ANALISI)]
            met = dict(met, durada_s=round(senyal.size / FS_ANALISI, 3), retallat=True)
        if trossos:
            trossos.append(silenci)
            total += silenci_s
        trossos.append(senyal)
        total += senyal.size / FS_ANALISI
        usats.append(met)

    if total < dur_min:
        return None, usats
    return np.concatenate(trossos), usats


# =========================================================================
# 4. Normalitzacio i escriptura
# =========================================================================

def normalitzar(senyal, sr, mode, lufs_objectiu, pic_objectiu_db):
    """-23 LUFS (EBU R128) o pic a -1 dBFS. En mode LUFS el pic tambe es limita:
    una frase molt dinamica pot demanar un guany que saturi el WAV de 16 bits.

    `mode` es el que s'ha demanat; `info["mode"]` es el que s'ha pogut aplicar
    (un clip de menys de 0.5 s no te mesura de sonoritat valida, i sense
    pyloudnorm instal.lat nomes queda la normalitzacio de pic).
    """
    info = {"mode": mode}
    if mode == "lufs":
        try:
            import pyloudnorm as pyln
        except ImportError:
            print("  [avis] pyloudnorm no instal.lat: normalitzacio de pic",
                  file=sys.stderr)
            pyln = None
        # El bloc del mesurador es de 400 ms: per sota no hi ha mesura valida.
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
        # VoxPopuli ja ve a 16 kHz i Common Voice a 32/48: pujar a 24 kHz no
        # afegeix informacio, pero deixa tot el banc amb el mateix format
        # (el clonador rebutja barrejar sample rates).
        senyal = soxr.resample(senyal16k, FS_ANALISI, sample_rate, quality="VHQ")
    senyal, info = normalitzar(senyal, sample_rate, mode, lufs, pic_db)
    sf.write(ruta, senyal, sample_rate, subtype="PCM_16")
    info["durada_s"] = round(senyal.size / sample_rate, 3)
    return info


# =========================================================================
# 5. Font: Common Voice
# =========================================================================

def _cv_descarregar_tsv(idioma, split, token, cache_dir):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        REPO_COMMON_VOICE, f"transcript/{idioma}/{split}.tsv", repo_type="dataset",
        token=token, cache_dir=cache_dir,
    ))


def _cv_descarregar_shards(idioma, split, token, cache_dir):
    """Descarrega els .tar d'audio d'un split i retorna {nom_shard: tarfile} i
    l'index {path_del_tsv: nom_shard}."""
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

    # Amb mes d'un .tar cal saber a quin viu cada fitxer. getnames() nomes
    # llegeix capceleres, aixi que es rapid encara que el .tar pesi GB.
    index = {}
    for nom_shard, tar in tars.items():
        for membre in tar.getnames():
            if "/" in membre:  # ignora l'entrada de directori arrel
                index[membre.split("/", 1)[1]] = nom_shard
    return tars, index


def _cv_files(tsv_path):
    """Files del .tsv amb pandas si hi es (lectura per trossos, el validated.tsv
    fa 140 MB), i amb csv com a pla B."""
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


def explorar(idioma, split, token, cache_dir):
    """Distribucio de `accents`/`gender`/`age` de tot el split (nomes el .tsv),
    per triar les paraules clau del filtre sense endevinar-les."""
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
                          clips_per_locutor, max_locutors):
    """Fase barata: nomes el .tsv. Retorna {speaker_id: perfil} amb les files
    de cada locutor, sense tocar ni un byte d'audio."""
    tsv_path = _cv_descarregar_tsv(idioma, split, token, cache_dir)
    keywords = [_normalitzar(k) for k in accent_keywords]

    perfils = {}
    for fila in _cv_files(tsv_path):
        accent = _normalitzar(fila.get("accents"))
        if not accent or not any(k in accent for k in keywords):
            continue
        cid = fila["client_id"]
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


# =========================================================================
# 6. Font: VoxPopuli
# =========================================================================

def _audio_de_fila(fila):
    """Mono 16 kHz d'una fila de `datasets`, sigui quina sigui la versio.

    La columna `audio` ha canviat de forma entre versions: fins a `datasets` 4
    arribava com a dict amb `array`/`sampling_rate`, i a partir de la 5 com a
    `AudioDecoder` de torchcodec. Amb `decode=False` tornen els bytes crus, que
    es la forma que no depen de cap d'aquests dos camins.
    """
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
    mostres = audio.get_all_samples()  # torchcodec (datasets >= 5)
    return _a_mono_16k(mostres.data.numpy().T.squeeze(), int(mostres.sample_rate))


def perfilar_voxpopuli(idioma, split, token, cache_dir, clips_per_locutor,
                       max_locutors, max_files):
    """Passada en streaming: agrupa els clips per `speaker_id` i s'atura quan ja
    hi ha prou locutors amb prou segons acumulats.

    VoxPopuli no publica edat, aixi que `age` queda "unknown" i el balanceig per
    grups d'edat nomes actua sobre Common Voice.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO_VOXPOPULI, idioma, split=split, streaming=True,
                      token=token, cache_dir=cache_dir)
    try:
        from datasets import Audio

        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass  # `_audio_de_fila` tambe sap llegir la columna descodificada

    perfils, llegides = {}, 0
    for fila in ds:
        llegides += 1
        if llegides > max_files:
            break
        sid = str(fila.get("speaker_id") or "").strip()
        if not sid or sid.lower() == "none":
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
        perfil["clips"].append(senyal)
        perfil["segons_crus"] += senyal.size / FS_ANALISI

        llestos = sum(1 for p in perfils.values() if p["segons_crus"] >= DURADA_MIN_S * 1.5)
        if llestos >= max_locutors:
            break

    if llegides:
        print(f"  {llegides} files llegides en streaming, {len(perfils)} locutors")
    return perfils


# =========================================================================
# 7. Seleccio equilibrada
# =========================================================================

def _round_robin(grups):
    """Intercala diverses llistes: [[a1,a2],[b1]] -> [a1,b1,a2]. Serveix per
    repartir els grups d'edat dins d'un mateix genere."""
    barrejat, grups = [], [list(g) for g in grups]
    while any(grups):
        for grup in grups:
            if grup:
                barrejat.append(grup.pop(0))
    return barrejat


class SelectorEquilibrat:
    """Decideix quin locutor es prova a continuacio: ~50% de veus masculines i
    ~50% de femenines, i dins de cada genere repartides entre grups d'edat.

    La decisio es pren sobre la marxa, i no d'entrada, perque el control de
    qualitat descarta locutors: amb un ordre fix, tres descarts seguits de veus
    femenines s'acabaven cobrint amb masculines i el banc quedava esbiaixat.
    Aqui un descart no compta, aixi que el genere que en te menys torna a tenir
    la preferencia fins que s'acaben els candidats.

    Els locutors sense genere declarat nomes entren quan ja no en queda cap
    d'etiquetat: no es pot equilibrar el que no se sap.
    """

    def __init__(self, perfils, llavor):
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


# =========================================================================
# 8. Processament d'un locutor
# =========================================================================

def processar_locutor(perfil, clips_crus, vad, args, min_snr=None):
    """Clips crus d'un locutor -> senyal final a 16 kHz + metadades, o (None, motiu)."""
    min_snr = args.min_snr if min_snr is None else min_snr
    candidats = []
    for senyal in clips_crus:
        if senyal is None or senyal.size < int(MIN_DURADA_CLIP_S * FS_ANALISI):
            continue
        segments = vad.segments(senyal)
        if not segments:
            continue
        senyal, segments = retallar_silencis(senyal, segments)
        if senyal.size / FS_ANALISI > args.durada_max:
            # Primer es talla i despres es jutja: un clip llarg amb pauses
            # llargues suspendria el `ratio_parla` per uns silencis que el tall
            # li treu igualment.
            millor = millor_finestra(senyal, segments, args.durada_min, args.durada_max)
            if millor is None:
                continue
            senyal, segments, met = millor
        else:
            met = mesurar_qualitat(senyal, segments)

        motiu = clip_acceptable(met, min_snr, args.min_ratio_parla)
        if motiu:
            continue
        met["puntuacio"] = puntuar(met)
        candidats.append((senyal, segments, met))

    if not candidats:
        return None, "cap clip supera el control de qualitat"

    candidats.sort(key=lambda c: c[2]["puntuacio"], reverse=True)
    senyal, usats = muntar_clip(candidats, args.durada_min, args.durada_max)
    if senyal is None:
        acumulat = sum(m["durada_s"] for m in usats)
        return None, f"nomes {acumulat:.1f}s de parla valida (calen {args.durada_min:.0f}s)"

    # Durada i ratio de parla es mesuren sobre el clip muntat (els silencis de
    # confort hi compten). SNR i transitoris, en canvi, es prenen del pitjor
    # tros d'origen: el silenci de confort es digital, potencia zero, i mesurat
    # sobre el clip muntat dispararia l'SNR a l'infinit encara que els trossos
    # vinguessin d'una gravacio sorollosa.
    met_final = mesurar_qualitat(senyal, vad.segments(senyal))
    met_final["snr_db"] = min(m["snr_db"] for m in usats)
    met_final["transitoris_per_min"] = max(m["transitoris_per_min"] for m in usats)
    met_final["puntuacio_validacio"] = puntuar(met_final)
    met_final["n_clips_font"] = len(usats)
    if met_final["puntuacio_validacio"] < args.min_puntuacio:
        return None, f"puntuacio {met_final['puntuacio_validacio']:.2f} < {args.min_puntuacio}"
    return (senyal, met_final), None


def nom_fitxer(perfil):
    """es_ES_<dataset>_<speaker>_<gender>_<age>.wav"""
    return (f"es_ES_{perfil['dataset']}_{_id_curt(perfil['speaker_id'])}"
            f"_{perfil['gender']}_{_sanejar(perfil['age'])}.wav")


# =========================================================================
# 9. Manifest
# =========================================================================

def escriure_manifest(output_dir, entrades, args):
    """Fusiona amb el manifest d'execucions anteriors (si n'hi ha): el banc es
    va ampliant amb crides successives (una per font, per genere...) i cada
    crida nomes coneix els seus locutors."""
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


# =========================================================================
# 10. Orquestracio
# =========================================================================

def _mostrar_seleccio(selector, perfils, quota, camp_clips):
    """Que sortiria si tots els locutors passessin el control de qualitat."""
    for _ in range(quota):
        clau = selector.seguent()
        if clau is None:
            break
        perfil = perfils[clau]
        selector.registrar(perfil["gender"])
        print(f"  [dry-run] {nom_fitxer(perfil):58s} clips={len(perfil[camp_clips])}")


def _construir_cv(args, vad, quota):
    if quota <= 0:
        return []
    print(f"\n== Common Voice ({args.idioma}/{args.split}) ==")
    perfils = perfilar_common_voice(
        args.idioma, args.split, args.token, args.cache_dir, args.accent_contains,
        args.clips_per_locutor, max_locutors=quota * args.factor_candidats,
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
        crus = [carregar_audio_cv(f, tars, index) for f in perfil["files"]]
        nous = _emetre(perfil, crus, vad, args)
        if nous:
            selector.registrar(perfil["gender"])
        resultats += nous
    for tar in tars.values():
        tar.close()
    return resultats


def _construir_voxpopuli(args, vad, quota):
    if quota <= 0:
        return []
    print(f"\n== VoxPopuli ({args.idioma}/{args.split_voxpopuli}) ==")
    perfils = perfilar_voxpopuli(
        args.idioma, args.split_voxpopuli, args.token, args.cache_dir,
        args.clips_per_locutor, max_locutors=quota * args.factor_candidats,
        max_files=args.max_files_voxpopuli,
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
        nous = _emetre(perfil, perfil["clips"], vad, args, min_snr=args.min_snr_voxpopuli)
        if nous:
            selector.registrar(perfil["gender"])
        resultats += nous
    return resultats


def _emetre(perfil, clips_crus, vad, args, min_snr=None):
    """Processa un locutor i, si passa, escriu el .wav. Llista buida si es descarta."""
    resultat, motiu = processar_locutor(perfil, clips_crus, vad, args, min_snr=min_snr)
    nom = nom_fitxer(perfil)
    if resultat is None:
        print(f"  [--] {nom:60s} descartat: {motiu}")
        return []

    senyal, met = resultat
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
        "qualitat": {k: met[k] for k in ("snr_db", "ratio_parla", "durada_parla_s",
                                         "ratio_clipping", "transitoris_per_min")},
        "normalitzacio": info,
        "font_real": perfil["dataset"] == "voxpopuli",
    }
    print(f"  [ok] {nom:60s} {info['durada_s']:5.1f}s  snr={met['snr_db']:5.1f}dB  "
          f"punt={met['puntuacio_validacio']:.2f}  clips={met['n_clips_font']}")
    return [(nom, entrada)]


def main():
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
                        help="Files maximes a llegir en streaming de VoxPopuli. El "
                             "streaming baixa el parquet de manera seqüencial, aixi "
                             "que aquest numero es, sobretot, un limit de temps de "
                             "descarrega (molt notable sense --token)")
    parser.add_argument("--durada-min", type=float, default=DURADA_MIN_S)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Repartiment de places entre fonts: la darrera s'endu el residu de la divisio.
    quotes = {f: args.n_locutors // len(args.sources) for f in args.sources}
    quotes[args.sources[-1]] += args.n_locutors - sum(quotes.values())

    entrades = {}
    if "common_voice" in args.sources:
        entrades.update(dict(_construir_cv(args, vad, quotes["common_voice"])))
    if "voxpopuli" in args.sources:
        entrades.update(dict(_construir_voxpopuli(args, vad, quotes["voxpopuli"])))

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

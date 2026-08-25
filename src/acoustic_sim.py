#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador acustic per al dataset sintetic.

OmniVoice nomes sintetitza veu: dona sempre un senyal net d'estudi, sense sala,
sense canal i sense ambient. Si entrenem Whisper nomes amb aixo, el model
s'especialitza en un so que no existeix fora d'un estudi. Aquest modul afegeix
les condicions reals de captacio que falten.

La cadena segueix l'ordre fisic en que passen les coses a la realitat:

    veu seca  ->  [SALA]  ->  [+ SOROLL]  ->  [CANAL]  ->  normalitzacio
                   RIR         SNR (dB)      filtre/codec

  1. SALA    la veu rebota a l'espai on es parla (convolucio amb una resposta
             impulsional generada amb pyroomacoustics).
  2. SOROLL  l'ambient ja es dins d'aquella sala, per tant se suma despres de la
             reverberacio, amb una relacio senyal/soroll controlada en dB.
  3. CANAL   el microfon i la transmissio capturen el conjunt, per tant els
             filtres i els codecs s'apliquen al final de tot.

Us tipic:

    from acoustic_sim import ENTORNS, mesclar_amb_entorn
    audio, meta = mesclar_amb_entorn(waveform, 24000, ENTORNS["carrer"], rng)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR_SOROLL = ROOT / "datasets/audios/entorns/soroll"
DIR_RIR = ROOT / "datasets/audios/entorns/rir"

SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Utilitats de senyal
# ---------------------------------------------------------------------------

def a_mono_float32(waveform) -> np.ndarray:
    """Accepta np.ndarray o torch.Tensor, mono o (1, N)/(N, 1), i torna 1-D float32."""
    if hasattr(waveform, "detach"):  # torch.Tensor sense importar torch aqui
        waveform = waveform.detach().cpu().numpy()
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x.squeeze()
        if x.ndim > 1:  # multicanal de veritat -> barregem a mono
            eix = int(np.argmin(x.shape))
            x = x.mean(axis=eix)
    return np.ascontiguousarray(x, dtype=np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def rms_activa(x: np.ndarray, sr: int, finestra_ms: float = 20.0,
               llindar_db: float = -30.0) -> float:
    """RMS nomes de les trames amb veu.

    Calcular la SNR sobre el senyal sencer es un error classic: els silencis
    inicials i finals baixen l'RMS global i fan que el soroll acabi sonant molt
    mes fort del que demanava la SNR. Aqui ens quedem nomes amb les trames que
    estan dins de `llindar_db` respecte de la trama mes forta, que es una
    aproximacio prou bona al nivell de veu activa.
    """
    n = max(1, int(sr * finestra_ms / 1000))
    if len(x) < n:
        return rms(x)
    tallat = x[: len(x) // n * n].reshape(-1, n)
    rms_trames = np.sqrt(np.mean(np.square(tallat, dtype=np.float64), axis=1))
    pic = float(rms_trames.max())
    if pic <= 0:
        return rms(x)
    actives = rms_trames[rms_trames >= pic * (10.0 ** (llindar_db / 20.0))]
    return float(actives.mean() if len(actives) else pic) + 1e-12


def normalitzar(x: np.ndarray, pic_objectiu_dbfs: float = -3.0) -> np.ndarray:
    """Escala el senyal a un pic concret. Garanteix que no hi ha clipping.

    No fem servir normalitzacio a RMS fix a proposit: volem que el dataset tingui
    diversitat de nivell (els entorns criden un `nivell_dbfs` aleatori dins d'un
    rang), perque un corpus on tots els fitxers sonen exactament igual de fort
    tampoc no s'assembla a la realitat.
    """
    pic = float(np.abs(x).max())
    if pic <= 0 or not np.isfinite(pic):
        return np.zeros_like(x)
    objectiu = 10.0 ** (pic_objectiu_dbfs / 20.0)
    return (x * (objectiu / pic)).astype(np.float32)


# ---------------------------------------------------------------------------
# Banc de soroll ambiental
# ---------------------------------------------------------------------------

_cache_soroll: dict[str, tuple[np.ndarray, int]] = {}


def carregar_ambient(nom: str, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Carrega (i memoritza) una pista d'ambient del banc. None si no hi es."""
    if nom in _cache_soroll:
        dades, sr_cache = _cache_soroll[nom]
        if sr_cache == sr:
            return dades

    ruta = DIR_SOROLL / f"{nom}.wav"
    if not ruta.exists():
        return None

    import soundfile as sf
    dades, sr_fitxer = sf.read(ruta, dtype="float32")
    if dades.ndim > 1:
        dades = dades.mean(axis=1)
    if sr_fitxer != sr:
        import soxr
        dades = soxr.resample(dades, sr_fitxer, sr, quality="VHQ").astype(np.float32)
    _cache_soroll[nom] = (dades, sr)
    return dades


def soroll_sintetic(n: int, sr: int, rng: np.random.Generator,
                    color: str = "rosa") -> np.ndarray:
    """Genera soroll procedural com a alternativa quan no hi ha banc descarregat.

    Es menys realista que un ambient gravat (li falten els esdeveniments no
    estacionaris: portes, clàxons, tos) pero no depen de cap descarrega, aixi que
    el pipeline mai no es queda bloquejat.
    """
    blanc = rng.standard_normal(n).astype(np.float32)
    if color == "blanc":
        senyal = blanc
    else:
        # Donem forma espectral al soroll blanc al domini frequencial: 1/f per al
        # rosa i 1/f^2 per al marro (mes greu, tipus remor de transit llunya).
        espectre = np.fft.rfft(blanc)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        f[0] = f[1] if len(f) > 1 else 1.0
        exponent = {"rosa": 1.0, "marro": 2.0}.get(color, 1.0)
        senyal = np.fft.irfft(espectre / (f ** (exponent / 2.0)), n=n).astype(np.float32)

    if color == "zumzeig":  # brunzit de xarxa electrica + harmonics
        t = np.arange(n, dtype=np.float32) / sr
        senyal = sum(
            (0.6 ** k) * np.sin(2 * np.pi * 50.0 * (k + 1) * t + rng.uniform(0, 6.28))
            for k in range(4)
        ).astype(np.float32) + 0.1 * blanc

    return (senyal / (np.abs(senyal).max() + 1e-9)).astype(np.float32)


def babble_des_de_veus(n: int, sr: int, rng: np.random.Generator,
                       dir_veus: Path, n_parlants: int = 8) -> np.ndarray | None:
    """Construeix murmuri superposant retalls d'audios de veu ja generats.

    Té un avantatge sobre DEMAND per al nostre cas: DEMAND es va gravar en
    entorns anglofons/francofons, mentre que aixo dona murmuri **en castella**,
    que es el que de debo hi hauria de fons en una redaccio o una roda de premsa
    espanyola. La fonetica del soroll de fons importa: un babble en l'idioma
    objectiu es molt mes confusor per al model, i per tant mes util per entrenar.
    """
    fitxers = sorted(Path(dir_veus).glob("*.wav"))
    if len(fitxers) < 2:
        return None

    import soundfile as sf
    mescla = np.zeros(n, dtype=np.float32)
    tria = rng.choice(len(fitxers), size=min(n_parlants, len(fitxers)), replace=False)
    for idx in tria:
        dades, sr_fitxer = sf.read(fitxers[int(idx)], dtype="float32")
        if dades.ndim > 1:
            dades = dades.mean(axis=1)
        if sr_fitxer != sr:
            import soxr
            dades = soxr.resample(dades, sr_fitxer, sr, quality="HQ").astype(np.float32)
        # Repetim fins a cobrir la durada i desplacem l'inici a l'atzar, perque
        # els parlants no comencin tots alhora.
        if len(dades) < n:
            dades = np.tile(dades, int(np.ceil(n / len(dades))))
        desfase = int(rng.integers(0, max(1, len(dades) - n + 1)))
        mescla += dades[desfase:desfase + n] * float(rng.uniform(0.5, 1.0))

    return (mescla / (np.abs(mescla).max() + 1e-9)).astype(np.float32)


def retall_soroll(pista: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Agafa un fragment de `n` mostres en una posicio aleatoria de la pista."""
    if len(pista) < n:
        pista = np.tile(pista, int(np.ceil(n / len(pista))))
    inici = int(rng.integers(0, len(pista) - n + 1))
    return pista[inici:inici + n]


# ---------------------------------------------------------------------------
# Banc de respostes impulsionals (RIR)
# ---------------------------------------------------------------------------

@dataclass
class Sala:
    """Descripcio d'un espac acustic per generar-ne RIRs amb el metode d'imatges."""
    id: str
    dimensions: tuple[float, float, float]  # metres
    rt60: float                             # parametre d'entrada a inverse_sabine (s)
    distancia_micro: float                  # metres entre font i microfon
    descripcio: str = ""
    rt60_real: float = 0.0                  # RT60 realment mesurat a les RIRs (s)


# El camp `rt60` es el que s'entrega a `pra.inverse_sabine`, i no coincideix amb
# el RT60 que despres es mesura: la formula de Sabine assumeix camp difus, cosa
# que no es compleix en sales molt allargades (el passadis se n'anava de 0.90 a
# 1.28 s) ni en sales molt petites (el despatx toca el limit fisic de Sabine i no
# pot reverberar tant com se li demana). Els valors d'aqui estan **calibrats
# contra mesura**: `rt60_real` es el que surt de debo, verificat amb el metode de
# Schroeder sobre les RIRs generades. Si canvies `dimensions`, recalibra.
SALES = {
    "plato_tv": Sala("plato_tv", (9.0, 7.0, 3.2), 0.300, 0.4,
                     "Plató de televisió: tractat acusticament, micro a prop",
                     rt60_real=0.29),
    "despatx": Sala("despatx", (4.5, 3.5, 2.7), 0.383, 0.6,
                    "Despatx petit o locutori improvisat",
                    rt60_real=0.28),
    "redaccio": Sala("redaccio", (12.0, 9.0, 3.0), 0.363, 0.7,
                     "Redacció oberta: sostre baix, molta superficie dura",
                     rt60_real=0.46),
    "sala_premsa": Sala("sala_premsa", (14.0, 10.0, 3.6), 0.541, 1.5,
                        "Sala de premsa: micro de faristol a distancia mitjana",
                        rt60_real=0.70),
    "sala_actes": Sala("sala_actes", (25.0, 18.0, 8.0), 1.170, 3.0,
                       "Sala d'actes gran: cua de reverberacio llarga",
                       rt60_real=1.31),
    "passadis": Sala("passadis", (16.0, 2.4, 3.0), 0.340, 2.0,
                     "Passadis estret: reflexions laterals molt marcades",
                     rt60_real=0.94),
}

VARIANTS_PER_SALA = 6  # rooms lleugerament diferents per no repetir sempre la mateixa


def _generar_rir(sala: Sala, llavor: int, sr: int) -> np.ndarray:
    """Genera una RIR amb el metode de les fonts imatge (pyroomacoustics)."""
    import pyroomacoustics as pra

    rng = np.random.default_rng(llavor)
    # Perturbem la sala perque cada variant sigui un espai diferent de debo i no
    # el mateix filtre repetit: mides +-10%, RT60 +-15%.
    dims = np.array(sala.dimensions) * rng.uniform(0.90, 1.10, size=3)
    rt60 = float(np.clip(sala.rt60 * rng.uniform(0.85, 1.15), 0.12, 3.0))

    try:
        absorcio, ordre_max = pra.inverse_sabine(rt60, dims.tolist())
    except ValueError:
        # RT60 impossible per a aquest volum (limit de Sabine): agafem el mes
        # curt que la sala permet fisicament.
        absorcio, ordre_max = 0.85, 8
    ordre_max = int(min(ordre_max, 17))  # sostre per no disparar el cost

    room = pra.ShoeBox(dims.tolist(), fs=sr, materials=pra.Material(absorcio),
                       max_order=ordre_max)

    # Col·loquem la font en una posicio raonable i el microfon a la distancia
    # caracteristica de la sala, en una direccio aleatoria del pla horitzontal.
    marge = 0.8
    font = np.array([
        rng.uniform(marge, dims[0] - marge),
        rng.uniform(marge, dims[1] - marge),
        rng.uniform(1.2, 1.8),  # alçada de boca d'una persona
    ])
    angle = rng.uniform(0, 2 * np.pi)
    dist = sala.distancia_micro * rng.uniform(0.8, 1.25)
    micro = np.array([
        np.clip(font[0] + dist * np.cos(angle), 0.3, dims[0] - 0.3),
        np.clip(font[1] + dist * np.sin(angle), 0.3, dims[1] - 0.3),
        np.clip(font[2] + rng.uniform(-0.3, 0.3), 0.3, dims[2] - 0.3),
    ])

    room.add_source(font.tolist())
    room.add_microphone(micro.tolist())
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)

    pic = float(np.abs(rir).max())
    return rir / pic if pic > 0 else rir


def construir_banc_rir(sales=None, variants: int = VARIANTS_PER_SALA,
                       sr: int = SAMPLE_RATE, dir_rir: Path = DIR_RIR,
                       overwrite: bool = False, verbose: bool = True) -> dict[str, list[Path]]:
    """Pre-genera i desa el banc de RIRs. Es rapid (~20 ms per RIR) pero el
    guardem a disc perque el dataset sigui auditable: si algu vol saber com
    sonava la sala d'un audio concret, pot escoltar la RIR."""
    import soundfile as sf

    sales = sales or SALES
    dir_rir = Path(dir_rir)
    dir_rir.mkdir(parents=True, exist_ok=True)

    banc: dict[str, list[Path]] = {}
    for nom, sala in sales.items():
        rutes = []
        for i in range(variants):
            ruta = dir_rir / f"{nom}_{i:02d}.wav"
            if not ruta.exists() or overwrite:
                # Llavor derivada per sha256, no per `hash()`: el hash de Python
                # esta randomitzat per proces (PYTHONHASHSEED), aixi que fer-lo
                # servir donava un banc de RIRs diferent a cada execucio.
                rir = _generar_rir(sala, llavor=llavor_estable(nom, i) % (2**31), sr=sr)
                # PCM_24 i no FLOAT: en WAV de coma flotant libsndfile escriu un
                # chunk PEAK que inclou un timestamp, i aixo fa que dues
                # generacions identiques donin fitxers amb bytes diferents. Amb
                # 24 bits el banc es verificable amb checksums i el marge dinamic
                # (~144 dB) sobra per a una RIR normalitzada a +-1.
                sf.write(ruta, rir, sr, subtype="PCM_24")
            rutes.append(ruta)
        banc[nom] = rutes
        if verbose:
            dur = sf.info(rutes[0]).duration
            print(f"  {nom:14s} {variants} variants  (RT60 ~{sala.rt60_real:.2f}s, "
                  f"RIR de {dur:.2f}s)")
    return banc


_cache_rir: dict[Path, np.ndarray] = {}


def carregar_rir(ruta: Path) -> np.ndarray:
    ruta = Path(ruta)
    if ruta not in _cache_rir:
        import soundfile as sf
        dades, _ = sf.read(ruta, dtype="float32")
        if dades.ndim > 1:
            dades = dades.mean(axis=1)
        _cache_rir[ruta] = dades
    return _cache_rir[ruta]


def aplicar_sala(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convoluciona la veu amb la RIR conservant alineacio temporal i nivell.

    Dos detalls que importen:
      - Retallem el retard fins al pic directe de la RIR. Si no, tot l'audio
        queda desplaçat uns quants ms respecte de la transcripcio.
      - Reajustem el guany perque l'RMS es mantingui. La convolucio pot canviar
        l'energia molt, i si no ho compensem el calcul de SNR posterior deixa de
        voler dir res.
    """
    from scipy.signal import fftconvolve

    rms_previ = rms(x)
    retard = int(np.argmax(np.abs(rir)))
    humit = fftconvolve(x, rir)[retard:retard + len(x)]
    if len(humit) < len(x):
        humit = np.pad(humit, (0, len(x) - len(humit)))
    humit = humit * (rms_previ / rms(humit))
    return humit.astype(np.float32)


# ---------------------------------------------------------------------------
# Canals: microfon + transmissio
# ---------------------------------------------------------------------------

def _cadena_canal(canal: str, rng: np.random.Generator):
    """Construeix la cadena d'efectes de pedalboard per a un canal.

    Fem servir pedalboard perque porta codecs de veritat: `GSMFullRateCompressor`
    aplica el codec GSM 06.10 real, que es literalment el que degrada una trucada
    de telefon. Imitar-ho amb filtres nomes s'hi assembla de lluny.
    """
    import pedalboard as pb

    if canal in (None, "net"):
        return None

    if canal == "telefon":
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(280, 340))),
            pb.LowpassFilter(cutoff_frequency_hz=float(rng.uniform(3200, 3600))),
            pb.Distortion(drive_db=float(rng.uniform(3, 9))),
            pb.GSMFullRateCompressor(),
        ])

    if canal == "mobil_voip":
        # Banda ampla a proposit (~200 Hz - 7 kHz): un mobil actual fa servir
        # AMR-WB o Opus, no la banda estreta del telefon fix. El contrast amb el
        # canal "telefon" es justament el que dona varietat al dataset.
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(150, 250))),
            pb.LowpassFilter(cutoff_frequency_hz=float(rng.uniform(6000, 7500))),
            pb.Compressor(threshold_db=-20, ratio=4.0),
            pb.MP3Compressor(vbr_quality=float(rng.uniform(7.0, 9.0))),
        ])

    if canal == "walkie":
        # Filtres en cascada: un sol biquad (12 dB/oct) deixava passar fins a
        # gairebe 5 kHz i no s'assemblava a una radio. Doblats donen 24 dB/oct,
        # que si que retalla de debo a la banda 400-3000 Hz d'un equip de radio.
        hp = float(rng.uniform(380, 460))
        lp = float(rng.uniform(2800, 3200))
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=hp),
            pb.HighpassFilter(cutoff_frequency_hz=hp),
            pb.LowpassFilter(cutoff_frequency_hz=lp),
            pb.LowpassFilter(cutoff_frequency_hz=lp),
            pb.Distortion(drive_db=float(rng.uniform(10, 18))),
            pb.Bitcrush(bit_depth=float(rng.uniform(7, 10))),
            pb.Compressor(threshold_db=-24, ratio=8.0),
        ])

    if canal == "micro_corbata":
        # Talla els greus (soroll de manipulacio i roba) i perd una mica d'aguts.
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(150, 220))),
            pb.LowpassFilter(cutoff_frequency_hz=float(rng.uniform(8000, 10000))),
            pb.PeakFilter(cutoff_frequency_hz=float(rng.uniform(2500, 4000)),
                          gain_db=float(rng.uniform(1.5, 3.5)), q=1.2),
            pb.Compressor(threshold_db=-18, ratio=3.0),
        ])

    if canal == "micro_faristol":
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(90, 140))),
            pb.PeakFilter(cutoff_frequency_hz=float(rng.uniform(3000, 5000)),
                          gain_db=float(rng.uniform(1.0, 3.0)), q=0.9),
        ])

    if canal == "estudi":
        # Cadena d'estudi: nomes neteja de subgreus i una mica de compressio.
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(60, 90))),
            pb.Compressor(threshold_db=-16, ratio=2.5),
        ])

    raise ValueError(f"Canal desconegut: {canal!r}")


# ---------------------------------------------------------------------------
# Matriu d'entorns
# ---------------------------------------------------------------------------

@dataclass
class Entorn:
    id: str
    descripcio: str
    pes: float = 1.0                      # probabilitat relativa dins del dataset
    sala: str | None = None               # clau de SALES
    ambient: str | None = None            # fitxer del banc de soroll
    snr_db: tuple[float, float] | None = None
    canal: str | None = None
    nivell_dbfs: tuple[float, float] = (-12.0, -3.0)
    styles: tuple[str, ...] = field(default_factory=tuple)  # afinitat amb el style


# `styles` fa servir subcadenes en minuscules que es busquen dins del camp
# `style` de cada frase, per no dependre del text literal exacte del prompt.
ENTORNS: dict[str, Entorn] = {
    "estudi_net": Entorn(
        "estudi_net", "Estudi net, sense sala ni ambient: la condicio de referencia",
        pes=2.0, sala=None, ambient=None, snr_db=None, canal="estudi",
        styles=("titular", "noticia breve"),
    ),
    "plato_tv": Entorn(
        "plato_tv", "Plató: sala tractada, micro de faristol, ambient gairebe nul",
        pes=2.0, sala="plato_tv", ambient="redaccio", snr_db=(28.0, 38.0),
        canal="micro_faristol",
        styles=("entradilla", "plató", "plato", "presentador"),
    ),
    "redaccio": Entorn(
        "redaccio", "Redacció: teclats i murmuri de fons, micro de corbata",
        pes=1.5, sala="redaccio", ambient="redaccio", snr_db=(14.0, 24.0),
        canal="micro_corbata",
        styles=("noticia breve", "titular"),
    ),
    "sala_premsa": Entorn(
        "sala_premsa", "Roda de premsa: reverberacio de sala i murmuri contingut",
        pes=1.5, sala="sala_premsa", ambient="sala_premsa", snr_db=(16.0, 26.0),
        canal="micro_faristol",
        styles=("rueda de prensa", "declaraciones", "debate"),
    ),
    "sala_actes": Entorn(
        "sala_actes", "Sala d'actes gran: cua de reverberacio llarga, micro llunya",
        pes=0.8, sala="sala_actes", ambient="sala_premsa", snr_db=(20.0, 30.0),
        canal="micro_faristol",
        styles=("rueda de prensa", "declaraciones", "debate"),
    ),
    "carrer": Entorn(
        "carrer", "Crònica al carrer: transit, gens de sala, micro de reporter",
        pes=1.5, sala=None, ambient="carrer", snr_db=(8.0, 18.0),
        canal="micro_corbata",
        styles=("corresponsal", "crónica", "cronica"),
    ),
    "exterior_gent": Entorn(
        "exterior_gent", "Exterior amb public: plaça concorreguda",
        pes=1.0, sala=None, ambient="placa", snr_db=(10.0, 20.0),
        canal="micro_corbata",
        styles=("corresponsal", "crónica", "cronica"),
    ),
    "telefon": Entorn(
        "telefon", "Connexio telefonica: banda 300-3400 Hz i codec GSM",
        pes=1.2, sala="despatx", ambient="redaccio", snr_db=(12.0, 22.0),
        canal="telefon",
        styles=("corresponsal", "crónica", "cronica", "declaraciones"),
    ),
    "connexio_movil": Entorn(
        "connexio_movil", "Directe des del mobil: compressio VoIP i soroll d'estacio",
        pes=1.0, sala=None, ambient="estacio", snr_db=(10.0, 20.0),
        canal="mobil_voip",
        styles=("corresponsal", "crónica", "cronica", "noticia breve"),
    ),
    "cafeteria": Entorn(
        "cafeteria", "Declaracions en un local: murmuri dens i molt proper",
        pes=0.8, sala="despatx", ambient="cafeteria", snr_db=(6.0, 15.0),
        canal="micro_corbata",
        styles=("declaraciones", "debate"),
    ),
}


def entorns_per_style(style: str, entorns: dict[str, Entorn] = None) -> list[Entorn]:
    """Entorns coherents amb l'estil de la frase.

    Una cronica de corresponsal hauria de sonar a carrer o a telefon, no a plató.
    Si l'estil no encaixa amb cap entorn concret, tornem tots els entorns per no
    deixar mai la frase sense opcions.
    """
    entorns = entorns or ENTORNS
    style_norm = (style or "").lower()
    coincidents = [e for e in entorns.values()
                   if any(clau in style_norm for clau in e.styles)]
    return coincidents or list(entorns.values())


# ---------------------------------------------------------------------------
# Mesclador principal
# ---------------------------------------------------------------------------

def mesclar_amb_entorn(waveform, sr: int, entorn: Entorn | str,
                       rng: np.random.Generator | None = None,
                       dir_veus_babble: Path | None = None,
                       ) -> tuple[np.ndarray, dict]:
    """Aplica sala + soroll + canal a un audio i retorna (audio, metadades).

    Les metadades porten els valors concrets que s'han fet servir (SNR exacta,
    quina RIR, quin canal) perque el manifest els pugui registrar i el dataset
    sigui reproduible i auditable.
    """
    if isinstance(entorn, str):
        entorn = ENTORNS[entorn]
    if rng is None:
        rng = np.random.default_rng()

    x = a_mono_float32(waveform)
    meta: dict = {"entorn_id": entorn.id, "sala": None, "rir_path": None,
                  "ambient": None, "snr_db": None, "canal": entorn.canal,
                  "nivell_dbfs": None, "font_soroll": None}

    if x.size == 0:
        return x, meta

    # --- 1. Sala -----------------------------------------------------------
    if entorn.sala:
        rutes = sorted(DIR_RIR.glob(f"{entorn.sala}_*.wav"))
        if not rutes:  # el banc encara no s'ha construit: el generem al vol
            construir_banc_rir({entorn.sala: SALES[entorn.sala]}, sr=sr, verbose=False)
            rutes = sorted(DIR_RIR.glob(f"{entorn.sala}_*.wav"))
        if rutes:
            ruta = rutes[int(rng.integers(0, len(rutes)))]
            x = aplicar_sala(x, carregar_rir(ruta))
            meta["sala"] = entorn.sala
            meta["rir_path"] = str(ruta.relative_to(ROOT))

    # --- 2. Soroll additiu a SNR controlada --------------------------------
    if entorn.ambient and entorn.snr_db:
        pista = carregar_ambient(entorn.ambient, sr)
        font = "demand"
        if pista is None and dir_veus_babble is not None:
            pista = babble_des_de_veus(len(x), sr, rng, dir_veus_babble)
            font = "babble_tts"
        if pista is None:
            color = "marro" if entorn.ambient in ("carrer", "estacio") else "rosa"
            pista = soroll_sintetic(len(x), sr, rng, color=color)
            font = f"sintetic_{color}"

        soroll = retall_soroll(pista, len(x), rng)
        snr = float(rng.uniform(*entorn.snr_db))

        # Guany del soroll perque la SNR resultant sigui exactament la demanada,
        # mesurant la veu nomes sobre les trames actives.
        guany = rms_activa(x, sr) / (rms(soroll) * (10.0 ** (snr / 20.0)))
        x = x + (soroll * guany).astype(np.float32)

        meta["ambient"] = entorn.ambient
        meta["snr_db"] = round(snr, 2)
        meta["font_soroll"] = font

    # --- 3. Canal (microfon + transmissio) ---------------------------------
    cadena = _cadena_canal(entorn.canal, rng)
    if cadena is not None:
        # Deixem marge abans del canal: els codecs i les distorsions saturen si
        # els entra un senyal massa a prop de 0 dBFS.
        x = normalitzar(x, -6.0)
        x = np.ascontiguousarray(cadena(x, float(sr)).squeeze(), dtype=np.float32)

    # --- 4. Nivell final ---------------------------------------------------
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    nivell = float(rng.uniform(*entorn.nivell_dbfs))
    x = normalitzar(x, nivell)
    meta["nivell_dbfs"] = round(nivell, 2)

    return x, meta


# ---------------------------------------------------------------------------
# Suport per a mostreig equilibrat
# ---------------------------------------------------------------------------

def llavor_estable(*parts) -> int:
    """Llavor derivada del contingut, no de l'ordre d'iteracio.

    Aixi cada frase rep sempre les mateixes condicions encara que es reprengui
    una generacio a mitges o es reordeni el dataset.
    """
    clau = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(clau).digest()[:8], "big")


def mostreig_round_robin(items: list, n: int, rng: np.random.Generator) -> list:
    """Reparteix `items` en `n` posicions de manera equitativa.

    El mostreig purament aleatori sobre pocs elements deixa desequilibris grossos
    per pura sort (una veu que surt el triple que una altra). Aqui recorrem la
    llista sencera i la remenem a cada volta: cap element no pot sortir mes d'un
    cop mes que qualsevol altre, pero l'aparellament no queda fixat en un patro.
    """
    if not items:
        raise ValueError("Llista d'items buida")
    sortida = []
    while len(sortida) < n:
        volta = list(items)
        rng.shuffle(volta)
        sortida.extend(volta)
    return sortida[:n]


def expandir_per_pes(entorns: list[Entorn], resolucio: int = 10) -> list[Entorn]:
    """Repeteix cada entorn segons el seu pes, per fer-lo servir amb round-robin."""
    expandit = []
    for e in entorns:
        expandit.extend([e] * max(1, int(round(e.pes * resolucio))))
    return expandit

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulador acustic: sales (RIR), soroll a SNR controlada, canals i nivell, amb mostreig equilibrat."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR_SOROLL = ROOT / "data/comu/entorns/soroll"
DIR_RIR = ROOT / "data/comu/entorns/rir"

SAMPLE_RATE = 24000


def a_mono_float32(waveform) -> np.ndarray:
    """Accepta np.ndarray o torch.Tensor, mono o (1, N)/(N, 1), i torna 1-D float32."""
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x.squeeze()
        if x.ndim > 1:
            eix = int(np.argmin(x.shape))
            x = x.mean(axis=eix)
    return np.ascontiguousarray(x, dtype=np.float32)


def rms(x: np.ndarray) -> float:
    """Valor eficac d'un senyal."""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def rms_activa(x: np.ndarray, sr: int, finestra_ms: float = 20.0,
               llindar_db: float = -30.0) -> float:
    """RMS nomes de les trames amb veu."""
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
    """Escala el senyal a un pic concret."""
    pic = float(np.abs(x).max())
    if pic <= 0 or not np.isfinite(pic):
        return np.zeros_like(x)
    objectiu = 10.0 ** (pic_objectiu_dbfs / 20.0)
    return (x * (objectiu / pic)).astype(np.float32)


_cache_soroll: dict[str, tuple[np.ndarray, int]] = {}


def carregar_ambient(nom: str, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Carrega (i memoritza) una pista d'ambient del banc."""
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
    """Genera soroll procedural com a alternativa quan no hi ha banc descarregat."""
    blanc = rng.standard_normal(n).astype(np.float32)
    if color == "blanc":
        senyal = blanc
    else:
        espectre = np.fft.rfft(blanc)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        f[0] = f[1] if len(f) > 1 else 1.0
        exponent = {"rosa": 1.0, "marro": 2.0}.get(color, 1.0)
        senyal = np.fft.irfft(espectre / (f ** (exponent / 2.0)), n=n).astype(np.float32)

    if color == "zumzeig":
        t = np.arange(n, dtype=np.float32) / sr
        senyal = sum(
            (0.6 ** k) * np.sin(2 * np.pi * 50.0 * (k + 1) * t + rng.uniform(0, 6.28))
            for k in range(4)
        ).astype(np.float32) + 0.1 * blanc

    return (senyal / (np.abs(senyal).max() + 1e-9)).astype(np.float32)


def babble_des_de_veus(n: int, sr: int, rng: np.random.Generator,
                       dir_veus: Path, n_parlants: int = 8) -> np.ndarray | None:
    """Construeix murmuri superposant retalls d'audios de veu ja generats."""
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


@dataclass
class Sala:
    """Descripcio d'un espac acustic per generar-ne RIRs amb el metode d'imatges."""
    id: str
    dimensions: tuple[float, float, float]
    rt60: float
    distancia_micro: float
    descripcio: str = ""
    rt60_real: float = 0.0


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

VARIANTS_PER_SALA = 6


def _generar_rir(sala: Sala, llavor: int, sr: int) -> np.ndarray:
    """Genera una RIR amb el metode de les fonts imatge (pyroomacoustics)."""
    import pyroomacoustics as pra

    rng = np.random.default_rng(llavor)
    dims = np.array(sala.dimensions) * rng.uniform(0.90, 1.10, size=3)
    rt60 = float(np.clip(sala.rt60 * rng.uniform(0.85, 1.15), 0.12, 3.0))

    try:
        absorcio, ordre_max = pra.inverse_sabine(rt60, dims.tolist())
    except ValueError:
        absorcio, ordre_max = 0.85, 8
    ordre_max = int(min(ordre_max, 17))

    room = pra.ShoeBox(dims.tolist(), fs=sr, materials=pra.Material(absorcio),
                       max_order=ordre_max)

    marge = 0.8
    font = np.array([
        rng.uniform(marge, dims[0] - marge),
        rng.uniform(marge, dims[1] - marge),
        rng.uniform(1.2, 1.8),
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
    """Pre-genera i desa el banc de RIRs."""
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
                rir = _generar_rir(sala, llavor=llavor_estable(nom, i) % (2**31), sr=sr)
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
    """Carrega una RIR (amb cache)."""
    ruta = Path(ruta)
    if ruta not in _cache_rir:
        import soundfile as sf
        dades, _ = sf.read(ruta, dtype="float32")
        if dades.ndim > 1:
            dades = dades.mean(axis=1)
        _cache_rir[ruta] = dades
    return _cache_rir[ruta]


def aplicar_sala(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convoluciona la veu amb la RIR conservant alineacio temporal i nivell."""
    from scipy.signal import fftconvolve

    rms_previ = rms(x)
    retard = int(np.argmax(np.abs(rir)))
    humit = fftconvolve(x, rir)[retard:retard + len(x)]
    if len(humit) < len(x):
        humit = np.pad(humit, (0, len(x) - len(humit)))
    humit = humit * (rms_previ / rms(humit))
    return humit.astype(np.float32)


def _cadena_canal(canal: str, rng: np.random.Generator):
    """Construeix la cadena d'efectes de pedalboard per a un canal."""
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
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(150, 250))),
            pb.LowpassFilter(cutoff_frequency_hz=float(rng.uniform(6000, 7500))),
            pb.Compressor(threshold_db=-20, ratio=4.0),
            pb.MP3Compressor(vbr_quality=float(rng.uniform(7.0, 9.0))),
        ])

    if canal == "walkie":
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
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=float(rng.uniform(60, 90))),
            pb.Compressor(threshold_db=-16, ratio=2.5),
        ])

    raise ValueError(f"Canal desconegut: {canal!r}")


@dataclass
class Entorn:
    """Un entorn acustic: sala, soroll a una SNR, canal i nivell, i els estils de frase que hi encaixen."""
    id: str
    descripcio: str
    pes: float = 1.0
    sala: str | None = None
    ambient: str | None = None
    snr_db: tuple[float, float] | None = None
    canal: str | None = None
    nivell_dbfs: tuple[float, float] = (-12.0, -3.0)
    styles: tuple[str, ...] = field(default_factory=tuple)


ENTORNS: dict[str, Entorn] = {
    "font_real": Entorn(
        "font_real", "Veu de referencia ja gravada en condicions reals (p.ex. "
        "corpus radiofonic/parlamentari): no s'hi afegeix sala ni soroll ni canal "
        "per no duplicar-los sobre l'ambient que ja porta la clonacio -- nomes "
        "es normalitza el nivell",
        pes=0.0, sala=None, ambient=None, snr_db=None, canal=None,
        styles=(),
    ),
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
    """Entorns coherents amb l'estil de la frase."""
    entorns = entorns or ENTORNS
    style_norm = (style or "").lower()
    coincidents = [e for e in entorns.values()
                   if any(clau in style_norm for clau in e.styles)]
    return coincidents or list(entorns.values())


def mesclar_amb_entorn(waveform, sr: int, entorn: Entorn | str,
                       rng: np.random.Generator | None = None,
                       dir_veus_babble: Path | None = None,
                       ) -> tuple[np.ndarray, dict]:
    """Aplica sala + soroll + canal a un audio i retorna (audio, metadades)."""
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

    if entorn.sala:
        rutes = sorted(DIR_RIR.glob(f"{entorn.sala}_*.wav"))
        if not rutes:
            construir_banc_rir({entorn.sala: SALES[entorn.sala]}, sr=sr, verbose=False)
            rutes = sorted(DIR_RIR.glob(f"{entorn.sala}_*.wav"))
        if rutes:
            ruta = rutes[int(rng.integers(0, len(rutes)))]
            x = aplicar_sala(x, carregar_rir(ruta))
            meta["sala"] = entorn.sala
            meta["rir_path"] = str(ruta.relative_to(ROOT))

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

        guany = rms_activa(x, sr) / (rms(soroll) * (10.0 ** (snr / 20.0)))
        x = x + (soroll * guany).astype(np.float32)

        meta["ambient"] = entorn.ambient
        meta["snr_db"] = round(snr, 2)
        meta["font_soroll"] = font

    cadena = _cadena_canal(entorn.canal, rng)
    if cadena is not None:
        x = normalitzar(x, -6.0)
        x = np.ascontiguousarray(cadena(x, float(sr)).squeeze(), dtype=np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    nivell = float(rng.uniform(*entorn.nivell_dbfs))
    x = normalitzar(x, nivell)
    meta["nivell_dbfs"] = round(nivell, 2)

    return x, meta


def llavor_estable(*parts) -> int:
    """Llavor derivada del contingut, no de l'ordre d'iteracio."""
    clau = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(clau).digest()[:8], "big")


def mostreig_round_robin(items: list, n: int, rng: np.random.Generator) -> list:
    """Reparteix `items` en `n` posicions de manera equitativa."""
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baixa un tros d'àudio de YouTube i el deixa llest per a Whisper."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yt_dlp

AQUI = Path(__file__).resolve().parent
SORTIDA_DEFECTE = AQUI / "referencia_aranes.wav"


def baixar_audio(url: str, dest_dir: Path, inici: float, durada: float) -> Path:
    """Baixa el millor àudio del tram [inici, inici+durada) i torna el fitxer cru."""
    opcions = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "cru.%(ext)s"),
        "download_ranges": yt_dlp.utils.download_range_func(None, [(inici, inici + durada)]),
        "force_keyframes_at_cuts": True,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opcions) as ydl:
        info = ydl.extract_info(url, download=True)

    print(f"  títol: {info.get('title')}")
    print(f"  durada del vídeo: {info.get('duration')} s | canal: {info.get('uploader')}")

    fitxers = [f for f in dest_dir.iterdir() if f.stem == "cru"]
    if not fitxers:
        raise SystemExit("yt-dlp no ha deixat cap fitxer; prova-ho amb --verbose")
    return fitxers[0]


def convertir(entrada: Path, sortida: Path, sr: int, durada: float) -> None:
    """A mono, `sr` Hz, PCM 16 bits."""
    ordre = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(entrada),
        "-t", str(durada),
        "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
        str(sortida),
    ]
    subprocess.run(ordre, check=True)


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", help="URL del vídeo de YouTube")
    p.add_argument("--sortida", type=Path, default=SORTIDA_DEFECTE)
    p.add_argument("--inici", type=float, default=0.0, help="segon on comença el tall")
    p.add_argument("--durada", type=float, default=60.0, help="segons a baixar (defecte: 60)")
    p.add_argument("--sr", type=int, default=16000, help="freqüència de mostreig (defecte: 16000)")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        print("Falta ffmpeg al PATH.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Baixant {args.durada:.0f} s des del segon {args.inici:.0f}...")
        cru = baixar_audio(args.url, Path(tmp), args.inici, args.durada)
        print(f"  àudio cru: {cru.name} ({cru.stat().st_size / 1e6:.1f} MB)")
        args.sortida.parent.mkdir(parents=True, exist_ok=True)
        convertir(cru, args.sortida, args.sr, args.durada)

    import soundfile as sf
    info = sf.info(str(args.sortida))
    print(f"\n{args.sortida}")
    print(f"  {info.duration:.1f} s | {info.samplerate} Hz | {info.channels} canal(s) | {info.subtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

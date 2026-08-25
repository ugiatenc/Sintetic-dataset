#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construeix el banc de sorolls ambientals a partir del dataset DEMAND.

DEMAND (Diverse Environments Multichannel Acoustic Noise Database) son gravacions
reals d'ambients de 5 minuts. Baixem el mirror de HuggingFace en comptes de Zenodo
perque Zenodo dona timeouts i ~8 KB/s, mentre que el CDN de HF va a ~15 MB/s.

El mirror serveix cada ambient tallat en 75 segments de 4 s a 16 kHz. Aqui els
tornem a unir en un unic .wav continu per ambient (5 min) i el remostregem a la
frequencia del pipeline, de manera que despres es puguin agafar retalls aleatoris
de qualsevol durada sense haver de gestionar fitxers solts.

Nota sobre l'amplada de banda: el mirror es de 16 kHz, aixi que el soroll no te
contingut per sobre de 8 kHz. Es irrellevant per l'objectiu del projecte perque
Whisper treballa a 16 kHz i descarta aquesta banda igualment.

Llicencia: DEMAND es CC BY 4.0 (Thiemann, Ito & Vincent, 2013). Cal citar-la si
es publica el dataset resultant.

Exemple:
    python3 src/build_noise_bank.py
    python3 src/build_noise_bank.py --ambients redaccio carrer --overwrite
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

REPO_ID = "SPARCO-project/benchmark_DEMAND_noise"
DEFAULT_OUTPUT_DIR = Path("datasets/audios/entorns/soroll")
DEFAULT_SAMPLE_RATE = 24000

# Mapeig entre els ambients de DEMAND i els escenaris periodistics que ens
# interessen. La clau es el nom que faran servir els entorns del pipeline.
#   split -> subcarpeta del mirror on viu aquell ambient
AMBIENTS = {
    "redaccio":     {"demand": "OOFFICE",  "split": "scorer_val",
                     "descripcio": "Oficina: teclats, murmuri de fons, aire condicionat"},
    "carrer":       {"demand": "STRAFFIC", "split": "benchmark_test",
                     "descripcio": "Transit de carrer: cotxes, clàxons, motors"},
    "placa":        {"demand": "SPSQUARE", "split": "feature_scorer_train",
                     "descripcio": "Plaça publica: gent, passes, exteriors oberts"},
    "cafeteria":    {"demand": "PCAFETER", "split": "benchmark_test",
                     "descripcio": "Cafeteria: murmuri dens, gots, cadires"},
    "sala_premsa":  {"demand": "OMEETING", "split": "feature_scorer_train",
                     "descripcio": "Sala de reunions: murmuri contingut, papers"},
    "estacio":      {"demand": "PSTATION", "split": "feature_scorer_train",
                     "descripcio": "Estacio de tren: megafonia llunyana, gentada"},
}

CROSSFADE_MS = 20  # per dissimular el tall entre segments consecutius


def descarregar_ambient(nom, spec, cache_dir=None):
    """Baixa els 75 segments d'un ambient i retorna la llista de fitxers ordenada."""
    from huggingface_hub import snapshot_download

    patro = f"data/{spec['split']}/{spec['demand']}/*.wav"
    ruta = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[patro],
        cache_dir=cache_dir,
    )
    segments = sorted(Path(ruta).glob(f"data/{spec['split']}/{spec['demand']}/*.wav"))
    if not segments:
        raise RuntimeError(f"No s'ha baixat cap segment per a {nom} ({patro})")
    return segments


def unir_segments(segments, crossfade_ms=CROSSFADE_MS):
    """Concatena els segments amb un crossfade curt i retorna (senyal, sample_rate).

    Els segments son trossos consecutius de la mateixa gravacio, pero fem un
    crossfade igualment perque un tall sec introduiria un clic periodic cada 4 s
    que el model podria aprendre com a artefacte.
    """
    trossos, sr = [], None
    for seg in segments:
        dades, sr_seg = sf.read(seg, dtype="float32", always_2d=False)
        if dades.ndim > 1:
            dades = dades.mean(axis=1)
        if sr is None:
            sr = sr_seg
        elif sr_seg != sr:
            raise ValueError(f"Sample rate inconsistent a {seg}: {sr_seg} != {sr}")
        trossos.append(dades)

    n_fade = max(1, int(sr * crossfade_ms / 1000))
    fade_out = np.linspace(1.0, 0.0, n_fade, dtype="float32")
    fade_in = 1.0 - fade_out

    sortida = trossos[0]
    for tros in trossos[1:]:
        if len(sortida) < n_fade or len(tros) < n_fade:
            sortida = np.concatenate([sortida, tros])
            continue
        solapament = sortida[-n_fade:] * fade_out + tros[:n_fade] * fade_in
        sortida = np.concatenate([sortida[:-n_fade], solapament, tros[n_fade:]])

    return sortida, sr


def construir(nom, spec, output_dir, sample_rate, overwrite, cache_dir=None):
    desti = Path(output_dir) / f"{nom}.wav"
    if desti.exists() and not overwrite:
        info = sf.info(desti)
        print(f"  [skip] {nom:12s} ja existeix ({info.duration/60:.1f} min @ {info.samplerate} Hz)")
        return desti

    segments = descarregar_ambient(nom, spec, cache_dir=cache_dir)
    senyal, sr_origen = unir_segments(segments)

    if sr_origen != sample_rate:
        senyal = soxr.resample(senyal, sr_origen, sample_rate, quality="VHQ")

    # Normalitzem a un pic conegut perque despres el mesclador nomes hagi de
    # raonar en termes de SNR i no d'on venia el fitxer.
    pic = float(np.abs(senyal).max())
    if pic > 0:
        senyal = senyal / pic * 0.95

    desti.parent.mkdir(parents=True, exist_ok=True)
    sf.write(desti, senyal, sample_rate, subtype="PCM_16")
    print(f"  [ok]   {nom:12s} {len(segments):3d} segments -> {len(senyal)/sample_rate/60:.1f} min "
          f"@ {sample_rate} Hz  ({desti.stat().st_size/1e6:.1f} MB)  <- DEMAND/{spec['demand']}")
    return desti


def escriure_atribucio(output_dir, ambients):
    """Deixa constancia de la llicencia al costat dels fitxers."""
    linies = [
        "# Banc de sorolls ambientals",
        "",
        "Generat automaticament per `src/build_noise_bank.py`. NO editar a ma.",
        "",
        "## Font",
        "",
        "DEMAND: a collection of multi-channel recordings of acoustic noise in diverse",
        "environments. Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent (2013).",
        "Zenodo: https://doi.org/10.5281/zenodo.1227121 — llicencia **CC BY 4.0**.",
        "",
        "Descarregat via el mirror de HuggingFace "
        f"[`{REPO_ID}`](https://huggingface.co/datasets/{REPO_ID})",
        "(canal ch01, 16 kHz, segments de 4 s reunits en una unica pista continua).",
        "",
        "Si es publica el dataset sintetic derivat, cal mantenir aquesta atribucio.",
        "",
        "## Ambients",
        "",
        "| Fitxer | DEMAND | Descripcio |",
        "|---|---|---|",
    ]
    for nom, spec in ambients.items():
        linies.append(f"| `{nom}.wav` | {spec['demand']} | {spec['descripcio']} |")
    (Path(output_dir) / "README.md").write_text("\n".join(linies) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ambients", nargs="*", default=list(AMBIENTS),
                        choices=list(AMBIENTS),
                        help="Quins ambients construir (per defecte, tots)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache de HuggingFace (per defecte, la del sistema)")
    parser.add_argument("--netejar-cache", action="store_true",
                        help="Esborra la descarrega intermedia de HF en acabar")
    args = parser.parse_args()

    print(f"Banc de sorolls -> {args.output_dir}  (@{args.sample_rate} Hz)")
    for nom in args.ambients:
        construir(nom, AMBIENTS[nom], args.output_dir, args.sample_rate,
                  args.overwrite, cache_dir=args.cache_dir)

    escriure_atribucio(args.output_dir, {n: AMBIENTS[n] for n in args.ambients})
    print(f"\nAtribucio escrita a {args.output_dir}/README.md")

    if args.netejar_cache and args.cache_dir:
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        print(f"Cache esborrada: {args.cache_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prova d'escolta: amb quina grafia s'acosta OmniVoice a la `u` aranesa [y]?"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AQUI))

import ablacio as A
import respelling

DESTI = AQUI / "escolta/u_simbols"
SENTINELLA = "ẅ"
VOCALS = set("aeiouàèéíòóúïü")

CANDIDATS = [
    ("u",   "u",  "la grafia d'ara (control)"),
    ("udie", "ü", "[y] en alemany, turc, hongares"),
    ("y",   "y",  "[y] en danes, noruec, suec"),
    ("iu",  "iu", "aproximacio amb grafia catalana"),
    ("ui",  "ui", "aproximacio amb grafia catalana"),
    ("afi", "ʏ",  "el simbol AFI, per veure si el model hi reacciona"),
]


def u_nuclis(mot: str) -> list[int]:
    """Posicions de `u` inequivocament NUCLI."""
    fora = []
    for i, c in enumerate(mot):
        if c != "u":
            continue
        ant = mot[i - 1] if i else ""
        seg = mot[i + 1] if i + 1 < len(mot) else ""
        if ant in "qg":
            continue
        if ant in VOCALS:
            continue
        if seg in VOCALS:
            continue
        fora.append(i)
    return fora


def marca(text: str) -> str:
    """Les `u` nucli passen a sentinella, abans que `respelling` hi arribi."""
    import re
    def _mot(m):
        """Reescriu un mot marcant-hi les u a tractar."""
        mot = m.group(0)
        pos = set(u_nuclis(mot.lower()))
        return "".join(SENTINELLA if i in pos else c for i, c in enumerate(mot))
    return re.sub(r"[^\W\d_]+(?:[''][^\W\d_]+)*", _mot, text)


def text_variant(cru: str, simbol: str, plurals) -> str:
    """Reescriu la frase amb la variant i posa el simbol al lloc de la sentinella."""
    return respelling.reescriu(marca(cru), plurals=plurals).replace(SENTINELLA, simbol)


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frases", type=int, default=12)
    p.add_argument("--nomes-textos", action="store_true")
    p.add_argument("--dispositiu", default="auto")
    p.add_argument("--lot-tts", type=int, default=4)
    args = p.parse_args()

    man = A.v04.carregar_manifest(A.MANIFEST)
    test = [m for m in man if m["split"] == "test"]
    plurals = A.carregar_plurals()

    puntuades = [(sum(len(u_nuclis(w)) for w in respelling.mots(m["text"])), i, m)
                 for i, m in enumerate(test)]
    triades = [(i, m) for n, i, m in sorted(puntuades, key=lambda x: -x[0])[:args.frases] if n]
    print(f"{len(triades)} frases amb `u` nucli\n")

    textos = {et: [text_variant(m["text"], s, plurals) for _, m in triades]
              for et, s, _ in CANDIDATS}
    for k, (_, m) in enumerate(triades[:4 if args.nomes_textos else 2]):
        print(f"  ARANES   {m['text']}")
        for et, s, _ in CANDIDATS:
            print(f"  {et:<8} {textos[et][k]}")
        print()
    if args.nomes_textos:
        return 0

    from transformers.utils import logging as tlog
    tlog.set_verbosity_error(); tlog.disable_progress_bar()
    import torch
    dispositiu = A.w03.triar_dispositiu(args.dispositiu)
    veus = A.v04.triar_veus(man, 4)

    tts = None
    for et, _, _ in CANDIDATS:
        if et == "u":
            continue
        A.VARIANTS[f"u_{et}"] = (frozenset(), f"simbol {et}")
        items = [(i, m, textos[et][k]) for k, (i, m) in enumerate(triades)]
        tts = A.sintetitzar(f"u_{et}", items, veus, dispositiu, args.lot_tts, tts)
    if tts is not None:
        del tts
        torch.cuda.empty_cache()

    if DESTI.exists():
        shutil.rmtree(DESTI)
    DESTI.mkdir(parents=True)
    files = []
    for n, (i, m) in enumerate(triades, 1):
        clip = Path(m["audio"]).stem.removeprefix("common_voice_oc_")
        veu = veus[i % len(veus)]["id"]
        copia = {
            f"f{n:02d}_0_real.wav": Path(m["audio"]),
            f"f{n:02d}_1_actual_u.wav": A.REUTILITZATS["totes"] / f"cv{clip}_hibrid_{veu}.wav",
            f"f{n:02d}_9_ref-oc.wav": ROOT / f"lab/proves_inicials/aranes/audios/validacio_tts/oc/cv{clip}_tts-oc_{veu}.wav",
        }
        for k, (et, _, _) in enumerate([c for c in CANDIDATS if c[0] != "u"], 2):
            copia[f"f{n:02d}_{k}_{et}.wav"] = A.DIR_AUDIOS / f"u_{et}" / f"cv{clip}_u_{et}_{veu}.wav"
        for nom, orig in copia.items():
            shutil.copy(orig, DESTI / nom)
        files.append({"id": f"f{n:02d}", "aranes": m["text"],
                      "mots_amb_u": " ".join(w for w in respelling.mots(m["text"]) if u_nuclis(w)),
                      **{f"text_{et}": textos[et][n - 1] for et, _, _ in CANDIDATS},
                      "quin_sona_millor": "", "comentari": ""})

    with (DESTI / "frases.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(files[0]))
        w.writeheader(); w.writerows(files)

    taula = "\n".join(f"| `fNN_{k}_{et}.wav` | `{s}` | {q} |"
                      for k, (et, s, q) in enumerate(
                          [("actual_u", "u", "el text que genera `respelling.py` avui (control)")]
                          + [(e, s, q) for e, s, q in CANDIDATS if e != "u"], 1))
    (DESTI / "README.md").write_text(f"""# Escolta: quina grafia dona la `u` aranesa [y]?

{len(triades)} frases del split test amb `u` de nucli. Per a cada frase, la MATEIXA veu
clonada i el MATEIX text, canviant nomes la grafia d'aquelles `u`.

| Fitxer | Grafia | Que es |
|---|---|---|
| `fNN_0_real.wav` | — | **persona real** (Common Voice): aixi ha de sonar la `u` |
{taula}
| `fNN_9_ref-oc.wav` | `u` | **referencia**, no candidat: el mateix text amb `language="oc"` |

## Com escoltar-ho

Compara cada candidat amb el `_0_real` **nomes en els mots de la columna `mots_amb_u`**
del `frases.csv`; la resta de la frase es igual a totes les versions i no aporta res.

El `_9_ref-oc` hi es perque `language="oc"` sap que la `u` occitana es [y]. Si cap
candidat s'hi acosta, la conclusio es que per aquesta via no s'hi arriba i que el so
nomes es pot obtenir per l'etiqueta d'idioma, no per l'ortografia. Compte: la resta de
la frase, amb `oc`, sona afrancesada -- es per aixo que `oc` no es el joc de treball.

Anota a `frases.csv` (`quin_sona_millor`, `comentari`).

## On s'ha aplicat

Nomes a les `u` inequivocament nucli. **No** s'hi toca la de `qu-`/`gu-` (`que`,
`guardar`), ni la semivocal dels diftongs (`au`, `èu`, `iu`), ni la seguida de vocal
(`ua`, `sua`), que es dubtosa. Al test son 30 `u` de 238 (13 %).

La substitucio es fa ABANS de `respelling.py`, no despres: com que `respelling` converteix
`o`->`u` (`senhor` -> `senyur`), fer-ho al reves canviaria tambe aquelles `u`, que son
[u] i no [y].
""", encoding="utf-8")
    print(f"\nCarpeta d'escolta: {DESTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

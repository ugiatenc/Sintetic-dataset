#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reescriptura ortografica d'un idioma a la grafia d'un altre, per a un TTS que no el te (aranes -> catala)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VOCALS = set("aeiouàèéíòóúïü")
ACCENT_FINAL = {"a": "à", "e": "é", "i": "í", "o": "ó", "u": "ú"}


def ruta_normes(idioma: str = "aranes") -> Path:
    """Ruta del JSON de normes ortografiques d'un idioma."""
    import phonetics
    return phonetics.dir_diccionaris(idioma) / f"normes_ortografiques_{idioma}.json"


@lru_cache(maxsize=4)
def carregar_normes(idioma: str = "aranes") -> dict:
    """Carrega les normes ortografiques."""
    return json.loads(ruta_normes(idioma).read_text(encoding="utf-8"))


def _restaura_caixa(original: str, nou: str) -> str:
    """Torna al resultat la caixa de l'original."""
    if original.isupper() and len(original) > 1:
        return nou.upper()
    if original[:1].isupper():
        return nou[:1].upper() + nou[1:]
    return nou


def _reescriu_part(part: str, es_plural: bool, traca: dict,
                   nomes_digrafs: bool = False) -> str:
    """Un tros de mot (sense apostrofs), en minuscules."""
    def marca(regla, n=1):
        """Compta una aplicacio de regla a la traca."""
        if n:
            traca[regla] = traca.get(regla, 0) + n

    marca("nh", part.count("nh")); part = part.replace("nh", "ny")
    marca("lh", part.count("lh")); part = part.replace("lh", "ll")

    def _sh(m):
        """Regla sh -> x / ix segons el context."""
        marca("sh")
        anterior = m.group(1) or ""
        if anterior == "i" or anterior not in VOCALS:
            return anterior + "x"
        return anterior + "ix"
    part = re.sub(r"(.?)sh", _sh, part)

    if nomes_digrafs:
        return part

    if es_plural and part.endswith("ns") and len(part) > 2:
        arrel = part[:-2]
        if arrel and arrel[-1] in ACCENT_FINAL:
            arrel = arrel[:-1] + ACCENT_FINAL[arrel[-1]]
        marca("ns_plural")
        part = arrel + "s"

    sortida = []
    for i, c in enumerate(part):
        seguent = part[i + 1] if i + 1 < len(part) else ""
        anterior = part[i - 1] if i else ""
        if c in "oó" and ((seguent and seguent in "uúoó")
                          or (anterior and anterior in "uúoó")):
            marca("ou_sense_tocar")
            sortida.append(c)
        elif c == "o":
            marca("o_a_u"); sortida.append("u")
        elif c == "ó":
            marca("o_a_u"); sortida.append("ú")
        else:
            sortida.append(c)
    return "".join(sortida)


def reescriu(text: str, idioma: str = "aranes", traca: dict | None = None,
             plurals: set[str] | None = None, nomes_digrafs: bool = False) -> str:
    """Text aranes -> text en grafia catalana."""
    traca = traca if traca is not None else {}
    plurals = {p.lower() for p in (plurals or ())}
    if "·h" in text:  # el punt volat aranes (en·honsar) no es cap so
        traca["punt_volat_h"] = traca.get("punt_volat_h", 0) + text.count("·h")
        text = text.replace("·h", "")

    def _mot(m):
        """Reescriu un mot sencer."""
        mot = m.group(0)
        es_plural = mot.lower() in plurals
        trossos = re.split(r"([''])", mot)
        return "".join(t if t in "''" else
                       _restaura_caixa(t, _reescriu_part(t.lower(), es_plural, traca,
                                                         nomes_digrafs))
                       for t in trossos)

    return re.sub(r"[^\W\d_]+(?:[''][^\W\d_]+)*", _mot, text)


def mots(text: str) -> list[str]:
    """Els mots d'un text, en minuscules."""
    return re.findall(r"[^\W\d_]+(?:[''][^\W\d_]+)*", text.lower())


def candidats_plural(textos) -> list[str]:
    """Mots acabats en -ns: els unics que la llista de plurals ha de resoldre."""
    vistos = {m for t in textos for m in mots(t) if m.endswith("ns")}
    return sorted(vistos)


def main() -> int:
    """Prova rapida des de la linia d'ordres."""
    import argparse
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fitxer", type=Path, help="fitxer amb una frase per linia")
    p.add_argument("--text", help="una frase solta")
    p.add_argument("--plurals", type=Path, help="JSON amb la llista de mots plurals")
    p.add_argument("--idioma", default="aranes")
    p.add_argument("--traca", action="store_true", help="recompte de regles aplicades")
    args = p.parse_args()

    frases = ([args.text] if args.text else
              args.fitxer.read_text(encoding="utf-8").splitlines() if args.fitxer else [])
    plurals = set(json.loads(args.plurals.read_text(encoding="utf-8"))) if args.plurals else set()
    total: dict = {}
    for frase in frases:
        if frase.strip():
            print(f"  RAW  {frase}\n  TTS  {reescriu(frase, args.idioma, total, plurals)}\n")
    if args.traca:
        print("Regles aplicades:")
        for regla, n in sorted(total.items(), key=lambda kv: -kv[1]):
            print(f"  {regla:<20} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

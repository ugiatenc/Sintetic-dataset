#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reescriptura aranes -> grafia catalana, amb les regles ACTIVABLES d'una en una."""

from __future__ import annotations

import re

VOCALS = set("aeiouàèéíòóúïü")
ACCENTUADES = set("àèéíòóú")
ACCENT_FINAL = {"a": "à", "e": "é", "i": "í", "o": "ó", "u": "ú"}

REGLES = ("nh", "lh", "sh", "ns_plural", "o_tot", "o_tonica", "ng", "ts")
REGLES_ACTUALS = frozenset({"nh", "lh", "sh", "ns_plural", "o_tot"})


def nuclis(mot: str) -> list[tuple[int, int]]:
    """Grups de vocals contigues, com a aproximacio de nucli sillabic."""
    fora, i = [], 0
    while i < len(mot):
        if mot[i] in VOCALS:
            j = i
            while j + 1 < len(mot) and mot[j + 1] in VOCALS:
                j += 1
            fora.append((i, j))
            i = j + 1
        else:
            i += 1
    return fora


def nucli_tonic(mot: str) -> int | None:
    """Index del nucli tonic segons les regles d'accentuacio catalanes."""
    ns = nuclis(mot)
    if not ns:
        return None
    for k, (a, b) in enumerate(ns):
        if any(c in ACCENTUADES for c in mot[a:b + 1]):
            return k
    if len(ns) == 1:
        return 0
    if mot[-1] in VOCALS or re.search(r"([aeiouàèéíòóúïü]s|en|in)$", mot):
        return len(ns) - 2
    return len(ns) - 1


def _digrafs(part: str, regles: set, marca) -> str:
    """Aplica les regles de digrafs (nh, lh, sh) a un tros."""
    if "nh" in regles:
        marca("nh", part.count("nh")); part = part.replace("nh", "ny")
    if "lh" in regles:
        marca("lh", part.count("lh")); part = part.replace("lh", "ll")
    if "sh" in regles:
        def _sh(m):
            """Regla sh -> x / ix segons el context."""
            marca("sh")
            anterior = m.group(1) or ""
            if anterior == "i" or anterior not in VOCALS:
                return anterior + "x"
            return anterior + "ix"
        part = re.sub(r"(.?)sh", _sh, part)
    return part


def _ns_plural(part: str, es_plural: bool, marca) -> str:
    """Regla del plural en -ns."""
    if not (es_plural and part.endswith("ns") and len(part) > 2):
        return part
    arrel = part[:-2]
    if arrel and arrel[-1] in ACCENT_FINAL:
        arrel = arrel[:-1] + ACCENT_FINAL[arrel[-1]]
    marca("ns_plural")
    return arrel + "s"


def _ts(part: str, marca) -> str:
    """-tz final -> -ts."""
    if part.endswith("tz") and len(part) > 2:
        marca("ts")
        return part[:-2] + "ts"
    return part


def _ng(part: str, marca) -> str:
    """-n final darrere vocal -> -ng, per obtenir [ŋ] (`camin` -> `caming`)."""
    if len(part) > 1 and part.endswith("n") and part[-2] in VOCALS:
        marca("ng")
        return part + "g"
    return part


def _vocals(part: str, regles: set, marca) -> str:
    """o/ó -> u/ú."""
    if not (regles & {"o_tot", "o_tonica"}):
        return part
    nomes_tonica = "o_tonica" in regles
    it = nucli_tonic(part) if nomes_tonica else None
    ns = nuclis(part) if nomes_tonica else []
    tonics = set(range(ns[it][0], ns[it][1] + 1)) if (nomes_tonica and it is not None) else set()

    fora = []
    for i, c in enumerate(part):
        seguent = part[i + 1] if i + 1 < len(part) else ""
        if c in "oó" and seguent == "u":
            fora.append(c)
        elif c in "oó" and (not nomes_tonica or i in tonics):
            marca("o_tonica" if nomes_tonica else "o_tot")
            fora.append("u" if c == "o" else "ú")
        else:
            fora.append(c)
    return "".join(fora)


def _restaura_caixa(original: str, nou: str) -> str:
    """Torna al resultat la caixa (majuscules) de l'original."""
    if original.isupper() and len(original) > 1:
        return nou.upper()
    if original[:1].isupper():
        return nou[:1].upper() + nou[1:]
    return nou


def _reescriu_part(part: str, es_plural: bool, regles: set, traca: dict) -> str:
    """Reescriu un tros de mot amb les regles actives."""
    def marca(regla, n=1):
        """Compta una aplicacio de regla a la traca."""
        if n:
            traca[regla] = traca.get(regla, 0) + n

    part = _digrafs(part, regles, marca)
    if "ns_plural" in regles:
        part = _ns_plural(part, es_plural, marca)
    if "ts" in regles:
        part = _ts(part, marca)
    if "ng" in regles:
        part = _ng(part, marca)
    return _vocals(part, regles, marca)


MOT = r"[^\W\d_]+(?:[''][^\W\d_]+)*"


def reescriu(text: str, regles, traca: dict | None = None,
             plurals: set[str] | None = None) -> str:
    """Reescriu un text amb un subconjunt de regles, comptant-les a `traca`."""
    regles = set(regles)
    desconegudes = regles - set(REGLES)
    if desconegudes:
        raise ValueError(f"regles desconegudes: {sorted(desconegudes)}")
    if {"o_tot", "o_tonica"} <= regles:
        raise ValueError("'o_tot' i 'o_tonica' son excloents")
    traca = traca if traca is not None else {}
    plurals = {p.lower() for p in (plurals or ())}

    def _mot(m):
        """Reescriu un mot sencer."""
        mot = m.group(0)
        es_plural = mot.lower() in plurals
        trossos = re.split(r"([''])", mot)
        return "".join(t if t in "''" else
                       _restaura_caixa(t, _reescriu_part(t.lower(), es_plural, regles, traca))
                       for t in trossos)

    return re.sub(MOT, _mot, text)


def mots(text: str) -> list[str]:
    """Mots del text, en minuscula."""
    return re.findall(MOT, text.lower())

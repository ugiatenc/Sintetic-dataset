#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maquinaria generica de corpus: netejar, tallar en frases, filtrar, deduplicar i desar."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

CAR_PER_SEGON_REAL = 9.16
CAR_PER_SEGON_TTS = 14.7
CAR_PER_SEGON_LENT = 11.9

PATRONS_BROSSA = [
    re.compile(r"\[\d+\]"),
    re.compile(r"==+[^=]*==+"),
    re.compile(r"https?://\S+"),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"[​‎‏﻿]"),
]
ABREVIATURA_NO_TALLA = re.compile(
    r"(\b(sr|sra|srs|dr|dra|sta|st|núm|pàg|pag|p|fig|vs|etc|aprox)\.|\b[A-ZÀ-Ü]\.)$",
    re.IGNORECASE)
FINALS_FRASE = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][\"»')”’]))\s+(?=[\"«'(“‘]?[A-ZÀÁÈÉÍÏÒÓÚÜÇ])")
TALLS = [re.compile(r"(?<=[;:])\s+"), re.compile(r"\s+[–—]\s+")]
TALLS_FEBLES = [re.compile(
    r"(?<=,)\s+(?=(?:e|mès|pr'amor que|per çò que|mentre que|encara que|totun|alavetz|"
    r"donques|per tant|atau que|en cambi|ath delà|ath contrari|dempús|après)\s)")]

FINALS_IMPOSSIBLES = {
    "e", "o", "que", "qu", "de", "d", "a", "en", "per", "damb", "entà", "entara",
    "entath", "ath", "ara", "as", "dera", "deth", "des", "ders", "eth", "era", "es",
    "er", "un", "ua", "sus", "coma", "mes", "mès", "se", "non", "i", "li", "lo", "la",
    "les", "ne", "ja", "pera", "peth", "ena", "enes", "sense", "enquia", "tanben",
    "pr", "sense", "dempus", "segontes", "malgrat", "tant", "tan", "cada", "quan",
}


def normalitzar(text: str) -> str:
    """Minuscules i sense accents."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def clau_dedup(text: str) -> str:
    """Clau de duplicat: sense accents, sense puntuacio i sense espais de mes."""
    net = re.sub(r"[^\w\s]", "", normalitzar(text))
    return re.sub(r"\s+", " ", net).strip()


def netejar(text: str) -> str:
    """Treu la brossa (patrons) d'un text."""
    for patro in PATRONS_BROSSA:
        text = patro.sub(" ", text)
    text = (text.replace("\xa0", " ").replace("’", "'")
                .replace("“", '"').replace("”", '"').replace("…", "..."))
    return re.sub(r"[ \t]+", " ", text).strip()


def a_frases(text: str) -> list[str]:
    """Un bloc de text -> frases."""
    fora = []
    for bloc in text.split("\n"):
        bloc = netejar(bloc)
        if not bloc:
            continue
        peces = [f.strip() for f in FINALS_FRASE.split(bloc) if f.strip()]
        unides: list[str] = []
        for pe in peces:
            if unides and ABREVIATURA_NO_TALLA.search(unides[-1]):
                unides[-1] = f"{unides[-1]} {pe}"
            else:
                unides.append(pe)
        fora.extend(unides)
    return [f for f in fora if f]


def dividir_llarga(frase: str, max_objectiu: int, min_car: int,
                   max_dur: int | None = None) -> list[str]:
    """Parteix una frase massa llarga pels seus limits sintactics."""
    max_dur = max_objectiu if max_dur is None else max_dur
    if len(frase) <= max_objectiu:
        return [frase]

    trossos = [frase]
    for patro in TALLS:
        if all(len(t) <= max_objectiu for t in trossos):
            break
        nous = []
        for t in trossos:
            nous.extend(patro.split(t) if len(t) > max_objectiu else [t])
        trossos = [t.strip() for t in nous if t.strip()]
    for patro in TALLS_FEBLES:
        if all(len(t) <= max_objectiu for t in trossos):
            break
        nous = []
        for t in trossos:
            if len(t) <= max_objectiu:
                nous.append(t)
                continue
            peces = [p.strip() for p in patro.split(t) if p.strip()]
            nous.extend(_reagrupa(peces, max_objectiu))
        trossos = nous

    if len(trossos) == 1:
        return [frase] if len(frase) <= max_dur else []

    fora: list[str] = []
    for t in trossos:
        if fora and len(t) < min_car:
            fora[-1] = f"{fora[-1]} {t}"
        else:
            fora.append(t)
    nets = [normalitza_tros(t) for t in fora if len(t) <= max_dur]
    return [t for t in nets if t and _ultim_mot(t) not in FINALS_IMPOSSIBLES]


def _reagrupa(peces: list[str], max_objectiu: int) -> list[str]:
    """Torna a enganxar trossos veins mentre capiguen a `max_objectiu`."""
    grups: list[list[str]] = []
    for p in peces:
        if grups and len(" ".join(grups[-1])) + 1 + len(p) <= max_objectiu:
            grups[-1].append(p)
        else:
            grups.append([p])
    for k in range(len(grups) - 1, 0, -1):
        while (len(grups[k - 1]) > 1
               and len(" ".join(grups[k])) < len(" ".join(grups[k - 1])) - len(grups[k - 1][-1])
               and len(" ".join(grups[k - 1][-1:] + grups[k])) <= max_objectiu):
            grups[k].insert(0, grups[k - 1].pop())
    return [" ".join(g) for g in grups]


def _ultim_mot(frase: str) -> str:
    """Ultim mot d'una frase, en minuscula."""
    mots = re.findall(r"[^\W\d_]+", frase.lower())
    return mots[-1] if mots else ""


def normalitza_tros(tros: str) -> str:
    """Un tros de frase partida -> una frase per dret propi."""
    tros = tros.strip().rstrip(",;:–—-").strip()
    if not tros:
        return tros
    for i, c in enumerate(tros):
        if c.isalpha():
            tros = tros[:i] + tros[i].upper() + tros[i + 1:]
            break
    return tros if tros[-1] in ".!?" else tros + "."


def durada_estimada(text: str, car_per_segon: float = CAR_PER_SEGON_TTS) -> float:
    """Durada estimada del TTS a partir dels caracters."""
    return round(len(text) / car_per_segon, 2)


def repara_delimitadors(frase: str) -> str:
    """Treu els delimitadors ORFES d'una frase, sense tocar-ne el contingut."""
    if frase.count('"') == 1:
        frase = frase.replace('"', "")
    obre = frase.count("(")
    tanca = frase.count(")")
    if obre != tanca:
        fora, pendents = [], 0
        for c in frase:
            if c == "(":
                pendents += 1
            elif c == ")":
                if pendents == 0:
                    continue
                pendents -= 1
            fora.append(c)
        frase = "".join(fora)
        if pendents:
            restants = pendents
            net = []
            for c in reversed(frase):
                if c == "(" and restants:
                    restants -= 1
                    continue
                net.append(c)
            frase = "".join(reversed(net))
    return re.sub(r"\s{2,}", " ", frase).strip()


class Filtre:
    """Decideix si una frase entra al corpus, i per que no hi entra quan no hi entra."""

    def __init__(self, min_mots: int, max_mots: int, min_car: int, max_car: int,
                 puntuacio=None, min_marques: int = 0, exigir_final: bool = True,
                 politica_xifres: str = "descartar", expandir=None,
                 prohibides: set[str] | None = None):
        """Filtre de llargada i qualitat amb els seus llindars."""
        self.min_mots, self.max_mots = min_mots, max_mots
        self.min_car, self.max_car = min_car, max_car
        self.puntuacio, self.min_marques = puntuacio, min_marques
        self.exigir_final = exigir_final
        self.politica_xifres, self.expandir = politica_xifres, expandir
        self.prohibides = prohibides or set()

    def __call__(self, frase: str) -> tuple[str | None, str]:
        """(frase acceptada o None, motiu del descart)."""
        frase = netejar(frase)
        if not frase:
            return None, "buida"

        if re.search(r"\d", frase):
            if self.politica_xifres == "descartar":
                return None, "conte xifres"
            if self.politica_xifres == "expandir":
                if self.expandir is None:
                    return None, "conte xifres i no hi ha taula d'expansio"
                frase = self.expandir(frase)
                if re.search(r"\d", frase):
                    return None, "xifres que la taula no sap expandir"

        n_mots = len(frase.split())
        if not (self.min_mots <= n_mots <= self.max_mots):
            return None, f"llargada {n_mots} mots"
        if not (self.min_car <= len(frase) <= self.max_car):
            return None, f"llargada {len(frase)} caracters"
        if self.exigir_final:
            cua = frase.rstrip('"»)]\'’”')
            if not cua or cua[-1] not in ".!?":
                return None, "sense puntuacio final"
        frase = repara_delimitadors(frase)
        if frase.count('"') % 2 or frase.count("(") != frase.count(")"):
            return None, "cometes o parentesis desaparellats"
        if any(c.isupper() for c in frase) and frase.upper() == frase:
            return None, "tot majuscules"
        if self.prohibides and clau_dedup(frase) in self.prohibides:
            return None, "frase del test de Common Voice"
        if self.puntuacio is not None:
            fortes = self.puntuacio(frase)
            if fortes < self.min_marques:
                return None, f"filtre dialectal: {fortes} marques fortes"
        return frase, ""


def ordena_camps(r: dict) -> dict:
    """`origen` al final, darrere de `pas`."""
    if "origen" not in r:
        return r
    fora = {k: v for k, v in r.items() if k != "origen"}
    return {**fora, "origen": r["origen"]}


def escriu_jsonl(ruta: Path, registres) -> int:
    """Escriu registres en jsonl; torna quants."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with ruta.open("w", encoding="utf-8") as fh:
        for r in registres:
            fh.write(json.dumps(ordena_camps(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def llegeix_jsonl(ruta: Path) -> list[dict]:
    """Llegeix un jsonl (buit si no existeix)."""
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def hash_text(text: str) -> str:
    """Identitat d'una frase per a la deduplicacio."""
    return hashlib.sha1(clau_dedup(text).encode()).hexdigest()[:12]


def registre(font: str, i: int, text: str, origen: str = "", **extra) -> dict:
    """Construeix el registre d'una frase capturada."""
    return {"id": f"{font}-{i:07d}", "font": font, "origen": origen, "raw_text": text,
            "n_mots": len(text.split()), "n_car": len(text),
            "durada_est_s": durada_estimada(text),
            "hash": hash_text(text), **extra}


def desa_descartades(dir_font: Path, pas: int, noves: list[dict]) -> Path:
    """Afegeix les descartades d'UN pas al fitxer unic de la font, sense tocar les altres."""
    ruta = dir_font / "descartades.jsonl"
    velles = [r for r in llegeix_jsonl(ruta) if r.get("pas") != pas]
    escriu_jsonl(ruta, velles + [{**r, "pas": pas} for r in noves])
    return ruta


def desa_font(dir_font: Path, bones: list[dict], descartades: list[dict], meta: dict) -> None:
    """Escriu capturat, descartades i meta d'una font."""
    dir_font.mkdir(parents=True, exist_ok=True)
    escriu_jsonl(dir_font / "capturat.jsonl", bones)
    desa_descartades(dir_font, 1, descartades)
    hores = sum(r["durada_est_s"] for r in bones) / 3600
    (dir_font / "meta.json").write_text(json.dumps(
        {**meta, "frases": len(bones), "descartades": len(descartades),
         "caracters": sum(r["n_car"] for r in bones),
         "hores_estimades_tts": round(hores, 2),
         "data": date.today().isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {len(bones)} frases ({hores:.1f} h est.), {len(descartades)} descartades "
          f"a {dir_font}")


def aplicar_correccions(dir_font: Path, registres: list[dict]) -> tuple[list[dict], int, int]:
    """Aplica `correccions.jsonl` (si hi es)."""
    correccions = {c["id"]: c for c in llegeix_jsonl(dir_font / "correccions.jsonl")}
    if not correccions:
        return registres, 0, 0
    fora, n_cor, n_del = [], 0, 0
    for r in registres:
        c = correccions.get(r["id"])
        if c is None:
            fora.append(r)
        elif c.get("elimina"):
            n_del += 1
        else:
            nou = c["raw_text"]
            fora.append({**r, "raw_text": nou, "n_mots": len(nou.split()),
                         "n_car": len(nou), "durada_est_s": durada_estimada(nou),
                         "hash": hash_text(nou),
                         "corregida": True})
            n_cor += 1
    return fora, n_cor, n_del


def deduplicar(registres: list[dict], prioritat: list[str]) -> tuple[list[dict], list[dict]]:
    """Treu duplicats entre fonts, quedant-se la copia de la font mes fiable."""
    ordre = {f: i for i, f in enumerate(prioritat)}
    per_hash: dict[str, dict] = {}
    duplicats = []
    for r in sorted(registres, key=lambda r: ordre.get(r["font"], len(ordre))):
        guanyador = per_hash.setdefault(r["hash"], r)
        if guanyador is not r:
            duplicats.append({"id": r["id"], "font": r["font"],
                              "motiu": f"duplicada (es queda la de {guanyador['font']})",
                              "text": r["raw_text"]})
    return list(per_hash.values()), duplicats


def histograma(valors: list[float], vores: list[float]) -> list[tuple[str, int, float]]:
    """Histograma per intervals amb recompte i percentatge."""
    fora, total = [], max(1, len(valors))
    for a, b in zip(vores, vores[1:]):
        n = sum(1 for v in valors if a <= v < b)
        fora.append((f"{a:g}-{b:g}", n, 100 * n / total))
    n = sum(1 for v in valors if v >= vores[-1])
    fora.append((f">={vores[-1]:g}", n, 100 * n / total))
    return fora


def resum(registres: list[dict]) -> dict:
    """Resum d'un conjunt de registres: frases, hores, durades."""
    durades = [r["durada_est_s"] for r in registres]
    return {
        "frases": len(registres),
        "caracters": sum(r["n_car"] for r in registres),
        "hores_estimades_tts": round(sum(durades) / 3600, 2),
        "durada_mitjana_s": round(sum(durades) / max(1, len(durades)), 2),
        "per_font": {f: sum(1 for r in registres if r["font"] == f)
                     for f in sorted({r["font"] for r in registres})},
        "hores_per_font": {f: round(sum(r["durada_est_s"] for r in registres
                                        if r["font"] == f) / 3600, 2)
                           for f in sorted({r["font"] for r in registres})},
    }

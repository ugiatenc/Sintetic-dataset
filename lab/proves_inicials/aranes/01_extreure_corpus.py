#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extreu un corpus de text en ARANÈS per a la línia base d'aquest idioma."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

AQUI = Path(__file__).resolve().parent
SORTIDA_DEFECTE = AQUI / "corpus_aranes.txt"

USER_AGENT = os.environ.get(
    "CORPUS_UA", "sintetic-dataset/0.1 (recerca en ASR; contacte: posa-hi-el-teu@correu)")
API_WIKI = "https://oc.wikipedia.org/w/api.php"
PAUSA_S = 1.0

MARQUES_ARANES = {
    "eth", "era", "es", "eths", "eras", "er",
    "dera", "deth", "ders", "des", "dels",
    "ath", "ara", "as", "aths",
    "ena", "enes", "eneth", "ath",
    "ei", "son", "ac", "que",
    "damb", "ua", "un", "aguest", "aguesta", "aguestes", "aguesti",
    "tanben", "tostemp", "toti", "nau", "naui", "hèr", "hèsta", "hilh",
}
MARQUES_FORTES = {
    "eth", "eths", "dera", "deth", "ath", "aths", "enes", "eneth", "ei",
    "damb", "ua", "aguest", "aguesta", "aguestes", "aguesti", "naui",
    "tostemp", "hèr", "hilh", "hemna", "hònt", "huec",
}

MIN_MOTS = 6
MAX_MOTS = 45

PATRONS_BROSSA = [
    re.compile(r"\[\d+\]"),
    re.compile(r"\s*\|\s*"),
    re.compile(r"==+[^=]*==+"),
    re.compile(r"https?://\S+"),
]
FINALS_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀÁÈÉÍÒÓÚÇ])")


def normalitzar(text: str) -> str:
    """Minúscules i sense accents, per comparar marques i detectar duplicats."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def puntuacio_aranes(frase: str) -> tuple[int, int]:
    """(marques fortes, marques totals) que conté la frase."""
    mots = re.findall(r"[\w'’]+", normalitzar(frase))
    mots = [m.strip("'’") for m in mots]
    fortes = sum(1 for m in mots if m in {normalitzar(x) for x in MARQUES_FORTES})
    totals = sum(1 for m in mots if m in {normalitzar(x) for x in MARQUES_ARANES})
    return fortes, totals


def netejar(text: str) -> str:
    """Treu la brossa (patrons) d'un text."""
    for patro in PATRONS_BROSSA:
        text = patro.sub(" ", text)
    text = text.replace("\xa0", " ").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def a_frases(text: str) -> list[str]:
    """Parteix un bloc de text en frases."""
    frases = []
    for bloc in text.split("\n"):
        bloc = netejar(bloc)
        if not bloc:
            continue
        frases.extend(f.strip() for f in FINALS_FRASE.split(bloc))
    return [f for f in frases if f]


def filtrar(frases: list[str], min_marques: int) -> tuple[list[str], list[str]]:
    """Frases acceptades i descartades."""
    bones, dolentes, vistes = [], [], set()
    for frase in frases:
        n_mots = len(frase.split())
        if not (MIN_MOTS <= n_mots <= MAX_MOTS):
            dolentes.append(f"[llargada {n_mots}] {frase}")
            continue
        if not frase[-1] in ".!?":
            dolentes.append(f"[sense final] {frase}")
            continue
        fortes, _ = puntuacio_aranes(frase)
        if fortes < min_marques:
            dolentes.append(f"[no aranès: {fortes} marques] {frase}")
            continue
        clau = normalitzar(frase)
        if clau in vistes:
            dolentes.append(f"[duplicada] {frase}")
            continue
        vistes.add(clau)
        bones.append(frase)
    return bones, dolentes


def demanar(s: requests.Session, url: str, params: dict | None = None,
            intents: int = 4) -> requests.Response:
    """GET amb pausa i reintents."""
    for intent in range(intents):
        time.sleep(PAUSA_S)
        r = s.get(url, params=params, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        espera = float(r.headers.get("Retry-After", 5)) * (intent + 1)
        print(f"  429 d'{url.split('/')[2]}; esperant {espera:.0f} s", file=sys.stderr)
        time.sleep(espera)
    r.raise_for_status()
    return r


def sessio() -> requests.Session:
    """Sessio HTTP amb el User-Agent del projecte."""
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def font_wikipedia(termes: list[str], max_articles: int) -> list[str]:
    """Text pla dels articles d'oc.wikipedia que la cerca retorna per a cada terme."""
    s = sessio()
    titols: list[str] = []
    for terme in termes:
        r = demanar(s, API_WIKI, params={
            "action": "query", "list": "search", "srsearch": terme,
            "srlimit": max_articles, "format": "json",
        })
        for encert in r.json()["query"]["search"]:
            if encert["title"] not in titols:
                titols.append(encert["title"])
    print(f"  {len(titols)} articles trobats a oc.wikipedia")

    textos = []
    for i, titol in enumerate(titols, 1):
        r = demanar(s, API_WIKI, params={
            "action": "query", "prop": "extracts", "explaintext": 1,
            "exsectionformat": "plain", "titles": titol, "format": "json",
        })
        for pagina in r.json()["query"]["pages"].values():
            if pagina.get("extract"):
                textos.append(pagina["extract"])
        if i % 20 == 0:
            print(f"  {i}/{len(titols)} articles descarregats")
    return textos


def _paragrafs(html: str, selector: str) -> str:
    """Text dels paragrafs d'una pagina HTML, sense navegacio ni scripts."""
    sopa = BeautifulSoup(html, "html.parser")
    for tag in sopa(["script", "style", "nav", "header", "footer", "aside", "figcaption"]):
        tag.decompose()
    node = sopa.select_one(selector) or sopa
    return "\n".join(p.get_text(" ", strip=True) for p in node.find_all("p"))


def font_html(base: str, patro_article: str, selector: str, max_articles: int) -> list[str]:
    """Segueix els enllaços d'article de la portada i en treu els paràgrafs."""
    s = sessio()
    portada = demanar(s, base)
    sopa = BeautifulSoup(portada.text, "html.parser")

    urls, vistes = [], set()
    for a in sopa.find_all("a", href=True):
        url = requests.compat.urljoin(base, a["href"])
        if re.search(patro_article, url) and url not in vistes:
            vistes.add(url)
            urls.append(url)
    urls = urls[:max_articles]
    print(f"  {len(urls)} articles trobats a {base}")

    textos = []
    for url in urls:
        try:
            textos.append(_paragrafs(demanar(s, url).text, selector))
        except requests.RequestException as e:
            print(f"  avís: {url} ha fallat ({type(e).__name__})", file=sys.stderr)
    return textos


FONTS = {
    "wikipedia": lambda termes, n: font_wikipedia(termes, n),
    "arannoticies": lambda termes, n: font_html(
        "http://arannoticies.com/", r"arannoticies\.com/[a-z0-9-]{25,}/$",
        "article, .entry-content, main", n),
    "jornalet": lambda termes, n: font_html(
        "https://www.jornalet.com/", r"jornalet\.com/(nova|opinion|editorial|entrevista)/\d+/",
        "article, .article-body, main", n),
}


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--font", choices=sorted(FONTS), default="wikipedia")
    p.add_argument("--cerca", nargs="+", default=["Val d'Aran", "Aran", "aranés"],
                   help="termes de cerca (només per a --font wikipedia)")
    p.add_argument("--n-frases", type=int, default=50, help="frases a desar (defecte: 50)")
    p.add_argument("--max-articles", type=int, default=30,
                   help="articles a llegir per terme de cerca (defecte: 30)")
    p.add_argument("--min-marques", type=int, default=2,
                   help="marques araneses fortes mínimes per frase (defecte: 2). Amb 1 "
                        "entra més text però també lengadocià amb algun gasconisme")
    p.add_argument("--sortida", type=Path, default=SORTIDA_DEFECTE)
    p.add_argument("--descartades", action="store_true",
                   help="desa també les frases rebutjades i el motiu, per calibrar el filtre")
    args = p.parse_args()

    print(f"Font: {args.font}")
    textos = FONTS[args.font](args.cerca, args.max_articles)
    if not textos:
        print("Cap text descarregat.", file=sys.stderr)
        return 1

    frases = []
    for text in textos:
        frases.extend(a_frases(text))
    print(f"  {len(frases)} frases en brut")

    bones, dolentes = filtrar(frases, args.min_marques)
    print(f"  {len(bones)} frases passen el filtre d'aranès")
    if len(bones) < args.n_frases:
        print(f"  avís: només {len(bones)} de les {args.n_frases} demanades. Puja "
              f"--max-articles, afegeix termes a --cerca o baixa --min-marques.",
              file=sys.stderr)

    bones = bones[:args.n_frases]
    args.sortida.write_text("\n".join(bones) + "\n", encoding="utf-8")
    print(f"\n{len(bones)} frases -> {args.sortida}")

    if args.descartades:
        ruta = args.sortida.with_name(args.sortida.stem + "_descartades.txt")
        ruta.write_text("\n".join(dolentes) + "\n", encoding="utf-8")
        print(f"{len(dolentes)} descartades -> {ruta}")

    for frase in bones[:3]:
        print(f"  · {frase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

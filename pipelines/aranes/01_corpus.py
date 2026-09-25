#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 1: captura el text de cada font (brut.jsonl, reutilitzable) i hi aplica el filtre dialectal (capturat.jsonl)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import config as C
import corpus

USER_AGENT = os.environ.get(
    "CORPUS_UA", "sintetic-dataset/0.1 (recerca en ASR; contacte: ugiat@ugiat.com)")
PAUSA_S = 1.0


def fer_filtre() -> corpus.Filtre:
    """El filtre amb els llindars de `config.py`, i amb les frases del test de Common Voice prohibides."""
    prohibides = set()
    if C.MANIFEST_CV.exists():
        for linia in C.MANIFEST_CV.read_text(encoding="utf-8").splitlines():
            if linia.strip():
                prohibides.add(corpus.clau_dedup(json.loads(linia)["text"]))
    print(f"  {len(prohibides)} frases de Common Voice excloses (test i la resta de splits)")
    return corpus.Filtre(
        min_mots=C.MIN_MOTS, max_mots=10 ** 6,
        min_car=C.MIN_CAR, max_car=C.LIMIT_CAPTURA,
        puntuacio=C.puntuacio_aranes, min_marques=C.MIN_MARQUES,
        politica_xifres=C.POLITICA_XIFRES, expandir=C.expandir_xifres,
        prohibides=prohibides)


def processa(font: str, textos, filtre: corpus.Filtre, max_frases: int) -> tuple[list, list]:
    """Blocs de text -> registres acceptats i descartats."""
    bones, dolentes, vistes, i = [], [], set(), 0
    for origen, text in textos:
        for frase in corpus.a_frases(text):
            if frase[-1] not in ".!?":
                if len(dolentes) < 50000:
                    dolentes.append({"font": font, "origen": origen,
                                     "motiu": "sense puntuacio final", "text": frase[:300]})
                continue
            for tros in (frase,):
                net, motiu = filtre(tros)
                if net is None:
                    if len(dolentes) < 50000:
                        dolentes.append({"font": font, "origen": origen,
                                         "motiu": motiu, "text": tros[:300]})
                    continue
                clau = corpus.clau_dedup(net)
                if clau in vistes:
                    continue
                vistes.add(clau)
                fortes, foranes = C.compta_marques(net)
                bones.append(corpus.registre(font, i, net, origen,
                                             marques_fortes=fortes, marques_foranes=foranes))
                i += 1
                if len(bones) >= max_frases:
                    return bones, dolentes
    return bones, dolentes


def _sessio():
    """Sessio HTTP amb reintents i el User-Agent del projecte."""
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _demanar(s, url, params=None, intents=5):
    """GET amb pausa i reintents."""
    import requests
    ultim = None
    for intent in range(intents):
        time.sleep(PAUSA_S)
        try:
            r = s.get(url, params=params, timeout=90)
        except (requests.ConnectionError, requests.Timeout) as e:
            ultim = e
            espera = 2 ** intent
            print(f"    {type(e).__name__}; reintent en {espera} s", file=sys.stderr)
            time.sleep(espera)
            continue
        if r.status_code != 429:
            r.raise_for_status()
            return r
        espera = float(r.headers.get("Retry-After", 5)) * (intent + 1)
        print(f"    429; esperant {espera:.0f} s", file=sys.stderr)
        time.sleep(espera)
    if ultim:
        raise ultim
    r.raise_for_status()
    return r


def font_hf(cfg: dict, max_files: int):
    """Corpus de HuggingFace, en streaming."""
    from datasets import load_dataset
    d = load_dataset(cfg["dataset"], split="train", streaming=True)
    for i, fila in enumerate(d):
        if i >= max_files:
            return
        text = (fila.get(cfg["camp"]) or "").strip()
        if text:
            yield f"{cfg['dataset']}#{i}", text


def font_wikipedia(cfg: dict, max_articles: int):
    """Captura els articles de la Wikipedia occitana de la categoria d'Aran."""
    s = _sessio()
    titols = []
    for terme in cfg["cerca"]:
        r = _demanar(s, cfg["api"], params={"action": "query", "list": "search",
                                            "srsearch": terme, "srlimit": max_articles,
                                            "format": "json"})
        titols += [h["title"] for h in r.json()["query"]["search"]]
    titols = list(dict.fromkeys(titols))
    print(f"  {len(titols)} articles trobats")
    for i, titol in enumerate(titols, 1):
        try:
            r = _demanar(s, cfg["api"], params={"action": "query", "prop": "extracts",
                                                "explaintext": 1, "exsectionformat": "plain",
                                                "titles": titol, "format": "json"})
            for pg in r.json()["query"]["pages"].values():
                if pg.get("extract"):
                    yield f"oc.wikipedia:{titol}", pg["extract"]
        except Exception as e:
            print(f"    avis: {titol} ha fallat ({type(e).__name__})", file=sys.stderr)
        if i % 20 == 0:
            print(f"    {i}/{len(titols)}")


def font_html(cfg: dict, max_articles: int):
    """Captura les pagines HTML d'una font (llistat i articles)."""
    import requests
    from bs4 import BeautifulSoup
    s = _sessio()
    sopa = BeautifulSoup(_demanar(s, cfg["base"]).text, "html.parser")
    urls = []
    for a in sopa.find_all("a", href=True):
        url = requests.compat.urljoin(cfg["base"], a["href"])
        if re.search(cfg["patro"], url) and url not in urls:
            urls.append(url)
    urls = urls[:max_articles]
    print(f"  {len(urls)} articles trobats")
    for url in urls:
        try:
            pagina = BeautifulSoup(_demanar(s, url).text, "html.parser")
            for tag in pagina(["script", "style", "nav", "header", "footer",
                               "aside", "figcaption"]):
                tag.decompose()
            node = pagina.select_one(cfg["selector"]) or pagina
            yield url, "\n".join(p.get_text(" ", strip=True) for p in node.find_all("p"))
        except Exception as e:
            print(f"    avis: {url} ha fallat ({type(e).__name__})", file=sys.stderr)


def font_wp(cfg: dict, max_articles: int):
    """Arxiu sencer d'un WordPress per la seva API REST."""
    import html as _html
    s = _sessio()
    per_pagina, pagina, vistos = 25, 1, 0
    while vistos < max_articles:
        r = _demanar(s, cfg["api"], params={"per_page": per_pagina, "page": pagina,
                                            "orderby": "date", "order": "desc",
                                            "_fields": "id,link,title,content"})
        articles = r.json()
        if not articles:
            return
        for a in articles:
            vistos += 1
            if vistos > max_articles:
                return
            cos = (a.get("content") or {}).get("rendered") or ""
            titol = (a.get("title") or {}).get("rendered") or ""
            cos = re.sub(r"<\s*(/p|br|/li|/h[1-6]|/div)\s*/?>", "\n", cos, flags=re.I)
            cos = re.sub(r"<[^>]+>", " ", cos)
            text = _html.unescape(f"{titol}.\n{cos}")
            if text.strip():
                yield a.get("link") or f"{cfg['api']}#{a.get('id')}", text
        total = r.headers.get("X-WP-TotalPages")
        if total and pagina >= int(total):
            return
        pagina += 1
        print(f"    pagina {pagina - 1}, {vistos} articles")


LECTORS = {"hf": font_hf, "wikipedia": font_wikipedia,
           "html": font_html, "wp": font_wp}


def captura(nom: str, cfg: dict, limit: int, rebaixar: bool):
    """Els blocs (origen, text) d'una font, baixats UN sol cop."""
    ruta = C.dir_font(nom) / "brut.jsonl"
    if ruta.exists() and not rebaixar:
        n = sum(1 for _ in ruta.open(encoding="utf-8"))
        data = datetime.fromtimestamp(ruta.stat().st_mtime).strftime("%d/%m %H:%M")
        print(f"  brut.jsonl ja hi es ({n:,} blocs, del {data}): no es torna a baixar "
              f"(--rebaixar per forcar)")
    else:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        part = ruta.with_suffix(".jsonl.part")
        n = 0
        with part.open("w", encoding="utf-8") as f:
            for origen, text in LECTORS[cfg["tipus"]](cfg, limit):
                f.write(json.dumps({"origen": origen, "text": text}, ensure_ascii=False) + "\n")
                n += 1
        part.replace(ruta)
        print(f"  {n:,} blocs baixats -> {ruta.name}")
    for linia in ruta.open(encoding="utf-8"):
        d = json.loads(linia)
        yield d["origen"], d["text"]


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--font", nargs="+", choices=sorted(C.FONTS), help="per defecte, totes")
    p.add_argument("--max-frases", type=int, default=400000, help="frases per font")
    p.add_argument("--max-files", type=int, default=1000000,
                   help="files a llegir d'un corpus de HuggingFace")
    p.add_argument("--max-articles", type=int, default=20000,
                   help="articles a llegir de les fonts web")
    p.add_argument("--incloure-poc-fiables", action="store_true",
                   help="hi afegeix la reserva (`bsc_ca_arn`, `wikipedia_oc`). Per defecte "
                        "queda fora: es l'ultim recurs si falten hores, no materia primera "
                        "de la mateixa qualitat")
    p.add_argument("--rebaixar", action="store_true",
                   help="torna a baixar la font encara que ja tingui `brut.jsonl` (per "
                        "exemple, per agafar articles nous)")
    p.add_argument("--llista", action="store_true", help="descriu les fonts i surt")
    args = p.parse_args()

    if args.llista:
        for nom in C.PRIORITAT:
            cfg = C.FONTS[nom]
            print(f"\n[{nom}]  tipus={cfg['tipus']}")
            print(f"  {cfg['que_es']}")
        return 0

    if C.POLITICA_XIFRES == "expandir":
        print("!! POLITICA_XIFRES = 'expandir' amb una taula de numeros PENDENT DE\n"
              "!! VALIDACIO per un parlant nadiu. Revisa `config.UNITATS`/`DESENES`\n"
              "!! abans de fiar-te'n.\n")

    filtre = fer_filtre()
    disponibles = C.PRIORITAT if args.incloure_poc_fiables else C.FONTS_FIABLES
    noms = args.font or [n for n in disponibles if C.FONTS[n]["tipus"] != "generat"]
    total = 0
    for nom in noms:
        cfg = C.FONTS[nom]
        if cfg["tipus"] == "generat":
            print(f"\n[{nom}] es genera amb 02_frases_claude.py; s'omet")
            continue
        print(f"\n[{nom}] {cfg['tipus']}")
        limit = args.max_files if cfg["tipus"] == "hf" else args.max_articles
        textos = captura(nom, cfg, limit, args.rebaixar)
        bones, dolentes = processa(nom, textos, filtre, args.max_frases)
        corpus.desa_font(C.dir_font(nom), bones, dolentes,
                         {"font": nom, "tipus": cfg["tipus"], "que_es": cfg["que_es"],
                          "min_marques": C.MIN_MARQUES, "max_foranes": C.MAX_FORANES,
                          "politica_xifres": C.POLITICA_XIFRES})
        total += sum(r["durada_est_s"] for r in bones) / 3600

    print(f"\n{'=' * 62}\nTotal d'aquesta execucio: {total:.1f} h estimades")
    print(f"Objectiu del pla: {C.HORES_OBJECTIU[0]}-{C.HORES_OBJECTIU[1]} h")
    print("Seguent: 03_neteja.py (porta de qualitat, per font) -> 04_entitats.py -> 05_respelling.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

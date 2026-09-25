#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 2: genera frases amb Claude per cobrir el que els corpus reals no donen (font `claude`)."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import claude_cli
import config as C
import corpus

TRETS = {
    "nh / lh": "mots amb `nh` i `lh`: montanha, senhor, hilh, Vielha, trabalhar, uelh, "
               "besonh, campanha, Escunhau, familha",
    "sh": "mots amb `sh`: deishar, peish, shinhau, coneisher, baish, Baish Aran, "
          "gessuda, shordant",
    "th": "mots amb `th`: eth, deth, ath, aqueth, bèth, audèth, nauèth, castèth, còth, "
          "tath, peth",
    "-tz de 2a persona": "verbs en `-tz` de segona persona del plural: auetz, podetz, "
                         "sabetz, digatz, credetz, deishatz, coneishetz, guardetz",
    "h- gascona": "mots amb `h-` inicial (la f- llatina del gascon): hèr, hemna, huec, "
                  "hilh, hèsta, hont, hred, haria, huelha, hum",
    "-n final": "mots acabats en `-n`: camin, besonh, an, plan, ben, matin, vin, jardin, "
                "carrèr principau",
    "ò tonica": "mots amb `ò` oberta: pòrt, còth, bòrda, nòsta, pòble, fòrça, "
                "Bossòst, pòt, tòrne",
    "articles i contraccions": "articles i contraccions araneses: eth, era, es, deth, "
                               "dera, des, ath, ara, as, ena, enes, entara, entath, "
                               "pera, damb, ua",
}

TOPONIMS = (
    "Vielha e Mijaran, Naut Aran, Baish Aran, Bossòst, Les, Canejan, Bausen, Vilamòs, "
    "Es Bòrdes, Arres, Vilac, Betren, Escunhau, Casarilh, Garòs, Arties, Gessa, "
    "Salardú, Unha, Bagergue, Tredòs, Baqueira, Beret, Montgarri, Aubèrt, Arró, Benós, "
    "Begós, Era Bordeta, Pònt de Rei, Pòrt dera Bonaigua, Portilhon, Malh dera Artiga, "
    "riu Garona, riu Nere, Còth de Baretja, Tunèl de Vielha")
INSTITUCIONS = (
    "Conselh Generau d'Aran, Sindic d'Aran, Ostau dera Lengua Occitana, Institut "
    "d'Estudis Aranesi, Acadèmia Aranesa dera Lengua Occitana, Aran TV, Ràdio Aran, "
    "Torisme Val d'Aran, Espitau Val d'Aran, Escòla Oficiau d'Idiòmes")

ESCENARIS_LLAVOR = [
    "obertura d'informatiu de ràdio damb es titulars deth dia",
    "cronica deth temps: nheu, vent e temperatures ena Val d'Aran",
    "informacion sus er estat des carreteres e deth Pòrt dera Bonaigua",
    "plen deth Conselh Generau d'Aran: acòrds e votacions",
    "declaracions deth Sindic d'Aran dempus d'ua reunion",
    "noticia sus era campanha d'esqui e era ocupacion ostalèra",
    "informacion sus era ensenhança der aranes enes escòles",
    "cronica d'ua hèsta populara o ua tradicion aranesa",
    "noticia d'esports: futbol, esqui de montanha o corses",
    "informacion sus obres publiques e servicis municipaus",
    "entrevista curta a un vesin o ua vesia dera Val",
    "noticia sus agricultura, ramaderia e transhumància",
    "informacion sus cultura: musica, libres e exposicions",
    "avisi de proteccion civiu per aiguats o alut",
    "noticia sus torisme e visitants ena Val d'Aran",
    "informacion sus sanitat e er Espitau Val d'Aran",
    "cronica d'ua session sus era lengua occitana",
    "noticia sus transpòrt public e autobusi ena Val",
    "informacion sus medi ambient e parcs naturaus",
    "noticia economica: comèrç, emplec e enterpreses dera Val",
]

PROMPT_ESCENARIS = """Ès un expert en informatius de ràdio dera Val d'Aran.

Genera {n} SITUACIONS o CONTEXTES distints en què un locutor o periodista parlarie en
aranès dins d'un informatiu de ràdio o television locau dera Val d'Aran.

Genera era SITUACION, non era intervencion. Exemple valid: "cronica deth temps damb
nheu enes pòrts". Exemple NON valid: "Deman nheuarà en Baqueira" -- aquerò ja ei ua
intervencion.

Varia eth tèma (politica locau, esports, cultura, meteorologia, economia, societat,
lengua, torisme, sanitat, educacion) e eth tipe d'intervencion (titular, cronica,
entrevista, declaracion, avanç informatiu).

Respon NOMÉS un objècte JSON, sense tèxte ath entorn:
{{"escenaris": ["...", "..."]}}"""

PROMPT_FRASES = """Ès un guionista d'informatius de ràdio dera VAL D'ARAN qu'escriu en
ARANÉS (era varietat gascona der occitan dera Val d'Aran, damb era sua ortografia
oficiau deth Conselh Generau d'Aran).

Genera {n} frases independentes entà aguest escenari:
  {escenari}

ORTOGRAFIA -- aquerò ei çò de mès important. Aguestes frases son eth GROUND TRUTH d'un
dataset entà ensenhar a un sistèma de reconeishença de votz a ESCRIURE aranés. Ua sola
paraula mau escrita ensenhe ua error:
- Aranés, NON lengadocian: `eth`/`era`/`es` (non `lo`/`la`/`los`), `damb` (non `amb`),
  `ua` (non `una`), `aguest` (non `aquest`), `ei` (non `es`).
- Respècta es dígrafs `nh`, `lh`, `sh`, `th` e era `h-` gascona.
- Escriu es accents damb cuedat: `è`, `ò`, `é`, `à`.

EN AGUEST LOT, carga es frases de: {tret}

Fè servir noms pròpris dera Val d'Aran tant coma pogues:
  toponims: {toponims}
  institucions: {institucions}

Regles des frases:
- Entre {min_car} e {max_car} caractèrs cadua. Que sigue ua frasa, non un paragraf.
- Acabades en punt, interrogant o exclamacion.
- NON escrigues chifres: escriu es nombres en letres (`vint-e-cinc`, non `25`).
- Varia era estructura sintactica: non comences totes es frases dera madeisha manèra.
- Non repetisques frases ne hi ajustes numeracion ne explicacions.

Respon NOMÉS un objècte JSON, sense tèxte ath entorn:
{{"frases": ["...", "..."]}}"""


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--escenaris", type=int, default=60,
                   help="escenaris a fer servir (els primers surten de la llavor fixa)")
    p.add_argument("--frases-per-crida", type=int, default=25)
    p.add_argument("--objectiu-hores", type=float, default=None,
                   help="para quan s'arribi a aquestes hores estimades")
    p.add_argument("--llavor", type=int, default=42)
    args = p.parse_args()

    rnd = random.Random(args.llavor)
    filtre = None
    import importlib.util
    spec = importlib.util.spec_from_file_location("c01", AQUI / "01_corpus.py")
    c01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c01)
    filtre = c01.fer_filtre()

    escenaris = list(ESCENARIS_LLAVOR)
    if args.escenaris > len(escenaris):
        print(f"Demanant {args.escenaris - len(escenaris)} escenaris mes a Claude...")
        try:
            escenaris += claude_cli.llista(
                PROMPT_ESCENARIS.format(n=args.escenaris - len(escenaris)),
                "escenaris", model=args.model, minim=1)
        except Exception as e:
            print(f"  avis: no s'han pogut generar mes escenaris ({e}); es fan servir "
                  f"nomes els {len(escenaris)} de la llavor")
    escenaris = escenaris[:args.escenaris]

    bones, dolentes, vistes = [], [], set()
    trets = list(TRETS.items())
    hores = 0.0
    for i, escenari in enumerate(escenaris, 1):
        nom_tret, exemples = trets[i % len(trets)]
        prompt = PROMPT_FRASES.format(
            n=args.frases_per_crida, escenari=escenari, tret=exemples,
            toponims=TOPONIMS, institucions=INSTITUCIONS,
            min_car=C.MIN_CAR, max_car=C.MAX_CAR_OBJECTIU)
        try:
            frases = claude_cli.llista(prompt, "frases", model=args.model, minim=1)
        except Exception as e:
            print(f"  [{i}/{len(escenaris)}] ha fallat: {e}")
            continue

        n_abans = len(bones)
        for frase in frases:
            net, motiu = filtre(frase)
            if net is None:
                dolentes.append({"font": "claude", "origen": escenari,
                                 "motiu": motiu, "tret": nom_tret, "text": frase[:300]})
                continue
            clau = corpus.clau_dedup(net)
            if clau in vistes:
                dolentes.append({"font": "claude", "origen": escenari,
                                 "motiu": "duplicada", "tret": nom_tret, "text": frase[:300]})
                continue
            vistes.add(clau)
            fortes, foranes = C.compta_marques(net)
            bones.append(corpus.registre("claude", len(bones), net, escenari,
                                         marques_fortes=fortes, marques_foranes=foranes,
                                         tret=nom_tret, model=args.model))
        hores = sum(r["durada_est_s"] for r in bones) / 3600
        print(f"  [{i}/{len(escenaris)}] {nom_tret:<24} +{len(bones) - n_abans}/{len(frases)} "
              f"| total {len(bones)} ({hores:.2f} h)")
        if args.objectiu_hores and hores >= args.objectiu_hores:
            print(f"  objectiu de {args.objectiu_hores} h assolit")
            break

    n_tot = len(bones) + len(dolentes)
    print(f"\nTaxa de descart de Claude: {100 * len(dolentes) / max(1, n_tot):.0f}% "
          f"({len(dolentes)}/{n_tot})")
    if n_tot and len(dolentes) / n_tot > 0.3:
        print("  ATENCIO: mes d'un terc de descarts. Mira't `descartades.jsonl`: si son "
              "de filtre dialectal, el model no escriu prou be l'aranes i aquesta font "
              "no s'hauria de fer servir sense revisio humana.")
    corpus.desa_font(C.dir_font("claude"), bones, dolentes,
                     {"font": "claude", "tipus": "generat", "model": args.model,
                      "escenaris": len(escenaris), "llavor": args.llavor,
                      "que_es": C.FONTS["claude"]["que_es"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

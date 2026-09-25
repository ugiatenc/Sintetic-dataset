#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 3: parteix les frases llargues, aplica la porta de qualitat i deduplica (net.jsonl per font)."""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import random
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ARREL / ".hf_cache"))
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import config as C
import corpus
import qualitat

INFORME = C.DIR_TEXT / "INFORME_neteja.md"


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--incloure-reserva", action="store_true",
                   help="hi afegeix les fonts de reserva (bsc, aina, wikipedia)")
    p.add_argument("--min-frequencia", type=int, default=3,
                   help="cops que ha de sortir un mot al corpus per no comptar com a "
                        "desconegut (defecte: 3)")
    p.add_argument("--max-oov", type=float, default=None,
                   help="proporcio maxima de mots desconeguts per frase. Desactivat per "
                        "defecte: el vocabulari poc vist es el que interessa, no brossa. "
                        "Nomes te sentit per a una font de la qual es desconfie")
    p.add_argument("--max-propis", type=float, default=None,
                   help="proporcio maxima de mots damb majuscula. Desactivat per defecte: "
                        "els noms propis son contingut desitjat")
    p.add_argument("--min-comes-enum", type=int, default=4,
                   help="comes minimes perque una frase pugui comptar com a enumeracio")
    p.add_argument("--max-densitat-comes", type=float, default=0.23,
                   help="comes per mot per damunt de les quals es una enumeracio i no prosa")
    p.add_argument("--mostres", type=int, default=4, help="exemples per motiu de rebuig")
    p.add_argument("--llavor", type=int, default=0)
    args = p.parse_args()
    rnd = random.Random(args.llavor)

    fonts = list(C.FONTS_FIABLES)
    if args.incloure_reserva:
        fonts += C.FONTS_RESERVA
    print(f"Fonts: {', '.join(fonts)}"
          + ("" if args.incloure_reserva else "   (la reserva queda fora; "
             "--incloure-reserva per afegir-la)"))

    per_font: dict[str, list] = {}
    for font in fonts:
        registres = corpus.llegeix_jsonl(C.dir_font(font) / "capturat.jsonl")
        if not registres:
            print(f"  {font:<16} (buida)")
            continue
        registres, n_cor, n_del = corpus.aplicar_correccions(C.dir_font(font), registres)
        per_font[font] = registres
        print(f"  {font:<16} {len(registres):>8,}"
              + (f"  ({n_cor} corregides, {n_del} eliminades a ma)" if n_cor or n_del else ""))
    if not per_font:
        raise SystemExit("Cap font. Executa 01_corpus.py primer.")

    frases_cv = set()
    if C.MANIFEST_CV.exists():
        frases_cv = {corpus.clau_dedup(json.loads(l)["text"])
                     for l in C.MANIFEST_CV.read_text(encoding="utf-8").splitlines()
                     if l.strip()}
    print(f"Common Voice: {len(frases_cv):,} frases excloses (es tornen a comprovar "
          f"despres de partir)")

    text_lexic = [r["raw_text"] for rs in per_font.values() for r in rs]
    for font in C.FONTS_RESERVA:
        text_lexic += [r["raw_text"]
                       for r in corpus.llegeix_jsonl(C.dir_font(font) / "capturat.jsonl")]
    if C.MANIFEST_CV.exists():
        text_lexic += [json.loads(l)["text"]
                       for l in C.MANIFEST_CV.read_text(encoding="utf-8").splitlines()
                       if l.strip()]
    lexic = qualitat.construeix_lexic(text_lexic, args.min_frequencia)
    print(f"\nLexic: {len(lexic):,} mots damb frequencia >= {args.min_frequencia}, "
          f"de {len(text_lexic):,} frases (corpus + reserva + Common Voice)")

    inicials, interiors = collections.Counter(), collections.Counter()
    rx_ini = re.compile(r"(?:^|[.!?]\s+)([A-ZÀ-Ü]{2,})\b")
    rx_int = re.compile(r"(?<=[a-zàèéíòóúïü,;:])\s+([A-ZÀ-Ü]{2,})\b")
    for t in text_lexic:
        inicials.update(rx_ini.findall(t))
        interiors.update(rx_int.findall(t))
    sigles = {s for s, n in interiors.items() if n >= 2 and n > inicials[s]}
    print(f"Sigles conegudes (mes a mig de frase que al principi): {len(sigles):,}")
    porta = qualitat.PortaQualitat(lexic=lexic, sigles=sigles, max_oov=args.max_oov,
                                   max_propis=args.max_propis,
                                   min_comes_enum=args.min_comes_enum,
                                   max_densitat_comes=args.max_densitat_comes)
    nets, rebutjats, mostres = {}, {}, collections.defaultdict(list)
    for font, registres in per_font.items():
        bons, dolents = [], []
        for r in registres:
            def _fora(motiu, text):
                """Registra una frase descartada amb el motiu."""
                dolents.append({**{c: r[c] for c in ("id", "font", "origen")},
                                "motiu": motiu, "text": text[:300]})

            desenganxat = re.sub(r"([.!?])([A-ZÀ-Ü])", r"\1 \2", r["raw_text"])
            desenganxat = re.sub(r'([.!?])(["”»])(["“«])(?=[A-ZÀ-Ü])', r"\1\2 \3", desenganxat)
            frases_reals = corpus.a_frases(desenganxat) or [r["raw_text"]]
            trossos = []
            for frase in frases_reals:
                parts = corpus.dividir_llarga(frase, C.MAX_CAR_OBJECTIU, C.MIN_CAR,
                                              max_dur=C.MAX_CAR)
                if not parts:
                    _fora(f"massa llarga sense on tallar ({len(frase)} car)", frase)
                trossos += parts
            for k, tros in enumerate(trossos, 1):
                ident = r["id"] if len(trossos) == 1 else f"{r['id']}-{k}"
                if corpus.clau_dedup(tros) in frases_cv:
                    _fora("frase de Common Voice", tros)
                    continue
                if not (C.MIN_MOTS <= len(tros.split()) <= C.MAX_MOTS
                        and C.MIN_CAR <= len(tros) <= C.MAX_CAR):
                    _fora(f"llargada {len(tros)} car / {len(tros.split())} mots", tros)
                    continue
                if C.compta_marques(tros)[1] > C.MAX_FORANES:
                    _fora("marca forana", tros)
                    continue
                if C.presegmentada(font) and C.castella_tou(tros) > C.MAX_CASTELLA_TOU:
                    _fora(f"castella (regla tova hf, {C.castella_tou(tros)} senyals)", tros)
                    continue
                tros = C.expandir_unitats(tros)
                net, motiu = porta(tros, presegmentat=C.presegmentada(font))
                if net is None:
                    _fora(motiu, tros)
                    mostres[motiu.split(" (")[0]].append((font, tros))
                    continue
                if corpus.clau_dedup(net) in frases_cv:
                    _fora("frase de Common Voice", net)
                    continue
                if not (C.MIN_MOTS <= len(net.split()) <= C.MAX_MOTS
                        and C.MIN_CAR <= len(net) <= C.MAX_CAR):
                    _fora(f"llargada despres de normalitzar ({len(net)} car)", net)
                    continue
                bons.append({**r, "id": ident, "raw_text": net, "n_car": len(net),
                             "n_mots": len(net.split()),
                             "durada_est_s": corpus.durada_estimada(net),
                             "hash": corpus.hash_text(net),
                             **({"modificada": True} if net != r["raw_text"] else {})})
        nets[font], rebutjats[font] = bons, dolents

    plans = [r for rs in nets.values() for r in rs]
    plans, duplicats = corpus.deduplicar(plans, C.PRIORITAT)
    for r in duplicats:
        rebutjats[r["font"]].append(r)
    n_dup = len(duplicats)
    per_font_net = collections.defaultdict(list)
    for r in plans:
        per_font_net[r["font"]].append(r)

    for font in nets:
        d = C.dir_font(font)
        bones = per_font_net.get(font, [])
        corpus.escriu_jsonl(d / "net.jsonl", bones)
        corpus.desa_descartades(d, 3, rebutjats[font])
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        hores = sum(r["durada_est_s"] for r in bones) / 3600
        meta["neteja"] = {"frases": len(bones), "descartades": len(rebutjats[font]),
                          "hores_estimades_tts": round(hores, 2),
                          "min_frequencia": args.min_frequencia, "max_oov": args.max_oov,
                          "data": datetime.date.today().isoformat()}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"  {font:<18} {len(bones):>7,} netes, {len(rebutjats[font]):>6,} descartades"
              f"  -> {d}")

    print(f"\nDuplicats trets: {n_dup:,}")
    print(f"\n{'motiu de rebuig':<34}{'frases':>10}{'%':>8}")
    for motiu, n, pct in porta.informe():
        print(f"  {motiu:<32}{n:>10,}{pct:>7.1f}%")

    print(f"\n{'font':<16}{'abans':>10}{'despres':>10}{'passen':>9}{'hores':>9}")
    files = []
    for font in nets:
        a, d = len(per_font[font]), len(per_font_net.get(font, []))
        h = sum(r["durada_est_s"] for r in per_font_net.get(font, [])) / 3600
        print(f"  {font:<14}{a:>10,}{d:>10,}{100 * d / max(1, a):>8.0f}%{h:>9.1f}")
        files.append(f"| `{font}` | {a:,} | {d:,} | {100 * d / max(1, a):.0f} % | {h:.1f} |")
    total_h = sum(r["durada_est_s"] for r in plans) / 3600
    print(f"\n  TOTAL NET: {len(plans):,} frases · {total_h:.1f} h estimades")

    files_motiu = "\n".join(f"| {m} | {n:,} | {p:.1f} % |"
                            for m, n, p in porta.informe() if m != "ACCEPTADA")
    blocs = []
    for motiu, ex in sorted(mostres.items(), key=lambda kv: -len(kv[1])):
        tria = rnd.sample(ex, min(args.mostres, len(ex)))
        blocs.append(f"**{motiu}** ({len(ex):,})\n\n" +
                     "\n".join(f"- `[{f}]` {t[:150]}" for f, t in tria))
    INFORME.write_text(f"""# Neteja del corpus

{len(plans):,} frases netes · **{total_h:.1f} h** estimades.
Fonts: {', '.join(nets)}{'' if args.incloure_reserva else '  (nomes les fiables)'}.

## Per font

| Font | Abans | Despres | Passen | Hores |
|---|---|---|---|---|
{chr(10).join(files)}

Deduplicacio: {n_dup:,} frases repetides tretes, quedant-se sempre la copia de la font mes
fiable (ordre: {' > '.join(C.PRIORITAT)}).

## Per que ha caigut cada frase

| Motiu | Frases | % del total |
|---|---|---|
{files_motiu}

## Exemples de cada motiu

Son aqui perque es puga jutjar si una regla es massa dura. Si un motiu es emporta frases
bones, el llindar es a `src/qualitat.py` o als arguments d'aquest script.

{chr(10).join(chr(10) + b for b in blocs)}

## Les regles

La porta de qualitat treballa damb **llista blanca**: `qualitat.NOMES_ALFABET` diu quins
caracters pot tenir una frase aranesa dicible i tot el que no hi siga fa caure la frase.
A l'auditoria del corpus hi havia 218 caracters distints fora de l'alfabet; una llista
negra sempre hauria anat tard.

Abans de decidir res, es normalitza el recuperable: `‟`, `´`, `` ` `` i les cometes
tipografiques passen a apostrof, els guions llargs a guio normal. Nomes la `‟` sortia en
4.731 frases que d'altra manera s'haurien perdut per un problema de codificacio.

La resta de regles son de forma -- sigles, majuscula inicial, puntuacio final,
abreviatura tallada, marca de llista, densitat de comes i de noms propis -- llevat d'una:
**mots desconeguts**, que mira quina proporcio de mots de la frase surt menys de
{args.min_frequencia} cops a tot el corpus. No es un diccionari (no n'hi ha cap d'aranes en
obert) sino la distribucio del corpus mateix, i serveix per a allo que la forma no veu:
fugues d'altres llengues, llistes de noms i text malmes.

## Coses que van caure per error i ja no

Calibrant-ho damb el corpus real van sortir tres regles massa dures, totes tres arreglades:

- `d'Aran`, `d'Oilèu` queien per "sigla" perque tenen majuscula a dins. L'apostrof separa
  un clitic: ara el mot es parteix abans de mirar-ho. Eren 8.000 frases.
- "Sense compdar eth cap." queia per "acaba en abreviatura": `cap`, `vol`, `art` i `op`
  eren a la llista d'abreviatures i en aranes son mots corrents.
- `en·honsat` queia per caracter fora de l'alfabet. La interpunta separa `n`+`h` perque no
  es llegisquen com el digraf `nh`: es grafia aranesa correcta i ara hi es.
""", encoding="utf-8")
    print(f"\n  <carpeta de cada font>/net.jsonl + descartades.jsonl")
    print(f"  {INFORME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

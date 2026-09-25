#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 5: fusiona les fonts, deduplica, limita plantilles i escriu el tts_text (corpus.jsonl + INFORME.md)."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
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
import phonetics
import qualitat
import respelling

PLURALS = phonetics.dir_diccionaris(C.IDIOMA) / "plurals_aranes.json"

SORTIDA = C.DIR_TEXT / "corpus.jsonl"
INFORME = C.DIR_TEXT / "INFORME.md"
MIDA_LOT_PLURALS = 200

PROMPT_PLURALS = """Ès un expert en gramatica der aranés (occitan gascon dera Val d'Aran).

Te doni mots aranesi acabadi en -ns. Entà cadun, dides se era -s finau ei ua MARCA DE
PLURAL (un nom o adjectiu en plurau, coma `camins`, `vesins`) o non ac ei (adverbis,
preposicions e auti mots qu'acaben en -ns sense èster plurau, coma `mens`, `abans`,
`sens`).

Respon NOMÉS un objècte JSON, sense tèxte ath entorn:
{{"mots": {{"mot": true, "mot": false}}}}
true = ei plurau. false = non ac ei. Include TOTI es mots dera lista.

Mots:
{llista}"""


def carrega_plurals() -> dict:
    """Decisions de plurals en -ns desades."""
    if not PLURALS.exists():
        return {}
    return {k: v for k, v in json.loads(PLURALS.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}


def desa_plurals(decisions: dict, model: str) -> None:
    """Desa les decisions de plurals amb metadades."""
    plurals = sorted(m for m, v in decisions.items() if v)
    PLURALS.parent.mkdir(parents=True, exist_ok=True)
    PLURALS.write_text(json.dumps(
        {"_meta": {"descripcio": "Mots aranesos acabats en -ns: true = la -s es marca de "
                                 "plural i la -n emmudeix (camins -> camís). Decidit un "
                                 "cop per mot amb un LLM; el codi nomes aplica la decisio.",
                   "model": model, "plurals": len(plurals),
                   "no_plurals": len(decisions) - len(plurals)},
         **dict(sorted(decisions.items()))}, ensure_ascii=False, indent=2), encoding="utf-8")


def classifica_plurals(textos, model: str) -> dict:
    """Decideix els mots en -ns que encara no estiguin decidits."""
    decisions = carrega_plurals()
    candidats = respelling.candidats_plural(textos)
    nous = [m for m in candidats if m not in decisions]
    print(f"  mots acabats en -ns: {len(candidats)} | ja decidits: "
          f"{len(candidats) - len(nous)} | a preguntar: {len(nous)}")
    for i in range(0, len(nous), MIDA_LOT_PLURALS):
        lot = nous[i:i + MIDA_LOT_PLURALS]
        try:
            resposta = claude_cli.json_de_resposta(
                claude_cli.crida(PROMPT_PLURALS.format(llista="\n".join(lot)), model=model))
            d = resposta.get("mots", resposta)
            decisions.update({m: bool(v) for m, v in d.items() if m in set(lot)})
            falten = [m for m in lot if m not in decisions]
            print(f"    lot {i // MIDA_LOT_PLURALS + 1}: {len(lot) - len(falten)}/{len(lot)}"
                  + (f" (no tornats: {falten[:4]})" if falten else ""))
        except Exception as e:
            print(f"    lot {i // MIDA_LOT_PLURALS + 1} ha fallat: {e}")
    if nous:
        desa_plurals(decisions, model)
    return decisions


SENTINELLA = "ẅ"

TIPUS_SIGLA: dict = {}
COMUNS: set = set()


SIGLA = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")
CLITIC = re.compile(r"[^\W\d_]{1,2}['\u2019]")


def _es_sigla(mot: str) -> bool:
    """Majuscula a dins o tot majuscules."""
    for tros in re.split(r"['\u2019]", mot.strip(".,;:!?()\"'")):
        lletres = [c for c in tros if c.isalpha()]
        if len(lletres) < 2:
            continue
        if all(c.isupper() for c in lletres):
            return True
        if any(c.isupper() for c in lletres[1:]) and any(c.islower() for c in lletres):
            return True
    return False


NO_ES_NOM = {  # mots que, davant d'un roma, no son cap nom (no es fa l'ordinal)
    "eth", "era", "es", "er", "en", "ena", "enes", "ath", "ara", "as", "deth", "dera",
    "des", "der", "aguest", "aguesta", "aguesti", "aguestes", "un", "ua", "uns", "ues",
    "damb", "per", "tà", "entà", "que", "se", "non", "e", "o", "a", "de", "ja", "mès",
    "tanben", "pro", "pas", "lo", "la", "sègle", "segle", "siègle", "capítol", "capitol",
    "libre", "tòme", "tom", "part", "volum", "títol", "títou", "titol", "annèx", "annex",
    "apartat", "punt", "article", "paragraf", "pagina", "edicion", "temporada", "fasa",
    "etapa", "nivèu", "grau", "categoria", "division", "classe", "sèria", "version",
    "numèro", "interreg", "europèu", "programa", "plan", "legislatura", "congrès", "jòcs",
    "concors", "trofèu", "copa", "mòstra", "salon", "trobada", "jornades", "jornada",
    "olimpiada", "festivau", "certamen", "setmana", "dia", "guèrra", "republica", "val",
    "literaris", "internacionau", "nacionau", "generau", "mondiau", "premi", "prèmi",
    "prèmis", "premis", "gèr", "hereuèr", "març", "abriu", "mai", "junh", "juriòl", "agost",
    "seteme", "octobre", "noveme", "deseme",
}
REINES = {  # noms de reina: l'ordinal va en femeni (Isabel II -> Isabel dusaua)
    "Isabel", "Isabèl", "Isabela", "Elisabet", "Elisabeth", "Elizabeth", "Joana", "Juana",
    "Maria", "María", "Cristina", "Victòria", "Victoria", "Caterina", "Catalina",
    "Margarida", "Margarita", "Anna", "Ana", "Beatriu", "Beatriz", "Urraca", "Peronella",
    "Petronila", "Constança", "Constanza", "Blanca", "Sança", "Sancha", "Leonor", "Elionor",
    "Teresa", "Berenguera", "Ermessenda", "Ermengarda", "Matilde", "Violant", "Iolanda",
    "Elena", "Helena", "Sofia", "Sofía", "Carlota", "Carolina", "Frederica", "Adelaida",
    "Amàlia", "Amalia", "Lluïsa", "Luisa", "Loïsa", "Barbarita",
}
CAPCALERES = {"sègle", "segle", "siègle", "capítol", "capitol", "libre", "tòme", "tom", "part",  # darrere d'aquests el roma es diu amb el cardinal (capitol V -> capitol cinc)
              "volum", "títol", "títou", "titol", "annèx", "annex", "apartat", "article",
              "pagina", "paragraf", "punt", "numèro", "fasa", "etapa", "division",
              "categoria", "classe", "sèria", "grau", "nivèu", "version", "acte", "escèna"}
ROMA_DARRERE_MOT = re.compile(
    r"\b([^\W\d_]+)(\s+)([IVXLCDM]{1,7})(?![\w'’-])(?!\.\s+[A-ZÀ-Ü])")


def _romans_en_context(text: str) -> str:
    """`Jaime II` -> `Jaime dusau`, `Isabel II` -> `Isabel dusaua`, `capítol V` -> `capítol cinc`."""
    def _sub(m):
        """Substitueix un roma darrere d'un mot per l'ordinal o el cardinal."""
        mot, espai, roma = m.group(1), m.group(2), m.group(3)
        n = C.valor_roma(roma)
        if n is None:
            return m.group(0)
        if mot.lower() in CAPCALERES:
            paraules = C.en_lletres(n)
        elif (mot[:1].isupper() and mot[1:].islower() and n <= 50
              and mot.lower() not in NO_ES_NOM):
            paraules = C.ordinal(n, femeni=mot in REINES)
        else:
            return m.group(0)
        return f"{mot}{espai}{paraules}" if paraules else m.group(0)
    return ROMA_DARRERE_MOT.sub(_sub, text)


def context_proteccio(frases_raw: list[str]) -> tuple[set, dict]:
    """Carrega els diccionaris desats i el context de majuscules (TIPUS_SIGLA, COMUNS); torna (plurals, proteccio) tal com els fa servir el pas 5."""
    global TIPUS_SIGLA, COMUNS
    d = phonetics.dir_diccionaris(C.IDIOMA)
    llegeix = lambda nom: ({k: v for k, v in json.loads((d / nom).read_text(encoding="utf-8")).items() if not k.startswith("_")}
                           if (d / nom).exists() else {})
    TIPUS_SIGLA = llegeix("tipus_sigles_aranes.json")
    minuscules = collections.Counter(m.lower() for t in frases_raw for m in SIGLA.findall(t) if m[:1].islower())
    COMUNS = {w for w, n in minuscules.items() if n >= 3}
    plurals = {m for m, v in llegeix("plurals_aranes.json").items() if v}
    noms = llegeix("noms_propis_aranes.json")
    forans = {k: (k if v == "foran" else respelling.reescriu(k, nomes_digrafs=True))
              for k, v in noms.items() if v in ("foran", "foran_grafia")}
    desconeguts = {k: k for k in llegeix("noms_desconeguts_protegits.json")}
    dicc = phonetics.carregar(phonetics.ruta_diccionari(C.IDIOMA, "treball")).pla()
    return plurals, {**desconeguts, **forans, **dicc}


def reescriu_protegit(text: str, plurals: set, diccionari: dict, traca: dict) -> str:
    """Aplica les normes ortografiques SENSE tocar les entitats del diccionari."""
    taps: list[str] = []
    tapat = _tapar(_romans_en_context(text), diccionari, taps)
    reescrit = respelling.reescriu(tapat, traca=traca, plurals=plurals)
    for i, fonetica in enumerate(taps, 1):
        reescrit = reescrit.replace(SENTINELLA + "z" * i, fonetica, 1)
    return reescrit


def _tapar(text: str, diccionari: dict, taps: list) -> str:
    """Substitueix cada entitat del text per un sentinella numerat, en ordre."""
    titular = C.es_titular(text)

    def _tros(tros: str) -> str:
        """Resol un token: mot en majuscules, roma, sigla o nom protegit."""
        roma = C.valor_roma(tros)
        if (tros.isupper() and len(tros) > 1 and roma is None
                and (TIPUS_SIGLA.get(tros) == "mot"
                     or ((titular or tros not in diccionari) and tros.lower() in COMUNS))):
            return tros.lower()
        fonetica = diccionari.get(tros)
        if fonetica is None and roma is not None and C.es_ordinal_roma(tros):
            fonetica = C.fonetica_roma(tros)
        if fonetica is None and _es_sigla(tros):
            fonetica = tros
        if fonetica is None:
            return tros
        if roma is not None and fonetica != tros:
            fonetica = respelling.reescriu(fonetica)
        taps.append(fonetica)
        return SENTINELLA + "z" * len(taps)

    def _sub(m):
        """Tapa un token (o la part despres de l'apostrof) amb una sentinella."""
        tok = m.group(0)
        if tok in diccionari or not CLITIC.match(tok):
            return _tros(tok)
        return "".join(t if t in ("'", "\u2019") else _tros(t)
                       for t in re.split(r"(['\u2019])", tok))

    return SIGLA.sub(_sub, text)


_NUMERAL = re.compile(
    r"\b(?:" + "|".join(sorted((corpus.normalitzar(n) for n in qualitat.NUMERALS),
                               key=len, reverse=True)) + r")\b")


def esquelet(text: str) -> str:
    """Forma d'una frase sense els numerals: `eth conselh a aprovat # euros` (minuscules."""
    t = _NUMERAL.sub("#", corpus.clau_dedup(text))
    t = re.sub(r"#(?:[\s\-e]+#)*", "#", t)
    return re.sub(r"\s+", " ", t).strip()


def noms_desconeguts(registres, classificats: set, plurals: set) -> dict:
    """Noms propis que ningu no ha classificat: es deixen intactes per defecte."""
    minuscules, propis = collections.Counter(), collections.Counter()
    for r in registres:
        t = r["raw_text"]
        for mot in SIGLA.findall(t):
            if mot[:1].islower():
                minuscules[mot.lower()] += 1
            elif not _es_sigla(mot) and not t.startswith(mot):
                propis[mot] += 1
    comuns = {w for w, n in minuscules.items() if n >= 3}
    return {m: m for m in propis
            if m not in classificats and m.lower() not in comuns
            and respelling.reescriu(m, plurals=plurals) != m}


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--fonts", nargs="+", metavar="FONT",
                   help="nomes aquestes fonts entren al corpus final (per defecte, totes "
                        "les que hi ha a net/). Exemple: --fonts conselharan arannoticies "
                        "institut_estudis bsc_ca_arn")
    p.add_argument("--sense-fonts", nargs="+", metavar="FONT", default=[],
                   help="fonts que NO entren al corpus final. La decisio de quines fonts "
                        "poc fiables es fan servir es pren AQUI i queda a l'INFORME: "
                        "`--incloure-reserva` al pas 3 nomes les neteja, no obliga a "
                        "fer-les servir")
    p.add_argument("--no-protegir-desconeguts", action="store_true",
                   help="aplica les regles tambe als noms propis que 04 no ha "
                        "classificat (per defecte queden intactes)")
    p.add_argument("--sense-llm", action="store_true",
                   help="no classifica els -ns nous: els que no estiguin decidits es "
                        "deixen intactes (val mes `camins` que `mens`->`més`)")
    args = p.parse_args()

    tots, n_cor, n_del = [], 0, 0
    print("Fonts:")
    for font in C.PRIORITAT:
        if args.fonts and font not in args.fonts:
            continue
        if font in args.sense_fonts:
            print(f"  {font:<16} EXCLOSA (--sense-fonts)")
            continue
        dir_font = C.dir_font(font)
        registres = corpus.llegeix_jsonl(C.dir_font(font) / "net.jsonl")
        if not registres:
            print(f"  {font:<16} (buida o no generada)")
            continue
        registres, c, d = corpus.aplicar_correccions(dir_font, registres)
        n_cor += c
        n_del += d
        tots += registres
        print(f"  {font:<16} {len(registres):>7} frases"
              + (f"  ({c} corregides, {d} eliminades a ma)" if c or d else ""))
    if not tots:
        raise SystemExit("No hi ha cap font. Executa 01_corpus.py primer.")

    tots, duplicats = corpus.deduplicar(tots, C.PRIORITAT)
    n_dup = len(duplicats)
    descartades_5: dict = collections.defaultdict(list)
    for r in duplicats:
        descartades_5[r["font"]].append(r)
    print(f"\nDuplicats entre fonts: {n_dup} (es queda la copia de la font mes fiable)")

    per_esquelet: collections.Counter = collections.Counter()
    conservades = []
    for r in tots:
        e = esquelet(r["raw_text"])
        per_esquelet[e] += 1
        if per_esquelet[e] > C.MAX_PER_ESQUELET:
            descartades_5[r["font"]].append({
                "id": r["id"], "font": r["font"],
                "motiu": f"plantilla repetida (mes de {C.MAX_PER_ESQUELET} copies del mateix esquelet)",
                "text": r["raw_text"]})
        else:
            conservades.append(r)
    n_plantilla = len(tots) - len(conservades)
    tots = conservades
    print(f"Plantilles repetides (mes de {C.MAX_PER_ESQUELET} copies d'un esquelet): "
          f"{n_plantilla} frases fora")

    if C.MANIFEST_CV.exists():
        cv = {corpus.clau_dedup(json.loads(l)["text"])
              for l in C.MANIFEST_CV.read_text(encoding="utf-8").splitlines() if l.strip()}
        colades = [r for r in tots if corpus.clau_dedup(r["raw_text"]) in cv]
        if colades:
            raise SystemExit(f"ERROR: {len(colades)} frases de Common Voice al corpus: "
                             f"{[r['id'] for r in colades[:3]]}")
        print("Cap frase de Common Voice al corpus: correcte")

    print("\nClassificador de plurals:")
    if args.sense_llm:
        decisions = carrega_plurals()
        print(f"  --sense-llm: es fan servir els {len(decisions)} ja decidits")
    else:
        decisions = classifica_plurals([r["raw_text"] for r in tots], args.model)
    plurals = {m for m, v in decisions.items() if v}
    print(f"  {len(plurals)} plurals / {len(decisions)} mots decidits")

    dicc = phonetics.carregar(phonetics.ruta_diccionari(C.IDIOMA, "treball")).pla()
    global TIPUS_SIGLA, COMUNS
    ruta_tipus = phonetics.dir_diccionaris(C.IDIOMA) / "tipus_sigles_aranes.json"
    if ruta_tipus.exists():
        TIPUS_SIGLA = {k: v for k, v in json.loads(ruta_tipus.read_text(encoding="utf-8")).items()
                       if not k.startswith("_")}
    minuscules = collections.Counter(m.lower() for r in tots
                                     for m in SIGLA.findall(r["raw_text"]) if m[:1].islower())
    COMUNS = {w for w, n in minuscules.items() if n >= 3}

    ruta_noms = phonetics.dir_diccionaris(C.IDIOMA) / "noms_propis_aranes.json"
    noms = {}
    if ruta_noms.exists():
        noms = {k: v for k, v in json.loads(ruta_noms.read_text(encoding="utf-8")).items()
                if not k.startswith("_")}
    forans = {k: (k if v == "foran" else respelling.reescriu(k, nomes_digrafs=True))
              for k, v in noms.items() if v in ("foran", "foran_grafia")}
    desconeguts = {} if args.no_protegir_desconeguts else noms_desconeguts(tots, set(noms), plurals)
    ruta_cua = phonetics.dir_diccionaris(C.IDIOMA) / "noms_desconeguts_protegits.json"
    ruta_cua.write_text(json.dumps({"_meta": {"descripcio": "Noms propis que 04 no ha classificat "
                                    "(menys de 3 ocurrencies) i que 05 deixa intactes per defecte. "
                                    "Generat per 05_respelling.py; no editar: classifica'ls a "
                                    "noms_propis_aranes.json.", "noms": len(desconeguts)},
                                    **{k: "desconegut" for k in sorted(desconeguts)}},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    proteccio = {**desconeguts, **forans, **dicc}
    n_graf = sum(1 for v in noms.values() if v == "foran_grafia")
    print(f"Proteccio: {len(dicc)} entrades de diccionari fonetic + {len(forans)} noms "
          f"forans ({n_graf} damb els digrafs transcrits) + {len(desconeguts)} noms sense "
          f"classificar (intactes)")
    if not dicc and not forans:
        print("  (04_entitats.py encara no s'ha executat)")
    traca: dict = {}
    for r in tots:
        r["tts_text"] = reescriu_protegit(r["raw_text"], plurals, proteccio, traca)

    massa_llargues = [r for r in tots
                      if corpus.durada_estimada(r["tts_text"], corpus.CAR_PER_SEGON_LENT)
                         > C.MAX_CAR / corpus.CAR_PER_SEGON_LENT]
    if massa_llargues:
        print(f"\n{len(massa_llargues)} frases passen de "
              f"{C.MAX_CAR / corpus.CAR_PER_SEGON_LENT:.0f} s (a velocitat lenta) un cop "
              f"reescrites; es treuen")
        fora = {id(r) for r in massa_llargues}
        tots = [r for r in tots if id(r) not in fora]
        for r in massa_llargues:
            descartades_5[r["font"]].append({
                "id": r["id"], "font": r["font"],
                "motiu": f"massa llarga un cop reescrita "
                         f"({corpus.durada_estimada(r['tts_text']):.1f} s)",
                "text": r["raw_text"], "tts_text": r["tts_text"]})
    for r in tots:
        r["durada_est_s"] = corpus.durada_estimada(r["tts_text"])
    for font in C.FONTS:
        corpus.desa_descartades(C.dir_font(font), 5, descartades_5.get(font, []))
    if descartades_5:
        print(f"  {sum(len(v) for v in descartades_5.values()):,} descartades al pas 5 "
              f"-> descartades.jsonl de {len(descartades_5)} fonts")

    print(f"\nRegles disparades: {dict(sorted(traca.items(), key=lambda kv: -kv[1]))}")

    tots.sort(key=lambda r: (r["font"], r["id"]))
    corpus.escriu_jsonl(SORTIDA, tots)
    res = corpus.resum(tots)

    hist = corpus.histograma([r["durada_est_s"] for r in tots], [0, 2, 4, 6, 8, 10, 15, 20])
    files_font = "\n".join(
        f"| `{f}` | {res['per_font'][f]:,} | {res['hores_per_font'][f]:.1f} | "
        f"{C.FONTS[f]['que_es'][:110]}... |" for f in res["per_font"])
    files_hist = "\n".join(f"| {a} s | {n:,} | {p:.1f} % |" for a, n, p in hist)
    baix, alt = C.HORES_OBJECTIU
    h = res["hores_estimades_tts"]
    estat = ("dins de l'objectiu" if baix <= h <= alt else
             f"FALTEN {baix - h:.0f} h per arribar a l'objectiu" if h < baix else
             f"{h - alt:.0f} h per damunt de l'objectiu")

    INFORME.write_text(f"""# Corpus de text aranes

{res['frases']:,} frases · {res['caracters']:,} caracters · **{h:.1f} h** estimades de TTS
({estat}; objectiu {baix}-{alt} h).

Les hores son una ESTIMACIO a partir dels caracters, a {corpus.CAR_PER_SEGON_TTS} car/s.
Aquesta xifra es la velocitat d'OmniVoice mesurada al pilot de 200 frases (0,45 h d'audio),
no la d'una persona: el TTS parla un {100 * (corpus.CAR_PER_SEGON_TTS / corpus.CAR_PER_SEGON_REAL - 1):.0f} % mes
rapid que els lectors de Common Voice ({corpus.CAR_PER_SEGON_REAL} car/s), i fer servir la
velocitat humana sobreestimaria les hores en aquesta proporcio.

## Per font

| Font | Frases | Hores | Que es |
|---|---|---|---|
{files_font}

## Distribucio de durades

| Durada estimada | Frases | % |
|---|---|---|
{files_hist}

Durada mitjana: {res['durada_mitjana_s']:.1f} s. De referencia, Common Voice `oc` te una
mediana de 4,9 s i un p95 de 8,1 s.

**Per que aquestes llargades.** Whisper te dos limits durs: la finestra de l'encoder es
de 30 s i el decoder no passa de 448 tokens. Per sota d'aixo, entrenar NOMES damb frases
curtes degrada la transcripcio d'audio llarg i la prediccio de marques de temps; la
solucio publicada es CONCATENAR frases fins a 30-40 s en construir el dataset. Aqui les
frases son de la mida de Common Voice -- que es la unitat comoda per revisar i per
reescriure -- i la concatenacio es deixa per al pas de generacio, que ja tindra els
audios i les podra ajuntar damb les seves marques de temps.

## Reescriptura

Regles disparades sobre el corpus sencer: `{dict(sorted(traca.items(), key=lambda kv: -kv[1]))}`

Cada registre porta `raw_text` (aranes real, el ground truth) i `tts_text` (el que llegira
el TTS). `tts_text` es una funcio pura de `raw_text`: es pot tornar a calcular i comprovar.

Plurals decidits: {len(decisions)} mots, {len(plurals)} marcats com a plural.

## Com corregir-hi coses a ma

No editis `frases.jsonl`: tornar a executar `01_corpus.py` el regenera i s'emportaria els
canvis. Posa un `correccions.jsonl` a la carpeta de la font, damb una linia per canvi:

```json
{{"id": "bsc_ca_arn-0001234", "raw_text": "Eth tèxte corregit."}}
{{"id": "aina_es_arn-0000042", "elimina": true}}
```

`03_respelling.py` les aplica en fusionar. En aquesta execucio: {n_cor} corregides,
{n_del} eliminades.

Per saber QUE cal corregir, mira't `descartades.jsonl` de cada font: hi ha el motiu i el pas de
cada descart, i es la manera de veure si un llindar es massa dur.

## Limitacions conegudes

- **La politica de xifres es `{C.POLITICA_XIFRES}`.** La taula de numeros aranesos de
  `config.py` esta PENDENT DE VALIDACIO per un parlant nadiu, i mentre no ho estigui val
  mes descartar les frases damb xifres que omplir el corpus de numeros mal escrits. Un
  cop validada, posar `POLITICA_XIFRES = "expandir"` recupera una part important del
  corpus (a les proves, ~21 % dels descarts eren per xifres).
- **El filtre dialectal deixa passar un 0,5 % de lengadocia i perd un 19 % d'aranes bo**
  (calibrat damb 472 frases araneses, 2.000 catalanes i 1.183 lengadocianes). Els
  llindars son a `config.py`.
- **`aina_es_arn` es en bona part traduccio automatica** (Apertium). Pot arrossegar
  errors sistematics que cap filtre de grafia no veu.
- **`claude` es text generat**, no aranes atestat. Cobreix el domini i el vocabulari que
  els corpus reals no donen, pero l'hauria de revisar un nadiu abans de pesar gaire al
  dataset.
""", encoding="utf-8")

    print(f"\n{'=' * 62}")
    print(f"{res['frases']:,} frases · {h:.1f} h estimades · {estat}")
    print(f"  {SORTIDA}")
    print(f"  {INFORME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

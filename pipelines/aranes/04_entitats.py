#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 4: classifica sigles i noms propis amb Claude (amb context) i escriu els diccionaris d'entitats."""

from __future__ import annotations

import argparse
import collections
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

import claude_cli
import config as C
import corpus
import phonetics
import respelling


MIDA_LOT = 40
FINESTRA = 70
MOT = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")

NOMS_LLETRA = C.NOMS_LLETRA

ROMANA = C.ROMANA
valor_roma = C.valor_roma
es_ordinal = C.es_ordinal_roma
DIGRAF = re.compile(r"lh|nh|sh", re.IGNORECASE)
CLITIC = re.compile(r"[^\W\d_]{1,2}['\u2019]")
PLURAL_SIGLA = re.compile(r"(?<=[A-ZÀ-Ü])['\u2019]?s$")


def es_sigla(mot: str) -> bool:
    """Si un token te forma de sigla (majuscules, amb apostrof o punts)."""
    for tros in re.split(r"['’]", mot.strip(".,;:!?()\"'")):
        lletres = [c for c in tros if c.isalpha()]
        if len(lletres) < 2:
            continue
        if all(c.isupper() for c in lletres):
            return True
        if any(c.isupper() for c in lletres[1:]) and any(c.islower() for c in lletres):
            return True
    return False


def fonetica_sigla(sigla: str, tipus: str) -> str | None:
    """Com es diu una sigla segons el seu tipus (lletrejada, acronim, mot, romana)."""
    if tipus == "romana":
        return C.fonetica_roma(sigla)
    if tipus == "mot":
        return sigla[:1].upper() + sigla[1:].lower()
    if tipus == "lletrejada":
        sigla = PLURAL_SIGLA.sub("", sigla)
        lletres = [NOMS_LLETRA.get(c.lower()) for c in sigla if c.isalpha()]
        return " ".join(l for l in lletres if l) or None
    if tipus == "acronim":
        net = "".join(c for c in sigla if c.isalpha())
        return net[:1].upper() + net[1:].lower() if len(net) > 1 else None
    return None


PROMPT_SIGLES = """Ès un expert en sigles e en sistèmes de sintèsi de votz.

Te doni mots damb majuscules trapadi en un tèxte en ARANÉS, cadun damb un tròç dera frasa
a on apareish. Fè servir eth CONTEXT: er madeish mot pòt èster ua causa o ua auta.

Entà cadun, di coma se ditz en votz auta:
  "lletrejada"  letra per letra                    (DNI, EMD, DOGC, TGV)
  "acronim"     coma un mot                        (OTAN, UNESCO, RENFE)
  "mot"         un mot normau escrit tot en majuscules, non ua sigla  (HAMLET, SENHORA,
                e es mots corrents des titolars: ES, ERA, DAMB, CONSELH)
  "romana"      un nombre roman                    (sègle XIX, NET XII, Alfonso III)
  "normal"      ja se lieg plan tau coma s'escriu   (UdL, iPhone, Aran)

Compde: "CD" pòt èster un disc ("normal") o quate cents ("romana"); "II" pòt èster un
numèro de capitol ("romana") o part d'un nom; "UA" pòt èster eth partit Unitat d'Aran
("lletrejada": u a) o eth mot "ua" (una) escrit en majuscules ("mot"). Eth context ac ditz.

Respon NOMÉS un objècte JSON:
{{"sigles": {{"DOGC": "lletrejada", "XIX": "romana"}}}}
Include TOTES es entrades.

{llista}"""

PROMPT_NOMS = """Ès un expert en aranés (era varietat gascona der occitan dera Val d'Aran) e ena
ortografia des lengües vesies (catalan, castelhan, francés).

Un motor de règles va a aplicar es normes ortografiques araneses (o->u, nh->ny, lh->ll,
sh->x) a aguesti noms pròpis, entà qu'un TTS CATALAN les prononcie coma un aranés.

ERA QÜESTION EI ORTOGRAFICA, NON GEOGRAFICA: ei eth nom escrit en grafia ARANESA/OCCITANA,
o en era grafia d'ua auta lengua?

    "aranes" (s'apliquen es règles): eth nom ei aranés o occitan PER ERA SUA GRAFIA.
      - lòcs, institucions e persones dera Val d'Aran, damb eth sòn nom aranés:
            Vielha -> Viella    Bossòst -> Bussòst    Conselh -> Cunsell    Benós -> Benús
      - exonims escrits en grafia OCCITANA, diferenta dera catalana e dera castelhana:
            Portugau -> Purtugau    Japon -> Japun    Moscòu -> Muscòu    Bordèu -> Burdèu
            Catalonha -> Catalunya  Espanha -> Espanya  Occitània -> Uccitània
            Lheida -> Lleida        Comenge -> Cumenge  Luishon -> Luixun

    "foran" (se dèishe tau qui ei): eth nom ei escrit en era grafia d'UA AUTA LENGUA
    (catalan, castelhan, francés, rus, anglés...), o ei ua forma COMPARTIDA damb eth
    catalan o eth castelhan -- era madeisha grafia enes dues lengües --, qu'eth TTS
    catalan ja lieg ben:
            Barcelona, Girona, Roma, Tolosa, Madrid, Joan, Josep, Jordi, Oriol, Carlos,
            Antonio, Feijoo, Diputació, Institució, Shakespeare, Raskolnikov, Washington
      Compde: `Diputació`, `Institució`, `Federació` son CATALAN (acaben en -ció); `Jordi`,
      `Oriol`, `Coll`, `Pobla`, `Espot` son catalan; `Tomás`, `Amador`, `Domingo` son
      castelhan. Un nom catalan o castelhan NON cambie encara que sigue d'un lòc vesin
      d'Aran: `Jordi` non ei `Jurdi`, `Diputació de Lleida` non ei `Diputaciú`.

Entà cadun, damb eth tròç de frasa que te doni de context, di "aranes" o "foran".

Se dobtes SE ERA GRAFIA EI ARANESA, di "foran": deishar un nom sense tocar ei ua error
petita; cambiar-lo entà quauquarren que non existís, non.

Respon NOMÉS un objècte JSON:
{{"noms": {{"Conselh": "aranes", "Portugau": "aranes", "Barcelona": "foran", "Jordi": "foran"}}}}
Include TOTI es noms.

{llista}"""


PROMPT_GRAFIA = """Ès un expert en ortografia occitana (aranesa) e en noms pròpis internacionaus.

Aguesti noms son de FÒRA dera Val d'Aran (non se les aplique eth cambi vocalic aranés),
mès s'an d'ARTICULAR entà un TTS CATALAN, que non sap liéger es digrafs occitans `nh`,
`lh`, `sh`. Era question ei se aguestes dues letres son UN SON o DUES LETRES SEPARADES:

  "digraf"  -> es un son solet, e s'a de transcriuer entara grafia catalana:
      nh = [ɲ]  `Catalonha` -> `Catalonya`, `Espanha` -> `Espanya`, `Bretanha` -> `Bretanya`
      lh = [ʎ]  `Lhèida` -> `Llèida`, `Sevilha` -> `Sevilya`
      sh = [ʃ]  `Natasha` -> `Nataixa`, `Washington` -> `Waixington` (angles e rus tanben
                an aguest son; era grafia catalana entà [ʃ] ei `x`/`ix`)

  "letres"  -> son dues letres de lengües que non an aguest digraf, e s'an de deishar:
      `Eisenhower` (n+h alemand), `Copenhague`, `Guggenheim`, `Delhi` (l+h),
      `Bellingshausen` (s+h), `Lheureux` (francés l'heureux)

Era clau ei era LENGUA D'ORIGINA deth nom: occitan, catalan, portugués e es
transliteracions russes hètes en occitan an digrafs; alemand, neerlandés e es mots damb
frontèra de morfèma non.

Entà cadun, damb eth tròç de frasa de context, di "digraf" o "letres".
Se dobtes, di "letres": deishar un nom tau qui ei ei ua error petita.

Respon NOMÉS un objècte JSON:
{{"grafia": {{"Catalonha": "digraf", "Eisenhower": "letres"}}}}
Include TOTI es noms.

{llista}"""


def context(mot: str, frases: list[str]) -> str:
    """Un tros de frase al voltant del mot, per al model."""
    for frase in frases:
        i = frase.find(mot)
        if i >= 0:
            a, b = max(0, i - FINESTRA), min(len(frase), i + len(mot) + FINESTRA)
            return ("..." if a else "") + frase[a:b].strip() + ("..." if b < len(frase) else "")
    return ""


def llista_amb_context(mots: list[str], exemples: dict) -> str:
    """Llista de mots amb una frase d'exemple cadascun, per al prompt."""
    return "\n".join(f'- {m}   |  "{context(m, exemples.get(m, []))}"' for m in mots)


def classifica(nous: list[str], exemples: dict, plantilla: str, clau: str,
               valids: set, model: str) -> dict:
    """Classifica els mots nous per lots amb Claude i torna les decisions."""
    fora: dict = {}
    n_lots = (len(nous) - 1) // MIDA_LOT + 1 if nous else 0
    for i in range(0, len(nous), MIDA_LOT):
        lot = nous[i:i + MIDA_LOT]
        try:
            d = claude_cli.json_de_resposta(claude_cli.crida(
                plantilla.format(llista=llista_amb_context(lot, exemples)), model=model))
            d = d.get(clau, d)
            fora.update({k: v for k, v in d.items() if k in set(lot) and v in valids})
            falten = [m for m in lot if m not in fora]
            print(f"    lot {i // MIDA_LOT + 1}/{n_lots}: {len(lot) - len(falten)}/{len(lot)}"
                  + (f"  no tornats: {falten[:3]}" if falten else ""))
        except Exception as e:
            print(f"    lot {i // MIDA_LOT + 1}/{n_lots} ha fallat: {e}")
    return fora


def desa_json(ruta: Path, dades: dict, descripcio: str, **meta) -> None:
    """Escriu un diccionari JSON amb metadades."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"_meta": {"descripcio": descripcio, **meta},
                                **dict(sorted(dades.items()))},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def carrega_json(ruta: Path) -> dict:
    """Llegeix un diccionari JSON (buit si no existeix), sense les metadades."""
    if not ruta.exists():
        return {}
    return {k: v for k, v in json.loads(ruta.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--min-ocurrencies", type=int, default=3,
                   help="cops que ha de sortir un mot per preguntar-lo (defecte: 3). Els "
                        "que no hi arriben es deixen SENSE tocar, que es l'opcio segura")
    p.add_argument("--nomes-candidates", action="store_true")
    p.add_argument("--nomes", choices=("sigles", "noms"))
    args = p.parse_args()

    registres = []
    for font in C.FONTS_FIABLES + C.FONTS_RESERVA:
        registres += corpus.llegeix_jsonl(C.dir_font(font) / "net.jsonl")
    if not registres:
        raise SystemExit("No hi ha corpus net. Executa 03_neteja.py primer.")
    print(f"{len(registres):,} frases netes")

    plurals_json = carrega_json(phonetics.dir_diccionaris(C.IDIOMA) / "plurals_aranes.json")
    plurals = {m for m, v in plurals_json.items() if v}

    sigles, propis, minuscules = collections.Counter(), collections.Counter(), collections.Counter()
    exemples: dict[str, list[str]] = collections.defaultdict(list)
    exemples_titular: dict[str, list[str]] = collections.defaultdict(list)
    for r in registres:
        t = r["raw_text"]
        titular = C.es_titular(t)
        for mot in MOT.findall(t):
            if mot[:1].islower():
                minuscules[mot.lower()] += 1
                continue
            if es_sigla(mot):
                sigles[mot] += 1
            elif not t.startswith(mot):
                propis[mot] += 1
            if titular:
                if len(exemples_titular[mot]) < 2:
                    exemples_titular[mot].append(t)
            elif len(exemples[mot]) < 2:
                exemples[mot].append(t)
    for mot, ts in exemples_titular.items():
        if not exemples[mot]:
            exemples[mot] = ts
    comuns = {w for w, n in minuscules.items() if n >= 3}

    def es_mot_en_majuscules(sigla: str) -> bool:
        """Si la forma en minuscula surt mes cops que la sigla."""
        return minuscules[sigla.lower()] > sigles[sigla]

    cand_sigles = [s for s, n in sigles.most_common()
                   if n >= args.min_ocurrencies or valor_roma(s) is not None]
    descartades_com_a_mot = [s for s in cand_sigles if es_mot_en_majuscules(s)]
    cand_noms = [m for m, n in propis.most_common()
                 if n >= args.min_ocurrencies and m.lower() not in comuns
                 and respelling.reescriu(m, plurals=plurals) != m]
    print(f"sigles: {len(sigles):,} distintes, {len(cand_sigles):,} damb "
          f"{args.min_ocurrencies}+ ocurrencies "
          f"({len(descartades_com_a_mot)} son tambe mots corrents en majuscules i es "
          f"pregunten damb context: {descartades_com_a_mot[:8]})")
    print(f"noms propis que les regles canvien: {len(cand_noms):,} a preguntar "
          f"(el senyal de la minuscula n'ha tret {sum(1 for m, n in propis.most_common() if n >= args.min_ocurrencies and m.lower() in comuns):,})")

    if args.nomes_candidates:
        print("\n--- sigles ---")
        for s in cand_sigles[:25]:
            pista = valor_roma(s)
            print(f"  {sigles[s]:>5}  {s:<14}{f'[roma canonic = {pista}]' if pista else ''}"
                  f"  ctx: {context(s, exemples[s])[:70]}")
        print("\n--- noms propis ---")
        for m in cand_noms[:25]:
            print(f"  {propis[m]:>5}  {m:<18} -> {respelling.reescriu(m, plurals=plurals):<18}"
                  f"  ctx: {context(m, exemples[m])[:58]}")
        return 0

    dir_dicc = phonetics.dir_diccionaris(C.IDIOMA)
    ruta_sigles = dir_dicc / "tipus_sigles_aranes.json"
    ruta_noms = dir_dicc / "noms_propis_aranes.json"
    tipus = carrega_json(ruta_sigles)
    origen = carrega_json(ruta_noms)

    velles = [s for s in tipus if not MOT.fullmatch(s)]
    if velles:
        print(f"\nTreient {len(velles)} claus compostes d'un regex anterior: {velles[:10]}")
        for s in velles:
            del tipus[s]

    if args.nomes != "noms":
        nous = [s for s in cand_sigles if s not in tipus]
        print(f"\n[A] sigles a classificar: {len(nous):,}")
        amb_pista = {s: [f"[nombre roman canonic = {valor_roma(s)}] " + (e[0] if e else "")
                         for e in [exemples.get(s, [])]] if valor_roma(s) else exemples.get(s, [])
                     for s in nous}
        tipus.update(classifica(nous, {**exemples, **amb_pista}, PROMPT_SIGLES, "sigles",
                                {"lletrejada", "acronim", "mot", "romana", "normal"},
                                args.model))
        desa_json(ruta_sigles, tipus,
                  "Sigles araneses: com es diuen en veu alta. Decidit un cop per sigla "
                  "damb un LLM i el context de la frase; el codi nomes aplica la decisio.",
                  model=args.model, sigles=len(tipus))

    if args.nomes != "sigles":
        nous = [m for m in cand_noms if m not in origen]
        print(f"\n[B] noms propis a classificar: {len(nous):,}")
        origen.update(classifica(nous, exemples, PROMPT_NOMS, "noms",
                                 {"aranes", "foran"}, args.model))
        desa_json(ruta_noms, origen,
                  "Noms propis del corpus: 'aranes' = se li apliquen les normes "
                  "ortografiques (aranes i catala les segueixen), 'foran' = es deixa "
                  "intacte (castella, angles, rus, frances...). Decidit damb context.",
                  model=args.model, noms=len(origen))

    if args.nomes != "sigles":
        amb_digraf = [m for m, v in origen.items()
                      if v in ("foran", "foran_grafia") and DIGRAF.search(m)]
        nous = [m for m in amb_digraf if origen[m] == "foran"]
        print(f"\n[C] noms forans damb digraf occita a decidir: {len(nous):,}")
        if nous:
            decisio = classifica(nous, exemples, PROMPT_GRAFIA, "grafia",
                                 {"digraf", "letres"}, args.model)
            for m, v in decisio.items():
                if v == "digraf":
                    origen[m] = "foran_grafia"
            desa_json(ruta_noms, origen,
                      "Noms propis del corpus: 'aranes' = se li apliquen les normes "
                      "ortografiques, 'foran' = es deixa intacte, 'foran_grafia' = de fora "
                      "pero escrit en grafia occitana: nomes se li transcriuen els digrafs "
                      "(nh/lh/sh), sense el canvi vocalic.",
                      model=args.model, noms=len(origen))

    ruta_dicc = phonetics.ruta_diccionari(C.IDIOMA, "treball")
    diccionari = phonetics.carregar(ruta_dicc)
    orfes = [k for k in diccionari.pla()
             if k not in tipus and not diccionari.revisat(k)
             and (es_mot_en_majuscules(k) or not MOT.fullmatch(k) or CLITIC.match(k))]
    for sigla in velles + orfes:
        if diccionari.eliminar(sigla):
            pass
    if velles:
        print(f"  {len(velles)} tretes del diccionari per ser mots corrents o claus compostes")
    if orfes:
        print(f"  {len(orfes)} entrades ORFES tretes (mots corrents en majuscules): {orfes[:10]}")
    n_afegides = 0
    for sigla, t in tipus.items():
        f = fonetica_sigla(sigla, t)
        if f and not diccionari.revisat(sigla):
            diccionari.posar(sigla, f, origen="regla_sigles")
            n_afegides += 1
    phonetics.desar(diccionari, ruta_dicc, idioma=C.IDIOMA)

    print(f"\nsigles: {dict(collections.Counter(tipus.values()))}")
    print(f"noms propis: {dict(collections.Counter(origen.values()))}")
    print(f"{n_afegides} entrades al diccionari fonetic -> {ruta_dicc}")
    print(f"tipus de sigla -> {ruta_sigles}")
    print(f"origen dels noms -> {ruta_noms}")
    forans = [m for m, v in origen.items() if v == "foran"]
    print(f"\n{len(forans)} noms protegits de les regles. Exemples: {forans[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

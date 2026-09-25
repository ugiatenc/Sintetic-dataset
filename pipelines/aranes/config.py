#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tot el que es especific de l'aranes: rutes, fonts, filtre dialectal, llindars, numerals i romans."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ARREL = Path(__file__).resolve().parents[2]
DIR_DADES = ARREL / "data/aranes"
DIR_TEXT = DIR_DADES / "text"
DIR_AUDIO = DIR_DADES / "audio"
DIR_CV_CRU = DIR_AUDIO / "common_voice_cru"
DIR_RESERVA_CA = DIR_AUDIO / "reserva_catalanes"
DIR_VEUS = DIR_AUDIO / "veus"
DIR_ENTITATS = DIR_DADES / "entitats"
DIR_DATASET = ARREL / "datasets/aranes"
DIR_CORPUS = DIR_TEXT / "corpus"
DIR_RESERVA = DIR_TEXT / "corpus_reserva"
MANIFEST_CV = ARREL / "datasets/aranes/test/manifest.jsonl"
DIR_CV_REAL = DIR_DATASET / "cv_real"
MANIFEST_CV_REAL = DIR_CV_REAL / "manifest.jsonl"
DIR_CLIPS = DIR_DATASET / "dataset_aranes"  # els clips, en carpetes <font>_<inici>-<fi> de 4.000 indexs, wav + json
CLIPS_PER_CARPETA = 4000
_ID_SINTETIC = re.compile(r"^([a-z_]+)-(\d{7})(?:-\d+)?$")


def carpeta_clip(id_: str, index: int | None = None, per_blocs: bool = True) -> str:
    """Carpeta d'un clip: `<font>_<inici>-<fi>` pel seu index a la font (els `cv_*`, per `index`); nomes `<font>` si no va per blocs."""
    m = _ID_SINTETIC.match(id_)
    if m:
        font, n = m.group(1), int(m.group(2))
    elif id_.startswith("cv_"):
        font, n = "commonvoice", index or 0
    else:
        raise ValueError(f"id de clip desconegut: {id_}")
    if not per_blocs:
        return font
    inici = (n // CLIPS_PER_CARPETA) * CLIPS_PER_CARPETA
    return f"{font}_{inici:06d}-{inici + CLIPS_PER_CARPETA - 1:06d}"


DIR_CLIPS_EVAL = DIR_DATASET / "dataset_aranes_eval"  # el dev, mateixa estructura, fora del train
FRACCIO_DEV_SINTETIC = 0.005  # 1 de cada 200 frases sintetiques va al dev (hash estable de l'id)
LOCUTORS_DEV_REAL = {  # locutors reals de Common Voice reservats per al dev (frases no vistes al train)
    "c60a65d039b0949ca1fb3d0a620633c7f70198d3ef48204f7b7837de405dcfdbc58f8d2a966e5057bb3dcfed531b748750c459fecea3fb576404c31a1fe2e767": "192 clips, 16,6 min",
    "83c8b30158cbdb4e4a80e7c07611fb5c501d24f9cd040f8285bc0f25527ef9c654ad27067e48f0d0f3068e82087775bcef7e214d2e464691c875d2d217e3a571": "27 clips, 2,1 min",
    "476aa4af4a87a7bd6ec3d6e6664f6d0357deed84fb4730ac682b46278c740faa5be8f036ef5080e928f66ffb7fa2032a0c044f74e95c7d068f2b67696cd8d914": "16 clips, 1,3 min",
    "d708e8dec94010ab66dcb8fd0932ac333d0d5a8407dadee54a4b03c718ca72e75926113726deeb7d165a80e7b1bef32a61454c3710602433c2db1a071b8cb895": "12 clips, 1,1 min",
    "87694782291ae4d30e493f937c151dc5e174408c66c8d4743e18af1d1e10d947456523a9f9e16b772578c3ceb6e1bdbe783b022b68cf958256d8d300e0fa2b68": "8 clips, 0,6 min",
}


def split_clip(id_: str, client_id: str | None = None, text: str | None = None, frases_dev: set | None = None) -> str:
    """`dev`, `train` o `exclos`: els reals pel locutor (LOCUTORS_DEV_REAL), els sintetics per hash de l'id; una frase del train que tambe es al dev queda `exclos`."""
    if id_.startswith("cv_"):
        return "dev" if client_id in LOCUTORS_DEV_REAL else "train"
    h = int(hashlib.sha256(id_.encode("utf-8")).hexdigest()[:8], 16)
    if h % 1000 < FRACCIO_DEV_SINTETIC * 1000:
        return "dev"
    if text is not None and frases_dev:
        import corpus
        if corpus.clau_dedup(text) in frases_dev:
            return "exclos"  # les frases de Common Voice tambe son al BSC: la mateixa frase no pot ser al dev i al train
    return "train"


def frases_dev() -> set:
    """Claus de dedup de totes les frases del dev, reals (cv_real) i sintetiques (pel hash de l'id al pla)."""
    import json
    import corpus
    claus = set()
    if MANIFEST_CV_REAL.exists():
        for l in MANIFEST_CV_REAL.open(encoding="utf-8"):
            r = json.loads(l)
            if r.get("split") == "dev":
                claus.add(corpus.clau_dedup(r["text"]))
    pla = DIR_DATASET / "pla.jsonl"
    if pla.exists():
        for l in pla.open(encoding="utf-8"):
            r = json.loads(l)
            if split_clip(r["id"]) == "dev":
                claus.add(corpus.clau_dedup(r["raw_text"]))
    return claus


def ruta_clip(id_: str, index: int | None = None, split: str | None = None) -> Path:
    """Ruta del wav d'un clip (train a `DIR_CLIPS`, dev a `DIR_CLIPS_EVAL`); el json va al costat amb el mateix nom."""
    dev = (split or split_clip(id_)) == "dev"
    arrel = DIR_CLIPS_EVAL if dev else DIR_CLIPS
    return arrel / carpeta_clip(id_, index, per_blocs=not dev) / f"{id_}.wav"  # el dev, una carpeta per font


LOCUTORS_FORA_DEL_TEST = {  # locutors de Common Voice moguts del test al train (decisio Q, 21/09)
    "dc22966352954ade504cb6caee07f4d480648e17f245885d8375139a566b4c93cc1c732ddd19ecb4c446a84449ce6620655a0eadaa65c4c1dfc0c846a33070cd":
        "633 clips a other (0,93 h), 14 al test",
    "376f704bea9112bb0ce5b0b4c4484d5a3d472c70f0fa6bc48bd316602e31fb4919b62ecf812eff604b27cac122f3c1855a56026f4e9cd7499a3ff7ad98298337":
        "137 clips a other (0,20 h), 10 al test",
}

IDIOMA = "aranes"
IDIOMA_TTS = "ca"
HORES_OBJECTIU = (500, 700)

LIMIT_CAPTURA = 1200
MIN_MOTS, MAX_MOTS = 3, 60  # llargada admesa (decisio A, 21/09)
MIN_CAR, MAX_CAR = 12, 350
MAX_CAR_OBJECTIU = 290  # a partir d'aqui el pas 3 intenta partir la frase


MARQUES_FORTES = {
    "eth", "dera", "deth", "ath",
    "entara", "entath", "entad", "ena", "enes", "eneth", "pera", "peth",
    "ua", "uas", "damb",
    "ei", "auie", "auien", "auec", "auia", "auem", "auetz", "auer", "aueren",
    "hec", "her", "hem", "het", "hara", "haran", "hei",
    "didec", "dider", "diser", "vedetz", "ludien", "arrint",
    "aguest", "aguesta", "aguestes", "aguesti", "aguestos",
    "atau", "aciu", "aquiu", "alavetz", "enquia", "tostemp", "pramor", "arren",
    "hilh", "hilha", "hemna", "hont", "huec", "hesta", "hum", "hred", "haria",
    "gojat", "gojata", "audeth", "audeths", "deishar", "deishat", "deishec", "shinhau",
    "quauquarren", "quauqui", "quauques", "quauque", "maleros", "lheuec", "lheuar",
    "nosati", "vosati", "auti", "toti", "naui",
}
MARQUES_FORANES = {
    "amb", "aquest", "aquesta", "aquests", "aquestes", "aixo", "nosaltres", "vosaltres",
    "pero", "perque", "tambe", "molt", "fer", "fill", "filla", "dona", "foc",
    "els", "dels", "als", "pels", "aquell", "aquella", "anys", "seva", "seu", "meva",
    "los", "las", "porque", "tambien", "muy", "hacer", "hijo", "mujer", "fuego",
    "para", "cuando", "donde", "segun",
    "lei", "leis", "dau", "dei", "aquel", "aqueste",
    "con", "por", "desde", "hasta", "hacia", "durante", "su",
    "aunque", "entonces", "siempre", "mientras", "además",
    "este",
    "enero", "febrero", "marzo", "mayo", "junio", "julio", "agosto",
    "septiembre", "noviembre", "diciembre",
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado",
    "dab", "òmi", "mei", "deus", "medish", "medisha",
    "deths", "eths", "aths", "aras", "ders", "eras", "dens", "hauts",
    "gascoa", "navèra", "navèth", "navèra", "que's",
    "dans", "avec", "mais", "très", "cette", "c'est", "qu'il", "n'est", "j'ai",
    "monsieur", "oui", "jamais", "toujours", "était", "avait", "leur", "chez",
    "au",
    "lo", "la", "una", "del", "al",
}
ARTICLES_FORANS = {"lo", "la", "una", "del", "al"}

MIN_MARQUES = 1  # marques araneses fortes minimes per passar el filtre dialectal
MAX_FORANES = 0  # cap marca catalana, castellana ni del gascon de Franca

_MOT = re.compile(r"[\w']+")


def _tokens(frase: str) -> list[str]:
    """Mots de la frase, amb els CLITICS separats."""
    import corpus
    fora = []
    for m in _MOT.findall(corpus.normalitzar(frase)):
        fora.extend(t for t in m.split("'") if t)
    return fora


def compta_marques(frase: str) -> tuple[int, int]:
    """(marques araneses fortes, marques foranes)."""
    import corpus
    fortes = {corpus.normalitzar(x) for x in MARQUES_FORTES}
    foranes = {corpus.normalitzar(x) for x in MARQUES_FORANES}
    mots = _tokens(frase)
    n_f = sum(1 for m in mots if m in fortes)
    n_g = sum(1 for i, m in enumerate(mots)
              if m in foranes and not (m in ARTICLES_FORANS and i == 0))
    return n_f, n_g


def puntuacio_aranes(frase: str) -> int:
    """Puntuacio per al `Filtre` de `src/corpus.py`: marques fortes, o 0 si n'hi ha cap de forana."""
    fortes, foranes = compta_marques(frase)
    return 0 if foranes > MAX_FORANES else fortes


POLITICA_XIFRES = "expandir"  # les xifres es passen a lletres al pas 1

UNITATS = ["zero", "un", "dus", "tres", "quate", "cinc", "sies", "sèt", "ueit", "nau",
           "dètz", "onze", "dotze", "tretze", "catorze", "quinze", "setze",
           "dètz-e-sèt", "dètz-e-ueit", "dètz-e-nau"]

DESENES = {20: "vint", 30: "trenta", 40: "quaranta", 50: "cinquanta", 60: "seishanta",
           70: "setanta", 80: "ueitanta", 90: "nauanta"}

DESENA_AMB_E = {20}

MESOS = {1: "gèr", 2: "hereuèr", 3: "març", 4: "abriu", 5: "mai", 6: "junh",
         7: "juriòl", 8: "agost", 9: "seteme", 10: "octobre", 11: "noveme", 12: "deseme"}

ORDINALS_IRREGULARS = {1: "prumèr", 7: "setau"}
ORDINALS_FEMENINS = {1: "prumèra"}


def en_lletres(n: int) -> str | None:
    """Numero -> paraules araneses."""
    if n < 0 or n >= 1_000_000_000:
        return None
    if n < 20:
        return UNITATS[n]
    if n < 100:
        d, u = divmod(n, 10)
        base = DESENES[d * 10]
        if not u:
            return base
        return f"{base}-e-{UNITATS[u]}" if d * 10 in DESENA_AMB_E else f"{base} {UNITATS[u]}"
    if n < 1000:
        c, r = divmod(n, 100)
        base = "cent" if c == 1 else f"{UNITATS[c]} cents"
        return base if not r else f"{base} {en_lletres(r)}"
    if n < 1_000_000:
        m, r = divmod(n, 1000)
        base = "mil" if m == 1 else f"{en_lletres(m)} mil"
        return base if not r else f"{base} {en_lletres(r)}"
    m, r = divmod(n, 1_000_000)
    base = "un milion" if m == 1 else f"{en_lletres(m)} milions"
    return base if not r else f"{base} {en_lletres(r)}"


def ordinal(n: int, femeni: bool = False) -> str | None:
    """Numero -> ordinal aranes."""
    if n in ORDINALS_IRREGULARS:
        masculi = ORDINALS_IRREGULARS[n]
        return ORDINALS_FEMENINS.get(n, masculi + "a") if femeni else masculi
    base = en_lletres(n)
    if base is None:
        return None
    for valor, forma in ORDINALS_IRREGULARS.items():
        cardinal = en_lletres(valor)
        if valor > 1 and cardinal and base.endswith(cardinal):
            return base[:-len(cardinal)] + forma
    masculi = (base[:-1] if base.endswith("e") else base) + "au"
    return masculi + "a" if femeni else masculi


ROMANA = re.compile(r"^([IVXLCDM]+)(au|AU|aua|AUA)?$")  # roma canonic amb sufix ordinal opcional (IIau, IIaua)
VALORS_ROMA = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def a_roma(n: int) -> str:
    """Nombre roma canonic d'un enter."""
    taula = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    fora = []
    for valor, simbol in taula:
        while n >= valor:
            fora.append(simbol)
            n -= valor
    return "".join(fora)


def valor_roma(s: str) -> int | None:
    """Valor si `s` es un nombre roma CANONIC (amb sufix ordinal o sense), `None` altrament."""
    m = ROMANA.match(s)
    if not m:
        return None
    nucli = m.group(1)
    if m.group(2) and len(nucli) == 1 and nucli not in "IX":
        return None
    total, anterior = 0, 0
    for c in reversed(nucli):
        v = VALORS_ROMA[c]
        total += -v if v < anterior else v
        anterior = max(anterior, v)
    return total if total and a_roma(total) == nucli else None


def es_ordinal_roma(s: str) -> bool:
    """Si es un roma canonic amb sufix ordinal (IIau, Xau)."""
    m = ROMANA.match(s)
    return bool(m and m.group(2) and valor_roma(s) is not None)


def fonetica_roma(s: str) -> str | None:
    """Com es diu un roma canonic: ordinal si porta sufix (`IIau` -> `dusau`, `IIaua` -> `dusaua`)."""
    n = valor_roma(s)
    if n is None:
        return None
    m = ROMANA.match(s)
    if m.group(2):
        return ordinal(n, femeni=m.group(2).lower() == "aua")
    return en_lletres(n)


ORDINAL_XIFRA = re.compile(r"\b(\d+)(aua|au|èra|èr|a)\b")


def _expandir_ordinal(m) -> str:
    """Ordinal en xifres (3au, 5a) -> lletres."""
    n, sufix = int(m.group(1)), m.group(2)
    forma = ordinal(n, femeni=sufix in ("a", "aua", "èra"))
    return forma if forma else m.group(0)


LECTURES_SIMBOL = [
    (re.compile(r"\s*€"), " èuros"),
    (re.compile(r"\s*\$"), " dolars"),
    (re.compile(r"\s*º\s*C\b"), " graus"),
    (re.compile(r"\s*ºC\b"), " graus"),
    (re.compile(r"\s*km\s*/\s*h\b"), " quilomètres per ora"),
    (re.compile(r"\s*m²"), " mètres quadrats"),
    (re.compile(r"\s*km²"), " quilomètres quadrats"),
]

UNITATS_ABREUJADES = [
    (r"kms", "quilomètres"), (r"cms", "centimètres"),
    (r"km²", "quilomètres quadrats"), (r"km2", "quilomètres quadrats"),
    (r"m²", "mètres quadrats"), (r"m2", "mètres quadrats"),
    (r"km/h", "quilomètres per ora"),
    (r"km", "quilomètres"), (r"cm", "centimètres"), (r"mm", "milimètres"),
    (r"m", "mètres"),
    (r"kg", "quilograms"), (r"gr", "grams"),
    (r"ha", "ectàrees"),
    (r"h", "ores"), (r"min", "minutes"),
    (r"kW", "quilovats"), (r"MW", "megavats"), (r"W", "vats"), (r"V", "volts"),
    (r"A", "amperes"), (r"Hz", "hertz"),
]
NUMERAL_UNITAT = re.compile(
    r"(\b(?:" + "|".join(sorted(
        ["zero", "un", "ua", "dus", "dues", "tres", "quate", "cinc", "sies", "sèt", "ueit",
         "nau", "dètz", "onze", "dotze", "tretze", "catorze", "quinze", "setze", "vint",
         "trenta", "quaranta", "cinquanta", "seishanta", "setanta", "ueitanta", "nauanta",
         "cent", "cents", "mil", "milion", "milions", "mieja", "mieg"], key=len, reverse=True))
    + r")(?:-e-[a-zà-ú]+)?)\s+("
    + "|".join(re.escape(u) for u, _ in UNITATS_ABREUJADES)
    + r")(?![a-zà-ú0-9²])")
_LECTURA_UNITAT = {u: lectura for u, lectura in UNITATS_ABREUJADES}


NUMERAL_PER = re.compile(
    r"(?<=[a-zà-ú])\s+x\s+(?=(?:zero|un|ua|dus|dues|tres|quate|cinc|sies|sèt|ueit|nau|"
    r"dètz|onze|dotze|vint|trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|nauanta|"
    r"cent|cents|mil)\b)")


def expandir_unitats(frase: str) -> str:
    """`trenta quate m` -> `trenta quate mètres`."""
    frase = NUMERAL_PER.sub(" per ", frase)
    return NUMERAL_UNITAT.sub(
        lambda m: f"{m.group(1)} {_LECTURA_UNITAT[m.group(2)]}", frase)


MILERS = re.compile(r"\b(\d{1,3}(?:\.\d{3})+)\b")
DECIMAL = re.compile(r"\b(\d+),(\d+)\b")
ORA = re.compile(r"\b([01]?\d|2[0-3])[.:]([0-5]\d)(?:\s*h\b|\b)(?!\s*[.:]?\d)")


def _expandir_ora(m) -> str:
    """`12.00` -> `dotze`, `9:30` -> `nau e trenta`."""
    ora, minuts = int(m.group(1)), int(m.group(2))
    hh = en_lletres(ora)
    if hh is None:
        return m.group(0)
    return hh if minuts == 0 else f"{hh} e {en_lletres(minuts)}"


TELEFON = re.compile(
    r"(?<![\d.])(?:\+34[  ]?|0034[  ]?)?"
    r"(\d{3}[ . ]\d{2}[ . ]\d{2}[ . ]\d{2}"
    r"|\d{2}[ . ]\d{3}[ . ]\d{2}[ . ]\d{2}"
    r"|\d{3}[  -]\d{6})(?!\.?\d)")


def _expandir_telefon(m) -> str:
    """Telefon xifra a xifra."""
    return " ".join(UNITATS[int(c)] for c in m.group(1) if c.isdigit())


NOMS_LLETRA = {
    "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "efa", "g": "ge",
    "h": "hac", "i": "i", "j": "jota", "k": "ca", "l": "ela", "m": "ema", "n": "ena",
    "o": "o", "p": "pe", "q": "cu", "r": "erra", "s": "essa", "t": "te", "u": "u",
    "v": "ve", "w": "ve doble", "x": "ics", "y": "i grega", "z": "zeta",
}
CARRETERA = re.compile(r"\b([A-Z]{1,2})-(\d{1,4})(?![\w-])")


def _expandir_carretera(m) -> str:
    """Carretera o codi lletra-numero (N-230) -> lletres i nombre."""
    n = en_lletres(int(m.group(2)))
    lletres = " ".join(NOMS_LLETRA[c.lower()] for c in m.group(1))
    return f"{lletres} {n}" if n else m.group(0)


def expandir_xifres(frase: str) -> str:
    """Xifres, simbols i dates -> paraules."""
    frase = TELEFON.sub(_expandir_telefon, frase)
    frase = CARRETERA.sub(_expandir_carretera, frase)
    frase = re.sub(r"\b(\d+)\s*(?:º|ª|er|r|n|t|è|é|éme|ème)\b",
                   lambda m: ordinal(int(m.group(1))) or m.group(0), frase)
    frase = ORDINAL_XIFRA.sub(_expandir_ordinal, frase)
    frase = re.sub(r"(?<!\d)(\d{1,3})((?:[ \u00a0\u202f]\d{3})+)(?!\d)",
                   lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), frase)
    frase = MILERS.sub(lambda m: m.group(1).replace(".", ""), frase)
    frase = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b",
                   lambda m: (f"{en_lletres(int(m.group(1)))} de {MESOS.get(int(m.group(2)), '')} "
                              f"de {en_lletres(int(m.group(3)))}")
                   if MESOS.get(int(m.group(2))) and en_lletres(int(m.group(1)))
                   and en_lletres(int(m.group(3))) else m.group(0), frase)
    frase = ORA.sub(_expandir_ora, frase)
    frase = re.sub(r"\b([01]?\d|2[0-3]),([0-5]\d)\s*(?=h\b|ores\b|h\.)",
                   lambda m: _expandir_ora(m) + " ", frase)
    frase = re.sub(r"(?<=\d)(km²|m²|km/h|km|cm|mm|kg|gr|ha|kW|MW|Hz)(?![a-zà-ú])",
                   r" \1", frase)
    frase = re.sub(r"(?<=\d)(m2|km2)(?![a-zà-ú0-9])", r" \1", frase)
    frase = DECIMAL.sub(
        lambda m: (f"{en_lletres(int(m.group(1)))} coma {en_lletres(int(m.group(2)))}"
                   if en_lletres(int(m.group(1))) and en_lletres(int(m.group(2)))
                   else m.group(0)), frase)
    frase = re.sub(r"(?<![\w-])(\d+)\s*%\s*-\s*(\d+)\s*%",
                   lambda m: (f"{en_lletres(int(m.group(1)))} a {en_lletres(int(m.group(2)))} per cent"
                              if en_lletres(int(m.group(1))) and en_lletres(int(m.group(2)))
                              else m.group(0)), frase)
    frase = re.sub(r"(?<![\w-])(\d+)\s*[-–—]\s*(\d+)(?![\w-])",
                   lambda m: (f"{en_lletres(int(m.group(1)))} a {en_lletres(int(m.group(2)))}"
                              if en_lletres(int(m.group(1))) and en_lletres(int(m.group(2)))
                              else m.group(0)), frase)
    for patro, lectura in LECTURES_SIMBOL:
        frase = patro.sub(lectura, frase)
    frase = re.sub(r"(\d+)\s*%",
                   lambda m: f"{en_lletres(int(m.group(1))) or m.group(1)} per cent", frase)
    frase = re.sub(r"\s*%", " per cent", frase)
    frase = re.sub(r"\bde (?=(agost|abriu|octobre)\b)", "d'", frase)
    frase = re.sub(r"\b\d+\b", lambda m: en_lletres(int(m.group(0))) or m.group(0), frase)
    frase = expandir_unitats(frase)
    return re.sub(r"\s+", " ", frase).strip()


FONTS = {
    "arannoticies": {
        "tipus": "wp",
        "api": "http://arannoticies.com/wp-json/wp/v2/posts",
        "fiable": True,
        "que_es": "diari de la Val d'Aran, 6.946 articles. Registre periodistic actual: "
                  "es exactament el domini del projecte i la font de mes qualitat que hi "
                  "ha. Publica barrejat en aranes, catala i castella; el filtre dialectal "
                  "se n'ocupa. Es llegeix per l'API REST de WordPress i no rascant HTML: "
                  "l'API pagina sobre l'arxiu SENCER (seguint nomes els enllacos de la "
                  "portada en sortien 268 frases, de 60 articles) i torna el cos de "
                  "l'article ja separat del menu, el peu i els widgets.",
    },
    "conselharan": {
        "tipus": "wp",
        "fiable": True,
        "api": "https://www.conselharan.org/wp-json/wp/v2/posts",
        "que_es": "Conselh Generau d'Aran, 3.725 articles. La institucion de la Val: "
                  "noticies, acords de plens i comunicats. Registre institucionau e "
                  "periodistic, i aranes de referencia -- ei qui fixa era norma.",
    },
    "institut_estudis": {
        "tipus": "wp",
        "fiable": True,
        "api": "https://www.institutestudisaranesi.cat/wp-json/wp/v2/posts",
        "que_es": "Institut d'Estudis Aranesi, 426 articles. Pocs, pero es l'academia de "
                  "la llengua: l'aranes mes cuidat que es pot trobar publicat.",
    },
    "visitvaldaran": {
        "tipus": "wp",
        "fiable": True,
        "api": "https://visitvaldaran.com/wp-json/wp/v2/posts",
        "que_es": "Torisme Val d'Aran, 558 articles. Registre divulgatiu, damb molts "
                  "toponims i noms d'activitats de la Val.",
        "exclosa": "21/09: nomes 23 frases arribaven al corpus final, i eren castella damb "
                   "toponims aranesos (`Refugio dera Honeria, río Torán, antiguas bordas`): "
                   "passaven el filtre perque `dera` i `deth` son dins del nom propi. "
                   "Aportava 0,0 h.",
    },
    "wikipedia_oc": {
        "fiable": False,
        "tipus": "wikipedia",
        "api": "https://oc.wikipedia.org/w/api.php",
        "cerca": ["Val d'Aran", "Aran", "aranés", "Vielha", "Naut Aran", "Occitània",
                  "Pirenèus", "Catalonha", "gascon"],
        "que_es": "oc.wikipedia. Majoritariament LENGADOCIA: el filtre dialectal en "
                  "descarta la major part i nomes hi deixa els articles aranesos.",
    },
    "bsc_ca_arn": {
        "fiable": False,
        "tipus": "hf",
        "dataset": "BSC-LT/Catalan-Aranese_Parallel_Corpus",
        "camp": "oc_arn",
        "que_es": "539k frases catala-aranes del BSC. Aranes de debo (`entara`, `damb`, "
                  "`eth`), pero registre administratiu i institucional, no periodistic. "
                  "Nomes se'n fa servir el costat aranes: es text monolingue, l'alineament "
                  "amb el catala no hi te cap paper.",
    },
    "aina_es_arn": {
        "fiable": False,
        "tipus": "hf",
        "dataset": "projecte-aina/ES-OC_Parallel_Corpus",
        "camp": "arn",
        "que_es": "420k frases del Projecte AINA. ATENCIO: el corpus es en bona part "
                  "SINTETIC, generat amb el traductor de regles Apertium, i una part del "
                  "costat occita no es aranes sino gascon general (`lo loc`, no `eth loc`). "
                  "El filtre dialectal n'hi treu molt, pero el que passa pot arrossegar "
                  "errors sistematics del traductor. Es la font de menys confianca: va "
                  "l'ultima a la prioritat i conve mirar-se'n les descartades.",
        "exclosa": "21/09: 63.412 frases uniques (la resta son duplicats del BSC), pero es "
                   "la font damb mes castella residual (5,7 %: `trisección`, `prosperidad`, "
                   "`cloruro de vinilo`) i la que porta el gascon general. Fora del pla; la "
                   "captura es va esborrar en la neteja del 21/09 (`01_corpus.py --font "
                   "aina_es_arn` la torna a baixar si mai es reconsidera).",
    },
    "claude": {
        "fiable": True,
        "tipus": "generat",
        "que_es": "frases generades amb Claude per cobrir el que els corpus reals no "
                  "donen: to de radio, toponims i institucions d'Aran, i densitat alta "
                  "dels trets ortografics que Whisper falla. Veure 02_frases_claude.py.",
    },
}
PRIORITAT = [f for f in ["conselharan", "institut_estudis", "arannoticies", "visitvaldaran",
                         "claude", "bsc_ca_arn", "wikipedia_oc", "aina_es_arn"]
             if not FONTS[f].get("exclosa")]

FONTS_FIABLES = [f for f in PRIORITAT if FONTS[f].get("fiable")]
FONTS_RESERVA = [f for f in PRIORITAT if not FONTS[f].get("fiable")]


def presegmentada(nom: str) -> bool:
    """La font ja ve partida en frases per qui la publica?"""
    return FONTS[nom].get("tipus") == "hf"


def es_titular(frase: str) -> bool:
    """Frase escrita (gairebe) tota en majuscules: un titular o una acotacio de teatre."""
    lletres = [c for c in frase if c.isalpha()]
    return (len(frase.split()) >= 3 and bool(lletres)
            and sum(c.isupper() for c in lletres) / len(lletres) >= 0.8)


CASTELLA_TOU = re.compile(
    r"\b(?:[a-zà-ú]+(?:dad|mente|ismo|ismos|aje|ajes|illo|illa)"
    r"|[a-zà-ú]*ñ[a-zà-ú]*"
    r"|y|excepto|también|además|según|aunque|mientras|siempre|entonces|cuando|donde|hacia|"
    r"hasta|desde|todo|toda|todos|todas|nada|algo|otro|otra|otros|otras|mismo|misma|pero|"
    r"porque|ahora|después|antes|ayer|mañana|aquí|allí|así|muy|más|menos|bien|nunca|tampoco|"
    r"sin|con|por|para|los|las|unos|unas)\b")
Y_ENTRE_NOMS = re.compile(r"\b[A-ZÀ-Ü][a-zà-ú]+ y (?=[A-ZÀ-Ü])")
MAX_CASTELLA_TOU = 1  # nomes fonts presegmentades (BSC): amb 2+ senyals de castella la frase cau


def castella_tou(frase: str) -> int:
    """Nombre de senyals TOUS de castella (vegeu `CASTELLA_TOU`)."""
    return len(CASTELLA_TOU.findall(frase)) - len(Y_ENTRE_NOMS.findall(frase))


MAX_PER_ESQUELET = 20  # copies maximes d'una mateixa plantilla (frase sense els numerals)


def dir_font(nom: str) -> Path:
    """On viu una font: al corpus o a la reserva."""
    return (DIR_CORPUS if FONTS[nom].get("fiable") else DIR_RESERVA) / nom

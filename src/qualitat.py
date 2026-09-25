#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Porta de qualitat: normalitza una frase i decideix si es prou bona per generar-hi audio."""

from __future__ import annotations

import collections
import re
import unicodedata

LLETRES = "abcdefghijklmnopqrstuvwxyz" "àáâäãèéêëíîïòóôöõúûüçñýÿåøæœß"
PUNTUACIO = " .,;:!?()«»\"'-…·"
NOMES_ALFABET = set(LLETRES + LLETRES.upper() + PUNTUACIO)

SUBSTITUCIONS = {
    "‟": "'", "‛": "'", "‘": "'", "’": "'", "´": "'",
    "`": "'", "ʼ": "'",
    "“": '"', "”": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-", "­": "",
    "…": "...", " ": " ", " ": " ", " ": " ",
    "​": "", "‎": "", "‏": "", "﻿": "",
}

ABREVIATURA_FINAL = re.compile(
    r"(\b[A-ZÀ-ÜÇ]\.$)"
    r"|(\b(sr|sra|srs|etc|eca|núm|p|pag|pàg|fig|dr|dra|sta|ibid)\.$)",
    re.UNICODE)
_ABREV_MINUSCULA = re.compile(
    r"\b(sr|sra|srs|etc|eca|núm|p|pag|pàg|fig|dr|dra|sta|ibid)\.$", re.IGNORECASE)
MARCA_LLISTA = re.compile(r"^\s*([•·◦‣▪–—*+>»]|\d+[.)]|[a-z][.)])\s")
CAPCALERA_PAGINA = re.compile(
    r"\bp\.\s+(un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz|onze|dotze|tretze|catorze|"
    r"quinze|setze|vint|trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|nauanta|cent|"
    r"mil)\b", re.IGNORECASE)
MOT = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

ETIQUETA_CAMP = re.compile(r"^[A-ZÀ-Ü][a-zà-ú]+(?:,\s*-?[a-zà-ú]+)?\s*:\s+[A-ZÀ-Ü\"«]")
VERB_DICCIO = re.compile(r"^(?:Didec|Didie|Diguec|Ditz|Didetz|Responec|Respondec|Respon|"
                         r"Contestèc|Exclamèc|Cridèc|Pensèc|Comentèc|Escotatz|Demandèc)\s*:")
APOSTROF_PERDUT = re.compile(r"(?<!['\u2019])\b(d|l|qu|s)\s+(?=[aeiouàèéíòóú])")
FINALS_TRUNCADA = {
    "e", "o", "que", "qu", "de", "d", "a", "en", "per", "damb", "entà", "entara", "entath",
    "dera", "deth", "des", "pera", "peth", "ena", "enes", "sense", "entre", "sus", "coma",
    "pr'amor", "perque", "donques", "mentre", "segontes", "tà",
}


def _ultim_mot(frase: str) -> str:
    """Ultim mot d'una frase (amb apostrof), en minuscula."""
    mots = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", frase.lower())
    return mots[-1] if mots else ""


FUNCIONALS = {
    "eth", "era", "es", "er", "un", "ua", "uns", "ues", "de", "deth", "dera", "des",
    "a", "ath", "ara", "as", "en", "ena", "enes", "e", "o", "que", "qu", "se", "non",
    "non", "per", "pera", "peth", "damb", "coma", "mes", "mès", "tot", "toti", "ja",
    "i", "li", "lo", "la", "les", "sus", "entà", "entara", "entath", "ei", "son", "au",
    "jo", "tu", "eth", "era", "nosati", "vosati", "eri", "eres", "me", "te", "nos", "vos",
    "ac", "ne", "hè", "ha", "an", "as", "sò", "ès", "cap", "arren", "aquerò", "aquiu",
    "sòn", "sòns", "sua", "sues", "mèn", "mèns", "mia", "mies", "tòn", "tòns", "tua",
    "tues", "nòste", "nòsta", "nòsti", "nòstes", "vòste", "vòsta", "vòsti", "vòstes",
    "sòi", "mos", "vo", "'n", "en", "der", "ders", "ath", "aths", "ara", "aras",
    "aguest", "aguesta", "aguesti", "aguestes", "aqueth", "aquera", "aqueri", "aqueres",
    "çò", "quau", "quaus", "qui", "quan", "on", "com", "tan", "tant", "tanben", "tampoc",
    "sense", "damb", "entre", "sus", "jos", "dempús", "abans", "mentre", "pendent",
    "alavetz", "atau", "totun", "donques", "pr'amor", "perque", "tot", "tota", "totes",
    "bèth", "bèra", "cada", "quauque", "quauqui", "quauques", "fòrça", "plan", "mès",
    "ère", "èren", "auie", "auien", "an", "a", "son", "ei", "sigue", "siguec", "estat",
    "hèr", "hèt", "auer", "poder", "pòt", "pòden", "ja", "encara", "tostemp", "jamès",
    "dus", "dues", "part", "parts",
}


_NUMERAL = (r"(?:zero|un|ua|dus|dues|tres|quate|cinc|sies|sèt|ueit|nau|dètz|onze|dotze|tretze|"
            r"catorze|quinze|setze|vint|trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|"
            r"nauanta|cent|cents|mil)")
NUMERAL_PUNT = re.compile(r"\b(" + _NUMERAL + r")\.(" + _NUMERAL + r")\b")
PUNT_SENSE_ESPAI = re.compile(r"[a-zàèéíòóúïü]\.[a-zàèéíòóúïü]")
ETIQUETA = re.compile(r"^([A-ZÀ-Ü]{3,}(?:\s+[A-ZÀ-Ü'’]+)*)\s+(?=\S)")
ACOTACIO = re.compile(r"^\([^()]{1,80}\)\s+(?=[A-ZÀ-Ü]{3,}\s)")
SIGLA_MAJ = re.compile(r"\b[A-ZÀ-Ü]{2,}\b")
ROMA_CANONIC = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_VALOR_ROMA = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
SEGLE_ROMA = re.compile(r"\b(sègles?|s\.)\s+([ivxlcdm]{1,6})\b", re.IGNORECASE)
SEGLE_ROMA_CADENA = re.compile(r"\b([IVXLCDM]{1,6})\s+(e|a|as|ath)\s+([ivxlcdm]{1,6})\b")
SEGLE_ROMA_ELIPTIC = re.compile(r"\b(deth|dera|des|deth|del|en|a|ath)\s+([ivxlcdm]{1,6})\b")


def _valor_roma(s: str) -> int | None:
    """El valor d'un roman en minuscules, o None si no es canonic."""
    s = s.lower()
    if not s or not ROMA_CANONIC.fullmatch(s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = _VALOR_ROMA[c]
        total += -v if i + 1 < len(s) and _VALOR_ROMA[s[i + 1]] > v else v
    return total


def _es_segle(s: str) -> bool:
    """Si un roma val entre 1 i 21 (un segle)."""
    v = _valor_roma(s)
    return v is not None and 1 <= v <= 21


VERB_MAJUSCULA = re.compile(
    r"(?<=[a-zàèéíòóúïüç,])(\s+)(Ei|Auec|Auie|Auien|Auem|Siguec|Sigueren|Ère|Èren|Hec|Hège|Aurà|Auràn)"
    r"(?=\s+[a-zàèéíòóúïüç])")


def normalitza(text: str, presegmentat: bool = False) -> str:
    """Passa les variants recuperables a la seva forma bona."""
    for dolent, bo in SUBSTITUCIONS.items():
        text = text.replace(dolent, bo)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(?<=\w)'\s+(?=\w)", "'", text)
    text = re.sub(r"(?<=\w)\s+'(?=\w)", "'", text)
    text = text.replace("Mª", "Maria").replace("Dª", "Donya").replace("nº", "numero")
    text = re.sub(r'(?<=\w)"(?=\w)', "'", text)
    text = re.sub(r"\b[a-zà-ÿ]*[ìù][a-zà-ÿ]*\b",
                  lambda m: m.group(0).replace("ì", "í").replace("ù", "ú"), text)
    text = re.sub(r"^(?:[-—–]+\s*)+", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([a-zà-ú])-\s+([a-zà-ú])", r"\1-\2", text)
    text = re.sub(r"\b(\w{2,})\s+\1\b", r"\1", text)
    def _separa_enganxada(m):
        """Separa una capcalera en majuscules enganxada a la frase."""
        maj, cua = m.group(1), m.group(2)
        if ROMA_CANONIC.match(maj.lower()):
            return m.group(0)
        return f"{maj[:-1]} {maj[-1]}{cua}"
    text = re.sub(r"\b([A-ZÀ-Ü]{5,})([a-zà-ú]+)", _separa_enganxada, text)
    text = re.sub(r"^\([^()]{1,80}\)\s+(?=[A-ZÀ-Ü]{3,}\s)", "", text)
    text = re.sub(r"^(?:NET|SCÈNA|SCENA|CAPITOL|CAPÍTOL|LIBRE|ACTE|PART)\s+[IVXLC]+\s+(?=[A-ZÀ-Ü\"«])",
                  "", text, flags=re.IGNORECASE)
    text = VERB_MAJUSCULA.sub(lambda m: m.group(1) + m.group(2).lower(), text)
    text = re.sub(r"^\s*[•·◦‣▪*+>»]\s*(?=[A-ZÀ-Ü])", "", text)
    text = re.sub(r"^\s*[bcdfghjklmnpqrstwyzBCDFGHJKLMNPQRSTWYZ]\s+(?=[A-ZÀ-Ü])", "", text)  # lletra d'apartat sola (b Incorporar); I, V, X fora perque poden ser romans
    text = ENTRADA_DICCIONARI.sub("", text) or text
    text = CLAUSULA_NUMERADA.sub("", text) or text
    text = CLAUSULA_SENSE_PUNT.sub("", text) or text
    if presegmentat:
        text = CLAUSULA_NUMERADA_MIN.sub("", text) or text
        text = re.sub(r"\b([a-zà-ú]+)(ción|sión)\b",
                      lambda m: m.group(1) + m.group(2).replace("ó", "o"), text)
        text = re.sub(r"\b([a-zà-ú]+)í(as?)\b", r"\1i\2", text)
    if presegmentat and text[:1].islower():
        text = text[:1].upper() + text[1:]
    text = SEGLE_ROMA.sub(
        lambda m: f"{m.group(1)} {m.group(2).upper()}" if _es_segle(m.group(2)) else m.group(0),
        text)
    for _ in range(3):
        nou = SEGLE_ROMA_CADENA.sub(
            lambda m: (f"{m.group(1)} {m.group(2)} {m.group(3).upper()}"
                       if _es_segle(m.group(1)) and _es_segle(m.group(3)) else m.group(0)),
            text)
        if nou == text:
            break
        text = nou
    if re.search(r"\bsègles?\b", text, re.IGNORECASE) and re.search(r"[IVXLCDM]{2,}", text):
        text = SEGLE_ROMA_ELIPTIC.sub(
            lambda m: (f"{m.group(1)} {m.group(2).upper()}"
                       if _es_segle(m.group(2)) else m.group(0)), text)
    text = NUMERAL_PUNT.sub(r"\1 punt \2", text)
    text = re.sub(r"([a-zà-ú])\1{2,}", r"\1\1", text)
    text = re.sub(r"(?<=[ns])\.(?=h[a-zàèéíòóúïü])", "·", text)
    text = re.sub(r'^"\s*"\s*', "", text)
    text = re.sub(r'\(\s*\)|(?<![.!?])""|«\s*»', "", text)
    text = re.sub(r"\s*--\s*", " ", text)
    text = re.sub(r"([.!?])\s*[.!?](?:\s*[.!?])*\s*$", r"\1", text)
    text = re.sub(r'\s*\.{3,}(?=["»)\]])', ".", text)
    text = re.sub(r"\s*\.{3,}\s*$", ".", text)
    text = re.sub(r"\s*\.{3,}\s*", ", ", text)
    text = re.sub(r",\s*([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])\s*,", r"\1", text)
    text = re.sub(r",\s*(?=[)\]»])", "", text)
    text = re.sub(r"(?<=[(\[«])\s*,\s*", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


CONSONANTS_IMPOSSIBLES = re.compile(r"[bcdfgjkpqtvwxz]{4,}")
MOT_SENSE_VOCALS = re.compile(r"(?<![\w'’\-])[bcdfghjklmnpqrstvwxz]{3,}(?![\w'’\-])")
RANG_ENGANXAT = re.compile(
    r"\b(?!covid-)[a-zà-ú]+-(?:mil|cents?|milions?|dètz|vint|trenta|quaranta|cinquanta"
    r"|seishanta|setanta|ueitanta|nauanta|cent)\b", re.IGNORECASE)
NUMERAL_ZERO = re.compile(
    r"\b(?:un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz|onze|dotze|tretze|catorze|quinze"
    r"|setze|vint|trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|nauanta|cents?|mil"
    r"|milions?)(?:-e-[a-zà-ú]+)? zero\b")
NUMERALS = frozenset(
    "zero un ua dus dues tres quate cinc sies sèt ueit nau dètz onze dotze tretze catorze "
    "quinze setze vint trenta quaranta cinquanta seishanta setanta ueitanta nauanta cent "
    "cents mil milion milions coma punt".split())
_NUM_ALT = "|".join(sorted(NUMERALS, key=len, reverse=True))
ENTRADA_DICCIONARI = re.compile(r"^[a-zà-ú][a-zà-ú'\-]*(?:,\s*[a-zà-ú'\-]+)*\s*:\s+(?=[A-ZÀ-Ü])")
_NO_COMPOST = r"(?!-(?:e-|un\b|ua\b|dus\b|dues\b|tres\b|quate\b|cinc\b|sies\b|sèt\b|ueit\b|nau\b))"  # guionet + unitat es un numeral compost (vint-e-cinc), no un separador de clausula
CLAUSULA_NUMERADA = re.compile(
    r"^[—\-]?\s*(?i:(?:" + _NUM_ALT + r")(?:[.\-](?:e-)?(?:" + _NUM_ALT + r"))*)" + _NO_COMPOST
    + r"(?:[.\-—:]\s*)+(?=[A-ZÀ-Ü])")
CLAUSULA_SENSE_PUNT = re.compile(
    r"^(?i:" + _NUM_ALT + r")\s+(?=(?:Eth|Era|Es|Er|Un|Ua|Uns|Ues|En|Ena|Enes|Ath|Ara|As|"
    r"Aguest|Aguesta|Aguesti|Aguestes|Non|Se|Cau|Cada|Toti|Totes|Quan)\b)")
CLAUSULA_NUMERADA_MIN = re.compile(
    r"^[—\-]?\s*(?i:(?:" + _NUM_ALT + r")(?:[.\-](?:e-)?(?:" + _NUM_ALT + r"))*)" + _NO_COMPOST
    + r"(?:[.\-—:]\s*)+(?=[a-zà-ú])")
CONSONANT_SOLTA = re.compile(r"(?<![\w'’])([bcdfghjklmnpqrstvwxyzç])(?![\w'’])")


def _consonant_legitima(text: str, i: int, j: int, lletra: str) -> bool:
    """Els quatre casos on una consonant sola SI que hi pinta."""
    return (lletra == "y"
            or text[j:j + 1] in ".)"
            or text[max(0, i - 1):i] == "("
            or bool(re.search(r"\blletr|\bletr", text[max(0, i - 32):i])))


def _parentesis_desordenats(text: str) -> bool:
    """Un `)` abans del seu `(`, o algun sense parella."""
    n = 0
    for ch in text:
        if ch == "(":
            n += 1
        elif ch == ")":
            n -= 1
            if n < 0:
                return True
    return n != 0


def _caracters_forans(text: str) -> list[str]:
    """Caracters del text fora de l'alfabet permes."""
    return sorted({c for c in text if c not in NOMES_ALFABET})


def _sigla(mot: str) -> bool:
    """Un mot que el TTS no sabra dir i que no s'ha d'ensenyar a escriure."""
    for tros in re.split(r"['\u2019]", mot):
        lletres = [c for c in tros if c.isalpha()]
        if len(lletres) < 2:
            continue
        if any(c.isupper() for c in lletres[1:]) and any(c.islower() for c in lletres):
            return True
        if all(c.isupper() for c in lletres):
            return True
    return False


class PortaQualitat:
    """Regles amb nom."""

    def __init__(self, min_comes_enum: int = 4, max_densitat_comes: float = 0.23,
                 max_propis: float | None = None,
                 max_oov: float | None = None, lexic: set[str] | None = None,
                 min_mots_per_propis: int = 6, sigles: set[str] | None = None):
        """`max_propis` i `max_oov` son `None` per defecte."""
        self.min_comes_enum = min_comes_enum
        self.max_densitat_comes = max_densitat_comes
        self.max_propis = max_propis
        self.max_oov = max_oov
        self.lexic = lexic or set()
        self.sigles = sigles or set()
        self.min_mots_per_propis = min_mots_per_propis
        self.comptador: collections.Counter = collections.Counter()

    def _regles(self, t: str):
        """Nomes FORMA."""
        mots = t.split()
        yield "marca de llista", bool(MARCA_LLISTA.match(t))
        forans = _caracters_forans(t)
        yield f"caracter fora de l'alfabet ({''.join(forans[:4])})", bool(forans)
        yield "xifres", bool(re.search(r"\d", t))
        yield "inicial en minuscula", bool(t[:1].islower())
        cua = t.rstrip("\"»)]'")
        yield "sense puntuacio final", not cua or cua[-1] not in ".!?"
        yield "acaba en abreviatura", bool(ABREVIATURA_FINAL.search(t)
                                          or _ABREV_MINUSCULA.search(t))
        yield "capcalera de pagina", bool(CAPCALERA_PAGINA.search(t))
        yield "parentesis mal niuats", _parentesis_desordenats(t)
        yield "consonants impossibles", bool(CONSONANTS_IMPOSSIBLES.search(t))
        yield "consonant solta (mot partit)", any(
            not _consonant_legitima(t, m.start(), m.end(), m.group(1))
            for m in CONSONANT_SOLTA.finditer(t))
        yield "tot en majuscules", (len(mots) >= 5
                                    and sum(1 for w in mots if w.isupper() and len(w) > 1)
                                    >= 0.6 * len(mots))
        yield "punt sense espai (web, nota al peu, frases enganxades)", bool(PUNT_SENSE_ESPAI.search(t))
        yield "etiqueta de camp (Mot: Text)", bool(ETIQUETA_CAMP.match(t)
                                                   and not VERB_DICCIO.match(t))
        primera = next((c for c in t if c.isalpha()), "")
        yield "inicial en minuscula (dins de cometes)", primera.islower() and t[:1] in "\"«'"
        yield "truncada (acaba en preposicio o conjuncio)", _ultim_mot(t) in FINALS_TRUNCADA
        yield "apostrof perdut", bool(APOSTROF_PERDUT.search(t.lower()))
        yield "cometes desaparellades", t.count('"') % 2 == 1
        yield "parentesis desaparellats", t.count("(") != t.count(")")
        comes = t.count(",")
        densitat = comes / max(1, len(mots))
        elements = [e.split() for e in t.split(",") if e.strip()]
        propis = sum(1 for e in elements
                     if e and (e[0][:1].isupper() or e[0].lower() in NUMERALS))
        yield (f"enumeracio ({comes} comes en {len(mots)} mots, "
               f"{propis}/{len(elements)} elements propis o numerals)",
               comes >= self.min_comes_enum and densitat > self.max_densitat_comes
               and len(elements) >= 5 and propis >= 0.6 * len(elements))
        yield "puntuacio repetida", bool(re.search(r"[.,;:!?]{2,}", t.replace("...", "")))
        yield "apostrof solt", bool(re.search(r"(^|\s)'|'(\s|$)", t))

        mots_min = [m.lower() for m in MOT.findall(t)]
        octs = [" ".join(mots_min[i:i + 8]) for i in range(len(mots_min) - 7)]
        duplicat = len(octs) != len(set(octs))
        contingut = [m for m in mots_min
                     if m not in FUNCIONALS and m not in NUMERALS and len(m) >= 6]
        rep = collections.Counter(contingut).most_common(1)
        yield "text duplicat", duplicat or bool(rep and rep[0][1] >= 4)

        yield "mot sense vocals", bool(MOT_SENSE_VOCALS.search(t))
        yield "marcador de lletres repetides", bool(re.search(r"\b([A-Z])\1{3,}\b|\b([IVX]{2,})\2\b", t))
        yield "rang de numerals enganxat", bool(RANG_ENGANXAT.search(t))
        yield "rang de percentatges enganxat", "per cent-" in t
        yield "milers separats per espai mal expandits", bool(NUMERAL_ZERO.search(t))

        if self.max_propis is not None and len(mots) >= self.min_mots_per_propis:
            propis = sum(1 for m in mots[1:] if m[:1].isupper())
            yield "massa noms propis", (propis >= 4
                                        and propis / (len(mots) - 1) > self.max_propis)

        if self.max_oov is not None and self.lexic and contingut:
            fora = sum(1 for m in contingut if m not in self.lexic)
            yield "massa mots desconeguts", fora / len(contingut) > self.max_oov

    def __call__(self, text: str, presegmentat: bool = False) -> tuple[str | None, str]:
        """Aplica la porta a una frase: (text net o None, motiu)."""
        t = self._uneix_mot_partit(
            self._capcalera_o_etiqueta(normalitza(text, presegmentat)))
        if not t:
            self.comptador["buida"] += 1
            return None, "buida"
        for motiu, falla in self._regles(t):
            if falla:
                self.comptador[motiu.split(" (")[0]] += 1
                return None, motiu
        self.comptador["ACCEPTADA"] += 1
        return t, ""

    def _capcalera_o_etiqueta(self, t: str) -> str:
        """Mot(s) en MAJUSCULES al principi: sigla, capcalera o etiqueta de personatge."""
        for _ in range(3):
            nou = self._una_capa(ACOTACIO.sub("", t))
            if nou == t:
                return t
            t = nou
        return t

    def _una_capa(self, t: str) -> str:
        """Treu una etiqueta o capcalera del principi de la frase."""
        m = ETIQUETA.match(t)
        if not m:
            return t
        run = m.group(1)
        primer = run.split()[0]
        resta = t[m.end():]
        if primer in self.sigles or re.fullmatch(r"[IVXLCDM]+", primer):
            return t
        if resta[:1].islower():
            if self.lexic is None or primer.lower() in self.lexic:
                if len(run.split()) > 3:
                    return t
                cua = " ".join(w if (w in self.sigles or re.fullmatch(r"[IVXLCDM]+", w))
                               else w.lower() for w in run.split()[1:])
                return (primer[:1] + primer[1:].lower() + (" " + cua if cua else "")
                        + self._baixa_majuscules(t[len(run):]))
            mots_run = run.split()
            if len(mots_run) > 1 and len(mots_run[-1]) == 1:
                resta = mots_run[-1] + " " + resta
            return resta[:1].upper() + resta[1:]
        if re.match(r"[A-ZÀ-Ü](?:[a-zà-ú]|['’][a-zà-ú]|[!?,.])|\(", resta):
            return resta
        return t

    def _uneix_mot_partit(self, t: str) -> str:
        """`E r Ajuntament` -> `Er Ajuntament`; `vent f òrt` -> `vent fòrt`."""
        if self.lexic is None:
            return t
        while True:
            for m in CONSONANT_SOLTA.finditer(t):
                k, i, j = m.group(1), m.start(), m.end()
                if _consonant_legitima(t, i, j, k):
                    continue
                esq = re.search(r"([\wà-ú'’]+)\s+$", t[:i])
                if esq and (esq.group(1) + k).lower() in self.lexic:
                    t = t[:esq.start(1)] + esq.group(1) + k + t[j:]
                    break
                dre = re.match(r"\s+([\wà-ú'’]+)", t[j:])
                if dre and (k + dre.group(1)).lower() in self.lexic:
                    t = t[:i] + k + dre.group(1) + t[j + dre.end():]
                    break
            else:
                return t

    def _baixa_majuscules(self, t: str) -> str:
        """Passa a minuscula els mots en majuscules que no son sigles ni romans."""
        def _sub(m):
            """Minuscula si no es sigla ni roma."""
            w = m.group(0)
            if w in self.sigles or re.fullmatch(r"[IVXLCDM]+", w):
                return w
            return w.lower()
        return SIGLA_MAJ.sub(_sub, t)

    def informe(self) -> list[tuple[str, int, float]]:
        """Recompte de motius de rebuig amb percentatges."""
        total = sum(self.comptador.values())
        return [(m, n, 100 * n / max(1, total))
                for m, n in self.comptador.most_common()]


def construeix_lexic(textos, min_frequencia: int = 3) -> set[str]:
    """Mots que surten prou vegades al corpus com per no ser una anomalia."""
    freq: collections.Counter = collections.Counter()
    for t in textos:
        freq.update(m for m in MOT.findall(t) if m[:1].islower())
    return {m for m, n in freq.items() if n >= min_frequencia}

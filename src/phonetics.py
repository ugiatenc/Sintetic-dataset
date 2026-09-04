#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diccionaris fonetics dispersos i cache de decisions de pronunciacio.

No genera fonetica (aixo ho fa `dictionary.ipynb` cridant un LLM) -- es la capa
comuna perque tots els consumidors (`dictionary.ipynb`, `verify_entities.py`,
`generate_sentences.ipynb`) llegeixin i escriguin el fitxer de la mateixa manera:

- `clau(text)` -- cerca insensible a majuscules i accents (`AEMET` == `aemet`).
- `Diccionari.get(entitat)` -- lookup fet amb aquesta clau.
- `Diccionari.fusionar(altre)` -- combina dos diccionaris SENSE trepitjar les
  entrades marcades `revisat: true`.
- `ruta_diccionari(idioma, etapa)` -- nom de fitxer segons etapa (veure sota).

Els diccionaris nomes contenen *overrides*: si una entitat no hi apareix, el TTS
rep la grafia original. La llista completa d'entitats viu als fitxers de candidats,
mai al diccionari. Una cache separada recorda tambe les decisions ``sense_canvi``
per no tornar a pagar-les en cada execucio.

Barrejar-los en un sol fitxer feia que cada regeneracio esborres la revisio
manual anterior; `fusionar()` es qui evita que torni a passar.

Format del fitxer
-------------------
    {"_meta": {...}, "entrades": {"Lamine Yamal": {"fonetica": "Lamín Yamal",
                                                    "revisat": true, "origen": "llm"}}}

`carregar()` tambe accepta el format pla antic (`{"entitat": "fonetica"}`).
"""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_DICCIONARIS = ROOT / "lab/entitats/diccionaris"

ETAPES = ("treball", "final")
CACHE_VERSION = 4
POLITICA_PRONUNCIACIO_VERSION = 8   # 8: `requereix_avaluacio` mira les obertures impossibles


def ruta_diccionari(idioma: str = "castellano", etapa: str = "treball") -> Path:
    """Ruta canonica d'un diccionari. Tenir-la aqui evita que cada consumidor
    escrigui la seva versio del path (ja va passar: `generate_sentences.ipynb`
    apuntava a `lab/diccionari_..._small.json`, que no existeix)."""
    if etapa not in ETAPES:
        raise ValueError(f"etapa ha de ser una de {ETAPES}, no {etapa!r}")
    return DIR_DICCIONARIS / f"diccionari_fonetic_{etapa}_{idioma.lower()}.json"


def ruta_cache(idioma: str = "castellano") -> Path:
    """Cache comuna a treball/final amb decisions positives i negatives."""
    return DIR_DICCIONARIS / f"cache_pronunciacio_{idioma.lower()}.json"


# ---------------------------------------------------------------------------
# Normalitzacio de claus
# ---------------------------------------------------------------------------

def es_identitat(raw: str, fonetica: str) -> bool:
    """Diu si la substitucio no aporta cap canvi de pronunciacio.

    Ignora espais i canvis de caixa normals (`tofás` -> `Tofás`), perquè no canvien
    com sona el TTS. Conserva com a override el pas d'una sigla tota en majuscules a
    acronim lexical (`PSOE` -> `Psoe`, `AEMET` -> `Aemet`) i qualsevol canvi
    d'accent o de lletres.
    """
    raw, fonetica = raw.strip(), fonetica.strip()
    if raw == fonetica:
        return True
    if raw.casefold() != fonetica.casefold():
        return False
    lletres_raw = "".join(c for c in raw if c.isalpha())
    lletres_fonetica = "".join(c for c in fonetica if c.isalpha())
    sigla_a_acronim = (
        len(lletres_raw) >= 2
        and lletres_raw.isupper()
        and not lletres_fonetica.isupper()
    )
    return not sigla_a_acronim


def motiu_respelling_insegur(raw: str, fonetica: str) -> str | None:
    """Detecta corrupcions mecàniques típiques d'un LLM local.

    No jutja si la fonètica és bona; només bloqueja sortides que no poden ser una
    reescriptura castellana segura. Els overrides manuals no passen per aquí.
    """
    raw, fonetica = raw.strip(), fonetica.strip()
    if not fonetica:
        return "sortida_buida"
    # El model ha tornat l'entitat tal qual: es la decisio "no cal tocar-la", no una
    # sortida corrupta. Sense aixo, tota entitat amb un caracter no castella
    # (`Tofaş`, `País Valencià`) es marcava insegura per no haver-la canviat.
    if raw == fonetica:
        return None

    permeses = set("abcdefghijklmnopqrstuvwxyzñáéíóúüABCDEFGHIJKLMNOPQRSTUVWXYZÑÁÉÍÓÚÜ \'-.,")
    introduides = {c for c in fonetica if c.isalpha() and c not in permeses}
    if introduides:
        return f"caracters_no_castellans:{''.join(sorted(introduides))}"

    tokens_raw = raw.split()
    tokens_fon = fonetica.split()
    if (not any(len(t) == 1 and t.islower() for t in tokens_raw)
            and any(len(t) == 1 and t.islower() for t in tokens_fon)):
        return "token_aillat_introduit"

    def base(text: str) -> str:
        text = unicodedata.normalize("NFD", text.casefold())
        return "".join(c for c in text if c.isalpha() and unicodedata.category(c) != "Mn")

    def accents(text: str) -> str:
        """Patro d'accents: quines lletres van accentuades i quines no."""
        text = unicodedata.normalize("NFD", text.casefold())
        return "".join("1" if unicodedata.category(c) == "Mn" else "0"
                       for c in text if c.isalpha() or unicodedata.category(c) == "Mn")

    # La regla vigila que el model no MOGUI un accent que l'entitat ja portava
    # (`Fabián` -> `Fabían`). Afegir-ne un on no n'hi havia es legitim: es la regla dels
    # acronims lexics (`OTAN` -> `Otán`). I comparar `raw != fonetica` en comptes del
    # patro d'accents marcava `París FC` -> `París F C`, que nomes hi insereix un espai.
    te_accent = any(unicodedata.category(c) == "Mn"
                    for c in unicodedata.normalize("NFD", raw))
    if te_accent and base(raw) == base(fonetica) and accents(raw) != accents(fonetica):
        return "altera_accent_existent_sense_canviar_lletres"
    return None


def clau(text: str) -> str:
    """Clau de cerca insensible a majuscules, accents i puntuacio.

    Fa servir `unicodedata` en comptes de la taula `str.maketrans` que dupliquen
    `evaluate.py` i `verify_entities.py`: aquella nomes cobreix vocals accentuades
    castellanes i catalanes, i aqui hi entren entitats amb `ç`, `ñ`, dieresis
    alemanyes o caracters eslaus que la taula deixava passar sense normalitzar
    (`Türkiye` i `Turkiye` eren claus diferents). La `ñ` es preserva expressament:
    col·lapsar-la a `n` fusionaria `Peña` i `Pena`, que son entitats distintes.
    """
    text = text.strip().lower().replace("ñ", "\0")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text).replace("\0", "ñ")
    for c in ".,;:!¡?¿\"'«»…—–-()[]":
        text = text.replace(c, " ")
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Entitats que el round-trip no pot mesurar
# ---------------------------------------------------------------------------

# Caracters que Whisper transcriu de forma inconsistent (a vegades el simbol, a
# vegades la seva lectura): la comparacio literal contra la grafia raw no pot
# donar mai un resultat interpretable.
SIMBOLS_AMBIGUS = set("%ºª°€$£¥&@#/\\+=<>~^*_|")


CONSONANTS = set("bcdfghjklmnpqrstvwxyzñ")
# Uniques obertures de dues consonants que el castella admet a principi de paraula.
OBERTURES_CASTELLANES = {"bl", "br", "cl", "cr", "ch", "dr", "fl", "fr", "gl", "gr",
                         "ll", "pl", "pr", "qu", "tr", "tl", "rr"}


def _obertura_impossible(token: str) -> bool:
    ini = token[:2].casefold()
    return (len(ini) == 2 and all(c in CONSONANTS for c in ini)
            and ini not in OBERTURES_CASTELLANES)


def requereix_avaluacio(entitat: str) -> bool:
    """Guardarrail conservador abans del classificador barat.

    Sigles en majuscules i simbols no es poden descartar com a `sense_canvi`
    sense passar pel model expert. Evita falsos negatius com `FC Barcelona` o
    `EH Bildu`.
    """
    if SIMBOLS_AMBIGUS & set(entitat):
        return True
    tokens = re.findall(r"[^\W_]+", entitat, flags=re.UNICODE)
    if any(len(token) >= 2 and any(c.isalpha() for c in token)
           and token.isupper() for token in tokens):
        return True

    # Obertura de paraula impossible en castella: nomes admet consonant sola o
    # consonant + l/r. Qualsevol altre parell inicial ve d'una altra llengua i el TTS
    # hi ensopega. Abans nomes hi havia `sp-` escrit a ma i queien fora `Mbappé`
    # (`mb-`), `Ngannou` (`ng-`) o `Pfizer` (`pf-`), que son el cas central d'aquesta
    # funcio; les regles del prompt ja les citaven, pero el guardarrail no.
    if any(_obertura_impossible(t) for t in tokens):
        return True

    # Patrons que un TTS castellà acostuma a llegir literalment, però que en noms
    # estrangers solen representar altres sons. És deliberadament sensible: un fals
    # positiu només arriba al model local; un fals negatiu deixa una mala pronunciació.
    text = " ".join(tokens).casefold()
    patrons_estrangers = (
        r"(^|\s)sp",       # Spotify -> Espótifai
        r"sh|th|ph|wh",    # Shia, Smith, Philippe, White
        r"[aeiou]ng",      # Angeldahl, Harrington
        r"rr",              # Harrison/Harringson (h inicial aspirada)
        r"ou|ow|ee|oo",    # Hauser, Brown, Green, Boone
        r"[bcdfghjklmnpqrstvwxyz]y($|\s)",
        r"w|krasz|sz|cz|tsch|sch",
    )
    return any(re.search(p, text) for p in patrons_estrangers)


def es_expansio(raw: str, fonetica: str) -> bool:
    """Diu si la fonetica LLEGEIX l'entitat en comptes de reescriure-la.

    `EE.UU.` -> "Estados Unidos" i `%` -> "por ciento" son expansions: el TTS diu la
    lectura completa i Whisper escriu la lectura, aixi que comparar-ho literalment
    contra la grafia crua dona `tasa_error = 1.0` garantida. Es el mateix problema que
    `verificable_round_trip` ja detecta per als simbols, pero originat al diccionari en
    comptes de a la grafia.

    Un respelling conserva el material de la paraula (`Mbappé` -> `Embapé`); una
    expansio l'inventa gairebe tot.
    """
    def nucli(text: str) -> str:
        return clau(text).replace(" ", "")

    a, b = nucli(raw), nucli(fonetica)
    if not a or not b:
        return False
    # Comparacio de SEQUENCIA, no de conjunt de lletres: `EE.UU.` i `Estados Unidos`
    # comparteixen les dues lletres de la sigla, aixi que un conjunt els donava per
    # identics. Les sigles deletrejades ("UE" -> "U E") conserven la sequencia sencera
    # en treure els espais, de manera que no queden atrapades aqui.
    return difflib.SequenceMatcher(None, a, b).ratio() < 0.5


def verificable_round_trip(entitat: str) -> tuple[bool, str | None]:
    """Diu si te sentit passar una entitat pel round-trip TTS -> ASR.

    `%`, `ºC` i `km/h` estan al diccionari com a expansions ("por ciento",
    "grados Celsius"), de manera que el TTS diu la lectura i Whisper escriu el
    simbol o la lectura segons li convingui. Comparar literalment contra `%`
    donava `tasa_error = 1.0` garantida -- un fallo inventat que contaminava la
    mitjana i podia colar simbols a la llista final com si fossin les entitats
    mes dificils del corpus.
    """
    net = entitat.strip()
    if not net:
        return False, "buida"
    if SIMBOLS_AMBIGUS & set(net):
        return False, "conte_simbol_ambigu"
    if not any(c.isalpha() for c in net):
        return False, "sense_lletres"
    return True, None


# ---------------------------------------------------------------------------
# Aplicacio determinista d'un diccionari sobre text
# ---------------------------------------------------------------------------
#
# `generate_sentences.ipynb` construeix `tts_text` nomes amb aixo: el LLM escriu la frase
# amb ortografia real i la fonetica l'aplica un find-and-replace. Es la propietat que
# fa el dataset reproduible -- `tts_text` es una funcio pura de `raw_text` i del
# diccionari -- i la que garanteix que una entitat soni igual a totes les frases.
#
# Deixant la reescriptura al model, de 317 fragments reescrits n'hi havia 40 amb dues o
# tres grafies i unes quantes que canviaven la identitat de l'entitat (`Fernandes` ->
# `Fernández`). No es un problema de prompt: es que ningu decidia res una sola vegada.

def patro_entitat(entitat: str) -> str:
    """Regex que localitza una entitat dins d'un text.

    Els limits de paraula nomes es posen si el caracter del marge es alfanumeric:
    entitats com `%` o `km/h` no tenen frontera `\\w` a aquell costat i el limit les
    faria introbables.
    """
    cos = r"\s+".join(re.escape(part) for part in entitat.split())
    pre = r"(?<!\w)" if entitat[:1].isalnum() else ""
    post = r"(?!\w)" if entitat[-1:].isalnum() else ""
    return f"{pre}{cos}{post}"


def compilar_regex(claus) -> "re.Pattern | None":
    """Una unica regex amb totes les entitats en alternanca, de mes llarga a mes curta.

    Una sola passada evita l'efecte cascada (que una substitucio ja aplicada torni a
    ser capturada per una entitat mes curta) i l'ordre per longitud garanteix que guanyi
    la coincidencia mes especifica (`FC Barcelona` abans que `FC`).
    """
    patrons = [patro_entitat(c) for c in sorted(claus, key=len, reverse=True) if c]
    return re.compile("|".join(patrons), re.IGNORECASE) if patrons else None


def aplicar_diccionari(text: str, diccionari_pla: dict[str, str],
                       regex: "re.Pattern | None" = None) -> tuple[str, list[str]]:
    """Substitueix al text les entitats del diccionari per la seva fonetica.

    Determinista i idempotent: la mateixa entitat rep sempre la mateixa transcripcio a
    totes les frases del dataset. Es la propietat que es va perdre quan la reescriptura
    la feia el LLM frase a frase.

    Retorna `(text_resultant, entitats_aplicades)`.
    """
    if regex is None:
        regex = compilar_regex(diccionari_pla)
    if regex is None:
        return text, []
    mapa = {clau(k): v for k, v in diccionari_pla.items()}
    aplicades: list[str] = []

    def _substituir(m):
        original = m.group(0)
        fonetica = mapa.get(clau(original))
        if fonetica is None:
            return original
        # Es conserva la minuscula inicial quan l'entitat surt a mitja frase
        # ('una startup' -> 'una startap'), pero mai a les sigles deletrejades
        # ('U E'), on les majuscules formen part de la transcripcio.
        if original.islower() and not fonetica.isupper():
            fonetica = fonetica[:1].lower() + fonetica[1:]
        # Els simbols van enganxats al numero ('30%') pero la seva transcripcio es
        # una paraula: cal separar-la o quedaria '30por ciento'.
        inici, fi = m.span()
        if inici > 0 and text[inici - 1].isalnum() and fonetica[:1].isalnum():
            fonetica = " " + fonetica
        if fi < len(text) and text[fi].isalnum() and fonetica[-1:].isalnum():
            fonetica = fonetica + " "
        aplicades.append(original)
        return fonetica

    return regex.sub(_substituir, text), aplicades


# Unitats i simbols que el TTS ha de llegir en paraules. Es fan abans que les xifres
# perque el numero que els precedeix forma part de la lectura.
LECTURES_SIMBOL = (
    (re.compile(r"\s*%"), " por ciento"),
    (re.compile(r"\s*º\s*C\b"), " grados Celsius"),
    (re.compile(r"\s*km\s*/\s*h\b"), " kilómetros por hora"),
    (re.compile(r"\s*€"), " euros"),
)


def expandir_xifres(text: str, idioma_num2words: str = "es") -> str:
    """Escriu en lletres els numeros i els simbols que el TTS llegiria malament.

    Es un pas determinista i a part de la reescriptura d'entitats: abans ho feia el LLM
    dins de `texto_tts`, i per tant nomes a vegades -- `hasta 2026` es va expandir en
    una frase i no en la seguent. Aqui val per a totes.

    NO toca els digits enganxats a lletres (`G7`, `COVID-19`): alla el numero forma part
    del nom i qui decideix com es llegeix es el diccionari, no aquesta funcio.
    """
    try:
        from num2words import num2words
    except ImportError:            # pragma: no cover - dependencia declarada al README
        return text
    for patro, lectura in LECTURES_SIMBOL:
        text = patro.sub(lectura, text)

    def _llegir(m):
        obre, xifres, tanca = m.groups()
        return obre + num2words(int(xifres), lang=idioma_num2words) + tanca

    # El token SENCER ha de ser el numero (amb la seva puntuacio): aixi `G7` i
    # `COVID-19` queden intactes -- alla el digit forma part del nom.
    return re.sub(r"(?<!\S)([¿¡(\"'«]*)(\d+)([.,;:!?)\"'»]*)(?!\S)", _llegir, text)


# ---------------------------------------------------------------------------
# Diccionari
# ---------------------------------------------------------------------------

class Diccionari:
    """Mapa entitat -> fonetica amb cerca insensible a majuscules i accents.

    Conserva la grafia original de cada entrada (`entrades`) perque el fitxer
    segueixi sent llegible per una persona; nomes la CERCA es normalitza.
    """

    def __init__(self, entrades: dict[str, dict] | None = None, meta: dict | None = None):
        self.entrades: dict[str, dict] = entrades or {}
        self.meta: dict = meta or {}
        self._index: dict[str, str] = {}
        self._reindexar()

    def _reindexar(self) -> None:
        self._index = {}
        for grafia in self.entrades:
            k = clau(grafia)
            # Col·lisio real (dues grafies amb fonetiques diferents): guanya la
            # primera i s'avisa, en comptes de deixar que l'ordre d'insercio
            # decideixi en silenci quina fonetica sentira el TTS.
            if k in self._index:
                if self._fonetica(self._index[k]) != self._fonetica(grafia):
                    print(f"  AVIS diccionari: '{grafia}' i '{self._index[k]}' comparteixen clau "
                          f"'{k}' amb fonetiques diferents; es fa servir la primera")
                continue
            self._index[k] = grafia

    def _fonetica(self, grafia: str) -> str | None:
        """Fonetica d'una grafia concreta d'aquest diccionari."""
        entrada = self.entrades.get(grafia)
        return entrada.get("fonetica") if isinstance(entrada, dict) else None

    def get(self, entitat: str, defecte: str | None = None) -> str | None:
        """Fonetica de `entitat`, o `defecte`. Aquesta es la crida que abans
        fallava amb 'AEMET' quan el diccionari tenia 'aemet'."""
        grafia = self._index.get(clau(entitat))
        if grafia is None:
            return defecte
        return self.entrades[grafia].get("fonetica") or defecte

    def registre(self, entitat: str) -> dict | None:
        """Entrada sencera (fonetica + revisat + origen), per qui necessiti saber
        si una transcripcio ja ha passat per revisio humana."""
        grafia = self._index.get(clau(entitat))
        return self.entrades[grafia] if grafia else None

    def revisat(self, entitat: str) -> bool:
        r = self.registre(entitat)
        return bool(r and r.get("revisat"))

    def __contains__(self, entitat: str) -> bool:
        return clau(entitat) in self._index

    def __len__(self) -> int:
        return len(self.entrades)

    def pla(self) -> dict[str, str]:
        """Vista `{grafia: fonetica}`, per al codi que encara espera el format pla."""
        return {g: e["fonetica"] for g, e in self.entrades.items() if e.get("fonetica")}

    # -- escriptura --------------------------------------------------------

    def posar(self, entitat: str, fonetica: str, origen: str = "llm",
              revisat: bool = False, **extra) -> None:
        """`extra` desa camps d'auditoria al costat de la fonetica (quantes vegades
        s'ha proposat, quines variants, per que s'ha descartat). Els consumidors nomes
        llegeixen `fonetica`, aixi que afegir-n'hi no trenca res."""
        grafia = self._index.get(clau(entitat), entitat)
        self.entrades[grafia] = {"fonetica": fonetica, "revisat": revisat,
                                 "origen": origen, **extra}
        self._index[clau(grafia)] = grafia

    def eliminar(self, entitat: str) -> bool:
        """Elimina una entrada mantenint coherent l'index normalitzat."""
        grafia = self._index.get(clau(entitat))
        if grafia is None:
            return False
        del self.entrades[grafia]
        self._reindexar()
        return True

    def fusionar(self, altre: "Diccionari", protegir_revisats: bool = True) -> dict[str, int]:
        """Incorpora `altre` sobre aquest diccionari.

        `protegir_revisats` es la regla que fa segur regenerar: una entrada que
        una persona ja ha validat no la trepitja una tanda automatica del LLM.
        Sense aixo cada regeneracio de `dictionary.ipynb` esborrava la revisio
        manual anterior, que es exactament el que va passar amb el fitxer de 24
        entrades que va quedar congelat mesos.
        """
        stats = {"noves": 0, "actualitzades": 0, "protegides": 0}
        for grafia, entrada in altre.entrades.items():
            # `in` ja fa la comparacio per clau normalitzada; el recorregut lineal que
            # hi havia aqui repetia la mateixa comprovacio i feia la fusio quadratica.
            if grafia not in self:
                self.entrades[grafia] = dict(entrada)
                self._index[clau(grafia)] = grafia
                stats["noves"] += 1
            elif protegir_revisats and self.revisat(grafia):
                stats["protegides"] += 1
            else:
                existent = self._index.get(clau(grafia), grafia)
                self.entrades[existent] = dict(entrada)
                stats["actualitzades"] += 1
        self._reindexar()
        return stats

    def subconjunt(self, entitats: list[str]) -> "Diccionari":
        """Diccionari nomes amb `entitats` (les que en tinguin entrada aqui).

        Es el pas treball -> final: sembra el diccionari final amb la fonetica ja
        generada per als candidats que han superat el llindar, perque la revisio
        manual comenci des d'un esborrany i no des de zero.
        """
        nou = Diccionari(meta=dict(self.meta))
        for e in entitats:
            grafia = self._index.get(clau(e))
            if grafia:
                nou.entrades[grafia] = dict(self.entrades[grafia])
        nou._reindexar()
        return nou


def carregar(path: Path | str, silenciar_absent: bool = True) -> Diccionari:
    """Carrega un diccionari. Accepta el format estructurat i el pla antic."""
    path = Path(path)
    if not path.exists():
        if not silenciar_absent:
            raise FileNotFoundError(path)
        return Diccionari()

    dades = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(dades, dict) and "entrades" in dades:
        entrades = {
            g: (e if isinstance(e, dict) else {"fonetica": e, "revisat": False, "origen": "desconegut"})
            for g, e in dades["entrades"].items()
        }
        return Diccionari(entrades, dades.get("_meta", {}))

    # Format pla antic: `{"Lamine Yamal": "Lamín Yamal"}`. Es marca `revisat:
    # true` perque aquests fitxers venen de l'epoca en que s'editaven a ma i
    # perdre'ls en una regeneracio seria el fallo que aquest modul evita.
    entrades = {
        g: {"fonetica": v, "revisat": True, "origen": "format_pla_migrat"}
        for g, v in dades.items() if isinstance(v, str) and not g.startswith("_")
    }
    return Diccionari(entrades, {"migrat_de": str(path.name)})


def desar(diccionari: Diccionari, path: Path | str, idioma: str = "castellano",
          etapa: str = "treball", **meta_extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        **diccionari.meta,
        "idioma": idioma,
        "etapa": etapa,
        "generat": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_entrades": len(diccionari.entrades),
        "n_revisades": sum(1 for e in diccionari.entrades.values() if e.get("revisat")),
        **meta_extra,
    }
    sortida = {"_meta": meta, "entrades": diccionari.entrades}
    path.write_text(json.dumps(sortida, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cache de decisions: evita tornar a consultar entitats sense override
# ---------------------------------------------------------------------------

ESTATS_CACHE = ("sense_canvi", "pendent_fonetica", "amb_fonetica")


class CachePronunciacio:
    """Decisions de pronunciacio, incloses les negatives.

    ``sense_canvi`` no entra mai al diccionari, pero queda cachejat. D'aquesta
    manera un diccionari dispers no confon "ja decidit: usa el raw" amb "encara
    no processat".
    """

    def __init__(self, entrades: dict[str, dict] | None = None, meta: dict | None = None):
        self.entrades: dict[str, dict] = entrades or {}
        self.meta = meta or {}
        self._index: dict[str, str] = {}
        self._reindexar()

    def _reindexar(self) -> None:
        self._index = {}
        for grafia in self.entrades:
            self._index.setdefault(clau(grafia), grafia)

    def __contains__(self, entitat: str) -> bool:
        return clau(entitat) in self._index

    def __len__(self) -> int:
        return len(self.entrades)

    def registre(self, entitat: str) -> dict | None:
        grafia = self._index.get(clau(entitat))
        return self.entrades.get(grafia) if grafia else None

    def resolta(self, entitat: str) -> bool:
        registre = self.registre(entitat)
        return bool(registre and registre.get("estat") in ("sense_canvi", "amb_fonetica"))

    def necessita_fonetica(self, entitat: str) -> bool | None:
        registre = self.registre(entitat)
        if not registre:
            return None
        if registre.get("estat") == "sense_canvi":
            return False
        if registre.get("estat") in ("pendent_fonetica", "amb_fonetica"):
            return True
        return None

    def fonetica(self, entitat: str) -> str | None:
        registre = self.registre(entitat)
        return registre.get("fonetica") if registre else None

    def _posar(self, entitat: str, entrada: dict) -> None:
        grafia = self._index.get(clau(entitat), entitat)
        self.entrades[grafia] = entrada
        self._index[clau(grafia)] = grafia

    def eliminar(self, entitat: str) -> bool:
        """Oblida una decisió perquè torni a passar pel classificador."""
        grafia = self._index.get(clau(entitat))
        if grafia is None:
            return False
        del self.entrades[grafia]
        self._reindexar()
        return True

    def podar(self, entitats_actives: list[str]) -> int:
        """Elimina decisions de grafies que ja no pertanyen a cap etapa activa."""
        actives = {clau(e) for e in entitats_actives}
        antigues = [g for g in self.entrades if clau(g) not in actives]
        for grafia in antigues:
            del self.entrades[grafia]
        if antigues:
            self._reindexar()
        return len(antigues)

    def invalidar_automatiques(self, versio_politica: int) -> int:
        """Oblida les decisions automatiques quan canvia la politica de pronunciacio.

        Un `sense_canvi` o una fonetica que va decidir un LLM amb un prompt anterior no
        es evidencia eterna: en canviar les regles ha de tornar a passar pel
        classificador. Les entrades marcades `revisat` (revisio humana) no es toquen mai.

        Abans aixo es feia al notebook mirant si l'origen començava per `llm:qwen`, de
        manera que les entrades d'un model anterior (`llm:gpt-4o`) sobrevivien a totes
        les invalidacions i el diccionari deia que l'havia generat un model que no
        havia vist mai aquelles entitats.
        """
        if self.meta.get("politica_pronunciacio") == versio_politica:
            return 0
        obsoletes = [g for g, r in self.entrades.items()
                     if not r.get("revisat") and str(r.get("origen", "")).startswith("llm")]
        for grafia in obsoletes:
            self.eliminar(grafia)
        self.meta["politica_pronunciacio"] = versio_politica
        return len(obsoletes)

    def posar_sense_canvi(self, entitat: str, origen: str = "llm_decisio") -> None:
        self._posar(entitat, {"estat": "sense_canvi", "origen": origen})

    def posar_pendent(self, entitat: str, origen: str = "llm_decisio") -> None:
        self._posar(entitat, {"estat": "pendent_fonetica", "origen": origen})

    def posar_fonetica(self, entitat: str, fonetica: str, origen: str = "llm",
                       revisat: bool = False) -> None:
        fonetica = fonetica.strip()
        if not fonetica or es_identitat(entitat, fonetica):
            self.posar_sense_canvi(entitat, origen=origen)
            return
        self._posar(entitat, {"estat": "amb_fonetica", "fonetica": fonetica,
                              "origen": origen, "revisat": revisat})

    def importar_diccionari(self, diccionari: Diccionari) -> None:
        """Migra resultats ja pagats, incloses les identitats del format antic."""
        for grafia, entrada in diccionari.entrades.items():
            fonetica = entrada.get("fonetica", "")
            if not fonetica or es_identitat(grafia, fonetica):
                if grafia not in self:
                    self.posar_sense_canvi(grafia, origen="migrat_diccionari")
            else:
                # Una revisio manual sempre mana; una automatica existent nomes
                # omple entrades absents per no trepitjar decisions posteriors.
                actual = self.registre(grafia)
                if entrada.get("revisat") or actual is None:
                    self.posar_fonetica(grafia, fonetica,
                                        origen=entrada.get("origen", "migrat_diccionari"),
                                        revisat=bool(entrada.get("revisat")))

    def diccionari(self, entitats: list[str]) -> Diccionari:
        """Materialitza nomes els overrides de les entitats indicades."""
        resultat = Diccionari()
        for entitat in entitats:
            registre = self.registre(entitat)
            if registre and registre.get("estat") == "amb_fonetica":
                resultat.posar(entitat, registre["fonetica"],
                                origen=registre.get("origen", "cache"),
                                revisat=bool(registre.get("revisat")))
        return resultat


def carregar_cache(path: Path | str) -> CachePronunciacio:
    path = Path(path)
    if not path.exists():
        return CachePronunciacio()
    dades = json.loads(path.read_text(encoding="utf-8"))
    entrades = dades.get("entrades", {}) if isinstance(dades, dict) else {}
    netes = {}
    for grafia, entrada in entrades.items():
        if not isinstance(entrada, dict) or entrada.get("estat") not in ESTATS_CACHE:
            continue
        entrada = dict(entrada)
        if (entrada.get("estat") == "amb_fonetica"
                and es_identitat(grafia, entrada.get("fonetica", ""))):
            entrada = {"estat": "sense_canvi", "origen": entrada.get("origen", "cache")}
        netes[grafia] = entrada
    return CachePronunciacio(netes, dades.get("_meta", {}))


def desar_cache(cache: CachePronunciacio, path: Path | str, idioma: str = "castellano",
                **meta_extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recomptes = {estat: 0 for estat in ESTATS_CACHE}
    for entrada in cache.entrades.values():
        estat = entrada.get("estat")
        if estat in recomptes:
            recomptes[estat] += 1
    meta = {**cache.meta, "versio": CACHE_VERSION, "idioma": idioma,
            "generat": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_decisions": len(cache), "recomptes": recomptes, **meta_extra}
    path.write_text(json.dumps({"_meta": meta, "entrades": cache.entrades},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return path

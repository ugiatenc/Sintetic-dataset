#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diccionaris fonetics, cache de decisions de pronunciacio i substitucio d'entitats al text."""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_DATA = ROOT / "data"
CARPETA_IDIOMA = {"castellano": "es", "aranes": "aranes"}

ETAPES = ("treball", "final")
CACHE_VERSION = 4
POLITICA_PRONUNCIACIO_VERSION = 8


def dir_diccionaris(idioma: str) -> Path:
    """Carpeta dels diccionaris d'un idioma."""
    idioma = idioma.lower()
    return DIR_DATA / CARPETA_IDIOMA.get(idioma, idioma) / "entitats"


def ruta_diccionari(idioma: str = "castellano", etapa: str = "treball") -> Path:
    """Ruta canonica d'un diccionari."""
    if etapa not in ETAPES:
        raise ValueError(f"etapa ha de ser una de {ETAPES}, no {etapa!r}")
    return dir_diccionaris(idioma) / f"diccionari_fonetic_{etapa}_{idioma.lower()}.json"


def ruta_cache(idioma: str = "castellano") -> Path:
    """Cache comuna a treball/final amb decisions positives i negatives."""
    return dir_diccionaris(idioma) / f"cache_pronunciacio_{idioma.lower()}.json"


def es_identitat(raw: str, fonetica: str) -> bool:
    """Diu si la substitucio no aporta cap canvi de pronunciacio."""
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
    """Detecta corrupcions mecàniques típiques d'un LLM local."""
    raw, fonetica = raw.strip(), fonetica.strip()
    if not fonetica:
        return "sortida_buida"
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
        """Forma base d'un text: minuscules, sense accents ni signes."""
        text = unicodedata.normalize("NFD", text.casefold())
        return "".join(c for c in text if c.isalpha() and unicodedata.category(c) != "Mn")

    def accents(text: str) -> str:
        """Patro d'accents: quines lletres van accentuades i quines no."""
        text = unicodedata.normalize("NFD", text.casefold())
        return "".join("1" if unicodedata.category(c) == "Mn" else "0"
                       for c in text if c.isalpha() or unicodedata.category(c) == "Mn")

    te_accent = any(unicodedata.category(c) == "Mn"
                    for c in unicodedata.normalize("NFD", raw))
    if te_accent and base(raw) == base(fonetica) and accents(raw) != accents(fonetica):
        return "altera_accent_existent_sense_canviar_lletres"
    return None


def clau(text: str) -> str:
    """Clau de cerca insensible a majuscules, accents i puntuacio."""
    text = text.strip().lower().replace("ñ", "\0")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text).replace("\0", "ñ")
    for c in ".,;:!¡?¿\"'«»…—–-()[]":
        text = text.replace(c, " ")
    return " ".join(text.split())


SIMBOLS_AMBIGUS = set("%ºª°€$£¥&@#/\\+=<>~^*_|")


CONSONANTS = set("bcdfghjklmnpqrstvwxyzñ")
OBERTURES_CASTELLANES = {"bl", "br", "cl", "cr", "ch", "dr", "fl", "fr", "gl", "gr",
                         "ll", "pl", "pr", "qu", "tr", "tl", "rr"}


def _obertura_impossible(token: str) -> bool:
    """Si un token comenca per un grup de consonants impossible."""
    ini = token[:2].casefold()
    return (len(ini) == 2 and all(c in CONSONANTS for c in ini)
            and ini not in OBERTURES_CASTELLANES)


def requereix_avaluacio(entitat: str) -> bool:
    """Guardarrail conservador abans del classificador barat."""
    if SIMBOLS_AMBIGUS & set(entitat):
        return True
    tokens = re.findall(r"[^\W_]+", entitat, flags=re.UNICODE)
    if any(len(token) >= 2 and any(c.isalpha() for c in token)
           and token.isupper() for token in tokens):
        return True

    if any(_obertura_impossible(t) for t in tokens):
        return True

    text = " ".join(tokens).casefold()
    patrons_estrangers = (
        r"(^|\s)sp",
        r"sh|th|ph|wh",
        r"[aeiou]ng",
        r"rr",
        r"ou|ow|ee|oo",
        r"[bcdfghjklmnpqrstvwxyz]y($|\s)",
        r"w|krasz|sz|cz|tsch|sch",
    )
    return any(re.search(p, text) for p in patrons_estrangers)


def es_expansio(raw: str, fonetica: str) -> bool:
    """Diu si la fonetica LLEGEIX l'entitat en comptes de reescriure-la."""
    def nucli(text: str) -> str:
        """Clau sense espais."""
        return clau(text).replace(" ", "")

    a, b = nucli(raw), nucli(fonetica)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() < 0.5


def verificable_round_trip(entitat: str) -> tuple[bool, str | None]:
    """Diu si te sentit passar una entitat pel round-trip TTS -> ASR."""
    net = entitat.strip()
    if not net:
        return False, "buida"
    if SIMBOLS_AMBIGUS & set(net):
        return False, "conte_simbol_ambigu"
    if not any(c.isalpha() for c in net):
        return False, "sense_lletres"
    return True, None


def patro_entitat(entitat: str) -> str:
    """Regex que localitza una entitat dins d'un text."""
    cos = r"\s+".join(re.escape(part) for part in entitat.split())
    pre = r"(?<!\w)" if entitat[:1].isalnum() else ""
    post = r"(?!\w)" if entitat[-1:].isalnum() else ""
    return f"{pre}{cos}{post}"


def compilar_regex(claus) -> "re.Pattern | None":
    """Una unica regex amb totes les entitats en alternanca, de mes llarga a mes curta."""
    patrons = [patro_entitat(c) for c in sorted(claus, key=len, reverse=True) if c]
    return re.compile("|".join(patrons), re.IGNORECASE) if patrons else None


def aplicar_diccionari(text: str, diccionari_pla: dict[str, str],
                       regex: "re.Pattern | None" = None) -> tuple[str, list[str]]:
    """Substitueix al text les entitats del diccionari per la seva fonetica."""
    if regex is None:
        regex = compilar_regex(diccionari_pla)
    if regex is None:
        return text, []
    mapa = {clau(k): v for k, v in diccionari_pla.items()}
    aplicades: list[str] = []

    def _substituir(m):
        """Substitueix una entitat per la seva fonetica."""
        original = m.group(0)
        fonetica = mapa.get(clau(original))
        if fonetica is None:
            return original
        if original.islower() and not fonetica.isupper():
            fonetica = fonetica[:1].lower() + fonetica[1:]
        inici, fi = m.span()
        if inici > 0 and text[inici - 1].isalnum() and fonetica[:1].isalnum():
            fonetica = " " + fonetica
        if fi < len(text) and text[fi].isalnum() and fonetica[-1:].isalnum():
            fonetica = fonetica + " "
        aplicades.append(original)
        return fonetica

    return regex.sub(_substituir, text), aplicades


LECTURES_SIMBOL = (
    (re.compile(r"\s*%"), " por ciento"),
    (re.compile(r"\s*º\s*C\b"), " grados Celsius"),
    (re.compile(r"\s*km\s*/\s*h\b"), " kilómetros por hora"),
    (re.compile(r"\s*€"), " euros"),
)


def expandir_xifres(text: str, idioma_num2words: str = "es") -> str:
    """Escriu en lletres els numeros i els simbols que el TTS llegiria malament."""
    try:
        from num2words import num2words
    except ImportError:            # pragma: no cover - dependencia declarada al README
        return text
    for patro, lectura in LECTURES_SIMBOL:
        text = patro.sub(lectura, text)

    def _llegir(m):
        """Escriu una xifra en lletres, conservant el que l'envolta."""
        obre, xifres, tanca = m.groups()
        return obre + num2words(int(xifres), lang=idioma_num2words) + tanca

    return re.sub(r"(?<!\S)([¿¡(\"'«]*)(\d+)([.,;:!?)\"'»]*)(?!\S)", _llegir, text)


class Diccionari:
    """Mapa entitat -> fonetica amb cerca insensible a majuscules i accents."""

    def __init__(self, entrades: dict[str, dict] | None = None, meta: dict | None = None):
        """Diccionari fonetic en memoria, amb index per clau."""
        self.entrades: dict[str, dict] = entrades or {}
        self.meta: dict = meta or {}
        self._index: dict[str, str] = {}
        self._reindexar()

    def _reindexar(self) -> None:
        """Reconstrueix l'index per clau."""
        self._index = {}
        for grafia in self.entrades:
            k = clau(grafia)
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
        """Fonetica de `entitat`, o `defecte`."""
        grafia = self._index.get(clau(entitat))
        if grafia is None:
            return defecte
        return self.entrades[grafia].get("fonetica") or defecte

    def registre(self, entitat: str) -> dict | None:
        """Entrada sencera (fonetica + revisat + origen)."""
        grafia = self._index.get(clau(entitat))
        return self.entrades[grafia] if grafia else None

    def revisat(self, entitat: str) -> bool:
        """Si l'entrada esta marcada com revisada."""
        r = self.registre(entitat)
        return bool(r and r.get("revisat"))

    def __contains__(self, entitat: str) -> bool:
        """Si l'entitat es al diccionari."""
        return clau(entitat) in self._index

    def __len__(self) -> int:
        """Nombre d'entrades."""
        return len(self.entrades)

    def pla(self) -> dict[str, str]:
        """Vista `{grafia: fonetica}`, per al codi que encara espera el format pla."""
        return {g: e["fonetica"] for g, e in self.entrades.items() if e.get("fonetica")}


    def posar(self, entitat: str, fonetica: str, origen: str = "llm",
              revisat: bool = False, **extra) -> None:
        """`extra` desa camps d'auditoria al costat de la fonetica (quantes vegades s'ha proposat, quines variants."""
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
        """Incorpora `altre` sobre aquest diccionari."""
        stats = {"noves": 0, "actualitzades": 0, "protegides": 0}
        for grafia, entrada in altre.entrades.items():
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
        """Diccionari nomes amb `entitats` (les que en tinguin entrada aqui)."""
        nou = Diccionari(meta=dict(self.meta))
        for e in entitats:
            grafia = self._index.get(clau(e))
            if grafia:
                nou.entrades[grafia] = dict(self.entrades[grafia])
        nou._reindexar()
        return nou


def carregar(path: Path | str, silenciar_absent: bool = True) -> Diccionari:
    """Carrega un diccionari."""
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

    entrades = {
        g: {"fonetica": v, "revisat": True, "origen": "format_pla_migrat"}
        for g, v in dades.items() if isinstance(v, str) and not g.startswith("_")
    }
    return Diccionari(entrades, {"migrat_de": str(path.name)})


def desar(diccionari: Diccionari, path: Path | str, idioma: str = "castellano",
          etapa: str = "treball", **meta_extra) -> Path:
    """Desa un diccionari fonetic a JSON."""
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


ESTATS_CACHE = ("sense_canvi", "pendent_fonetica", "amb_fonetica")


class CachePronunciacio:
    """Decisions de pronunciacio, incloses les negatives."""

    def __init__(self, entrades: dict[str, dict] | None = None, meta: dict | None = None):
        """Cache de decisions de pronunciacio, amb index per clau."""
        self.entrades: dict[str, dict] = entrades or {}
        self.meta = meta or {}
        self._index: dict[str, str] = {}
        self._reindexar()

    def _reindexar(self) -> None:
        """Reconstrueix l'index per clau."""
        self._index = {}
        for grafia in self.entrades:
            self._index.setdefault(clau(grafia), grafia)

    def __contains__(self, entitat: str) -> bool:
        """Si l'entitat es a la cache."""
        return clau(entitat) in self._index

    def __len__(self) -> int:
        """Nombre d'entrades."""
        return len(self.entrades)

    def registre(self, entitat: str) -> dict | None:
        """Registre d'una entitat, o None."""
        grafia = self._index.get(clau(entitat))
        return self.entrades.get(grafia) if grafia else None

    def resolta(self, entitat: str) -> bool:
        """Si l'entitat ja te decisio final."""
        registre = self.registre(entitat)
        return bool(registre and registre.get("estat") in ("sense_canvi", "amb_fonetica"))

    def necessita_fonetica(self, entitat: str) -> bool | None:
        """Si l'entitat necessita fonetica (None si no se sap)."""
        registre = self.registre(entitat)
        if not registre:
            return None
        if registre.get("estat") == "sense_canvi":
            return False
        if registre.get("estat") in ("pendent_fonetica", "amb_fonetica"):
            return True
        return None

    def fonetica(self, entitat: str) -> str | None:
        """Fonetica desada d'una entitat, o None."""
        registre = self.registre(entitat)
        return registre.get("fonetica") if registre else None

    def _posar(self, entitat: str, entrada: dict) -> None:
        """Escriu l'entrada d'una entitat respectant la grafia indexada."""
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
        """Oblida les decisions automatiques quan canvia la politica de pronunciacio."""
        if self.meta.get("politica_pronunciacio") == versio_politica:
            return 0
        obsoletes = [g for g, r in self.entrades.items()
                     if not r.get("revisat") and str(r.get("origen", "")).startswith("llm")]
        for grafia in obsoletes:
            self.eliminar(grafia)
        self.meta["politica_pronunciacio"] = versio_politica
        return len(obsoletes)

    def posar_sense_canvi(self, entitat: str, origen: str = "llm_decisio") -> None:
        """Marca l'entitat com a sense canvi."""
        self._posar(entitat, {"estat": "sense_canvi", "origen": origen})

    def posar_pendent(self, entitat: str, origen: str = "llm_decisio") -> None:
        """Marca l'entitat com a pendent de fonetica."""
        self._posar(entitat, {"estat": "pendent_fonetica", "origen": origen})

    def posar_fonetica(self, entitat: str, fonetica: str, origen: str = "llm",
                       revisat: bool = False) -> None:
        """Desa la fonetica d'una entitat (o sense canvi si es identica)."""
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
    """Carrega la cache de pronunciacions (buida si no existeix)."""
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
    """Desa la cache de pronunciacions."""
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crides a LLM amb structured outputs: client, lots, reintents i prompts."""

from __future__ import annotations

import configparser
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

ARREL = Path(__file__).resolve().parent.parent
ERRORS_TRANSITORIS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
OLLAMA_BASE_URL = "http://localhost:11434/v1"

PREUS = {
    "gpt-4.1-mini": {"in": 0.400, "cached_in": 0.100, "out": 1.600},
    "gpt-4o-mini": {"in": 0.150, "cached_in": 0.075, "out": 0.600},
    "gpt-4o": {"in": 2.500, "cached_in": 1.250, "out": 10.000},
}


def proveidor_per_model(model: str) -> str:
    """Decideix el proveidor independentment de si en coneixem el preu."""
    families_openai = ("gpt-", "chatgpt-", "o1", "o3", "o4")
    es_nom_openai = model.startswith(families_openai) and not model.startswith("gpt-oss")
    if ":" not in model and es_nom_openai:
        return "openai"
    return "ollama"


def arrel_projecte(inici: Path | None = None) -> Path:
    """Arrel del projecte buscant cap amunt."""
    inici = inici or Path.cwd()
    return next(p for p in (inici, *inici.parents) if (p / "src/llm.py").exists())


def clau_openai() -> str | None:
    """Clau d'OpenAI de l'entorn o del config.ini."""
    if key := os.environ.get("OPENAI_API_KEY"):
        return key
    cfg = configparser.ConfigParser()
    cfg.read(ARREL / "others/config.ini")
    return cfg.get("OPENAI", "KEY", fallback=None)


def client_openai() -> OpenAI:
    """Client d'OpenAI autenticat."""
    if not (key := clau_openai()):
        raise RuntimeError("Cap OPENAI_API_KEY a l'entorn ni a others/config.ini")
    return OpenAI(api_key=key)


def client_per_model(model: str) -> tuple[OpenAI, str]:
    """Client segons el model."""
    if proveidor_per_model(model) == "openai":
        return client_openai(), "openai"
    return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=900.0), "ollama"


def cost(meta: dict, model: str) -> float:
    """Cost en dolars d'una resposta segons el model."""
    if not (p := PREUS.get(model)):
        return 0.0
    no_cachejats = max(0, meta["prompt_tokens"] - meta["cached_tokens"])
    return (no_cachejats * p["in"] + meta["cached_tokens"] * p["cached_in"]
            + meta["completion_tokens"] * p["out"]) / 1_000_000


def crida(client: OpenAI, model: str, system: str, user: str, schema: dict,
          schema_name: str, max_tokens: int | None = None) -> tuple[dict, dict]:
    """Una crida amb esquema forçat."""
    t0 = time.time()
    resposta = client.chat.completions.create(
        model=model, temperature=0, seed=42,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        **({"max_tokens": max_tokens} if max_tokens else {}),
    )
    msg = resposta.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"El model ha refusat la petició: {msg.refusal}")

    usage = resposta.usage
    detalls = getattr(usage, "prompt_tokens_details", None)
    meta = {"duration_s": round(time.time() - t0, 2),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": getattr(detalls, "cached_tokens", 0) or 0}
    try:
        return json.loads(msg.content), meta
    except json.JSONDecodeError as exc:
        raise ValueError(f"El model ha retornat un JSON invàlid: {msg.content[:200]!r}") from exc


def crida_amb_reintents(client: OpenAI, model: str, *args, max_reintents: int = 4,
                        **kwargs) -> tuple[dict, dict]:
    """Backoff exponencial amb jitter davant errors transitoris i JSON invàlid."""
    for intent in range(1, max_reintents + 1):
        try:
            return crida(client, model, *args, **kwargs)
        except (*ERRORS_TRANSITORIS, ValueError) as e:
            if intent == max_reintents:
                raise
            espera = 2 ** (intent - 1) + random.uniform(0, 1)
            print(f"  Avís: {type(e).__name__} (intent {intent}/{max_reintents}). "
                  f"Reintent en {espera:.1f}s...")
            time.sleep(espera)
    raise AssertionError("inabastable")


def per_lots(seq: Sequence, mida: int) -> Iterator[list]:
    """Trosseja una sequencia en lots."""
    for i in range(0, len(seq), mida):
        yield seq[i:i + mida]


def dedupe(elements: Iterable) -> list:
    """Elimina duplicats conservant l'ordre."""
    return list(dict.fromkeys(elements))


def validar_ids(items: list[dict], n_esperats: int) -> None:
    """El model ha de tornar un registre per entrada, en el mateix ordre."""
    ids = [item["id"] for item in items]
    if ids != list(range(n_esperats)):
        raise ValueError(f"Registres desalineats: esperava 0..{n_esperats - 1}, he rebut {ids!r}")


def processar_per_lots(items: Sequence, mida_lot: int, client: OpenAI, model: str,
                       system: str, schema: dict, schema_name: str, clau_resultat: str,
                       camp: str, max_tokens: int | None = None,
                       on_lot: Callable[[dict], None] | None = None,
                       etiqueta: str = "Lot") -> tuple[dict, dict]:
    """Envia `items` (llista de strings) per lots i retorna {item: valor}."""
    unics = dedupe(items)
    lots = list(per_lots(unics, mida_lot))
    resultats: dict[str, str] = {}
    total = {"duration_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
             "cached_tokens": 0, "cost_usd": 0.0}

    def processar(lot: list, prefix: str) -> None:
        """Processa un lot i, si el model no en surt, el parteix per la meitat."""
        payload = json.dumps([{"id": j, "entitat": e} for j, e in enumerate(lot)],
                             ensure_ascii=False)
        try:
            obj, meta = crida_amb_reintents(client, model, system,
                                            f"Entitats a processar:\n{payload}",
                                            schema, schema_name, max_tokens=max_tokens)
            registres = obj[clau_resultat]
            validar_ids(registres, len(lot))
        except ValueError:
            if len(lot) == 1:
                raise
            mig = len(lot) // 2
            print(f"  {prefix}: el model no ha respost amb {len(lot)} entitats; "
                  f"es parteix en {mig} i {len(lot) - mig}")
            processar(lot[:mig], prefix + "a")
            processar(lot[mig:], prefix + "b")
            return

        del_lot = {lot[r["id"]]: r[camp] for r in registres}
        resultats.update(del_lot)
        if on_lot:
            on_lot(del_lot)
        for k in ("duration_s", "prompt_tokens", "completion_tokens", "cached_tokens"):
            total[k] += meta[k]
        total["cost_usd"] += cost(meta, model)

    for i, lot in enumerate(lots, start=1):
        print(f"{etiqueta} {i}/{len(lots)} ({len(lot)} entitats, {model})...")
        processar(lot, f"{etiqueta} {i}")

    total["cost_usd"] = round(total["cost_usd"], 6)
    return resultats, total


def json_compacte(obj, nivell: int = 0, sagnia: int = 2) -> str:
    """Com `json.dumps(indent=2)` pero amb cada registre fulla en una sola linia."""
    pad, pad2 = " " * (nivell * sagnia), " " * ((nivell + 1) * sagnia)
    if isinstance(obj, dict) and obj:
        fulla = all(not isinstance(v, (dict, list)) or
                    (isinstance(v, list) and not any(isinstance(x, dict) for x in v))
                    for v in obj.values())
        if fulla:
            return json.dumps(obj, ensure_ascii=False)
        cos = ",\n".join(f"{pad2}{json.dumps(k, ensure_ascii=False)}: "
                          f"{json_compacte(v, nivell + 1, sagnia)}" for k, v in obj.items())
        return "{\n" + cos + f"\n{pad}}}"
    if isinstance(obj, list) and obj and any(isinstance(x, (dict, list)) for x in obj):
        cos = ",\n".join(f"{pad2}{json_compacte(x, nivell + 1, sagnia)}" for x in obj)
        return "[\n" + cos + f"\n{pad}]"
    return json.dumps(obj, ensure_ascii=False)


SCHEMA_FONETICA = {
    "type": "object",
    "properties": {
        "diccionari": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Identificador numèric rebut."},
                    "transcripcio_fonetica": {"type": "string",
                                              "description": "La reescriptura fonètica."},
                },
                "required": ["id", "transcripcio_fonetica"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["diccionari"],
    "additionalProperties": False,
}

SCHEMA_DECISIO = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "decisio": {"type": "string", "enum": ["no", "si", "dubte"]},
                },
                "required": ["id", "decisio"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}

NOTES_IDIOMA = {
    "Castellano": (
        "Usa tildes según las reglas de acentuación del castellano cuando la pronunciación real "
        "no coincida con la acentuación por defecto (p.ej. una palabra extranjera que termina en "
        "consonante distinta de 'n'/'s' pero se pronuncia aguda)."
    ),
    "Catalán": (
        "Usa las convenciones ortográficas y de acentuación propias del catalán (accents oberts/"
        "tancats, dígrafos catalanes). No apliques reglas de acentuación castellanas."
    ),
    "Euskera": (
        "El euskera batua no usa tildes en su ortografía nativa: NO añadas acentos gráficos. "
        "Usa los dígrafos propios del euskera (tx, tz, ts, x, j, k...) en vez de convenciones "
        "ortográficas castellanas para representar sonidos extranjeros."
    ),
}

EXEMPLES = {
    "Castellano": [
        ("Wall Street", "Guol estrit"),
        ("Lamine Yamal", "Lamín Yamal"),
        ("Mbappé", "Embapé"),
        ("Junior", "Yúnior"),
        ("hamas", "Hamás"),
        ("FC Barcelona", "F C Barcelona"),
        ("EH Bildu", "E H Bildu"),
        ("Spotify Camp Nou", "Espótifai Camp Nou"),
        ("Nico Harrison", "Nico Járrison"),
        ("Mohamed Shia al-Sudani", "Mohámed Shía al Sudáni"),
        ("Erling Haaland", "Erling Hóland"),
    ],
}


def _bloc_exemples(idioma: str) -> str:
    """Bloc d'exemples del prompt per a un idioma."""
    exemples = EXEMPLES.get(idioma)
    if not exemples:
        return (
            f"No se incluyen ejemplos verificados para {idioma} todavía: aplica los mismos "
            f"principios (respelling literal, vocal de apoyo en grupos consonánticos imposibles, "
            f"expansión de símbolos) usando exclusivamente la ortografía y fonética propias de "
            f"{idioma}, sin recurrir a convenciones castellanas."
        )
    linies = "\n".join(f'- "{orig}" -> "{fon}"' for orig, fon in exemples)
    return f"Ejemplos orientativos (ajusta si difieren de lo que oigas realmente en el TTS):\n{linies}"


def regles_respelling(idioma: str = "Castellano") -> str:
    """Les regles de reescriptura fonetica, sense el marc de cap tasca concreta."""
    nota = NOTES_IDIOMA.get(idioma, f"Usa las convenciones ortográficas y de acentuación propias de {idioma}.")
    return f"""Reglas de reescritura fonética:
1. Siglas deletreadas (ej. UE, FMI): separa cada letra con un espacio y mantenlas en mayúscula. (Resultado: "U E", "F M I").
2. Acrónimos léxicos (ej. PSOE, OTAN): capitaliza solo la primera letra y ajusta la acentuación si la pronunciación real no coincide con la acentuación por defecto de {idioma}. (Resultado: "Psoe", "Otán").
3. Extranjerismos y nombres propios complejos: aplica respelling fonético usando la ortografía literal de {idioma}, de forma que alguien sin conocimiento del idioma de origen se acerque a la pronunciación real. Revisa TODOS los tokens de la entidad: marcas inglesas, apellidos, sh/th/w y vocales inglesas suelen necesitar cambio aunque el resto de la frase sea fácil. Si la palabra empieza por un grupo consonántico imposible en {idioma} (ej. "Mb", "Ng", "Pf" o "Sp"), añade una vocal de apoyo o simplifica el grupo para que sea pronunciable.
4. Símbolos y unidades (ej. ºC, %, km/h): expándelos a su forma leída completa en palabras de {idioma}. (Resultado en castellano: "grados Celsius", "por ciento", "kilómetros por hora").

Cuidado con la letra 'j': en {idioma} suena como una jota dura. Úsala solo para una 'h'
inglesa realmente aspirada ("Harrison" -> "Járrison"). Para una 'h' muda o suave de otras
lenguas, elimínala o usa una vocal, nunca 'j' ("Haaland" -> "Hóland", no "Jaland").

Particularidades de {idioma}: {nota}

{_bloc_exemples(idioma)}"""


def system_fonetica(idioma: str = "Castellano") -> str:
    """Prompt de sistema per demanar la fonetica d'entitats."""
    return f"""Eres un lingüista experto en fonética y sistemas Text-to-Speech (TTS).
Vas a recibir una lista de entidades (nombres propios, acrónimos, extranjerismos, símbolos).
Tu objetivo es devolver su adaptación fonética para que un modelo TTS configurado en {idioma}
lo lea correctamente con naturalidad, como un presentador de noticias.

Reglas de integridad (obligatorias):
- Debes devolver exactamente una entrada por cada entidad de entrada, ni más ni menos.
- Conserva el orden y el id numérico de cada entrada.
- No devuelvas ni corrijas el nombre original; devuelve únicamente id y transcripción.
- Nunca traduzcas, abrevies, inventes, omitas ni dividas una entidad en varias.
- La entidad ya está canonizada: NO corrijas, completes ni sustituyas su identidad.
- Si ya se pronunciaría correctamente, devuélvela sin modificar.
- No cambies solo mayúsculas/minúsculas: no alteran la pronunciación. Excepción: convertir una sigla TODA EN MAYÚSCULAS en acrónimo léxico (PSOE -> Psoe).
- Devuelve ortografía normal de {idioma}, NUNCA alfabeto fonético internacional (IPA) ni separación en sílabas con guiones.

{regles_respelling(idioma)}
"""


def system_decisio(idioma: str = "Castellano") -> str:
    """Prompt de sistema per decidir si una entitat es reescriu."""
    return f"""Clasifica entidades para un TTS configurado en {idioma}.
La pregunta es si hay que reescribir la grafía para que el TTS pronuncie correctamente la entidad.
Devuelve 'no' SOLO si todos los tokens producirán la pronunciación natural correcta.
Devuelve 'si' para siglas, símbolos, extranjerismos o nombres cuya lectura necesita adaptación.
Devuelve 'dubte' si no puedes asegurarlo; los casos dudosos pasarán al modelo experto.
Devuelve exactamente un resultado por entrada, con el mismo id numérico y en el mismo orden.
No copies ni corrijas la entidad en la salida. Prioriza no perder adaptaciones necesarias.
Presta especial atención a marcas y nombres con sp- inicial, sh, th, w, vocales inglesas, consonantes dobles o grafías eslavas.
Ejemplos: Madrid=no, Pedro Sánchez=no, Mbappé=si, UE=si, PSOE=si, Wall Street=si, Spotify Camp Nou=si, Nico Harrison=si, Mohamed Shia al-Sudani=si."""


SCHEMA_VALIDACIO = {
    "type": "object",
    "properties": {
        "entidades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "grafia_correcta": {"type": "string"},
                    "tipo": {"type": "string",
                             "enum": ["PERSONA", "LUGAR", "ORGANIZACION", "EVENTO", "OTRO", "NO_ENTIDAD"]},
                    "entidad_conocida": {"type": "boolean"},
                    "contexto": {"type": "string"},
                    "motivo": {"type": "string"},
                },
                "required": ["id", "grafia_correcta", "tipo", "entidad_conocida",
                             "contexto", "motivo"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entidades"],
    "additionalProperties": False,
}


_CAMPS_VALIDACIO = """Para cada entrada devuelve:
- id: el mismo identificador numérico que has recibido, en el mismo orden.
- grafia_correcta: la grafía canónica y correctamente acentuada. Usa el contexto para desambiguar.
- tipo: PERSONA, LUGAR, ORGANIZACION, EVENTO, OTRO, o NO_ENTIDAD si no es un nombre propio.
- entidad_conocida: true si reconoces la entidad y confías en la `grafia_correcta` que devuelves; false si no la reconoces con seguridad.
- contexto: una frase que diga QUIÉN o QUÉ es la entidad y en qué ámbito aparece, escrita para que un guionista de informativos pueda redactar noticias verosímiles sobre ella. NO menciones NUNCA la ortografía, la grafía, las tildes, ni si está bien o mal escrita: eso va en `motivo`. Tampoco te refieras a otras entradas del lote ("misma persona que en id 20"). Si no la reconoces, dilo aquí explícitamente.
- motivo: una frase breve justificando la decisión sobre la GRAFÍA y el reconocimiento: por qué esa grafía es la correcta y por qué reconoces (o no) la entidad."""

_REGLA_NO_INVENTAR = """REGLA CRÍTICA: si no reconoces la entidad con seguridad (topónimos menores, periodistas locales, nombres poco conocidos), NO inventes una grafía. Devuelve grafia_correcta igual a la entrada recibida y entidad_conocida=false. Es preferible mandarla a revisión humana que corromper el dataset."""

_REGLA_NO_SUSTITUIR = """REGLA CRÍTICA 2: nunca sustituyas la entidad por OTRA distinta que se le parezca. Si el ground truth dice "Japoel Tel Aviv", la corrección es "Hapoel Tel Aviv" (arreglar la grafía), nunca "Maccabi Tel Aviv" (otro club). Ante la duda, entidad_conocida=false."""

_CAPCALERA = "Eres un experto en entidades nombradas del ámbito informativo español (RNE)."

SYSTEM_VALIDACIO_A = f"""{_CAPCALERA}

Recibes entidades extraídas automáticamente de una transcripción de referencia (ground truth) hecha por humanos, junto con cómo las transcribió mal el modelo Whisper y la frase de contexto. Cada entrada lleva un `id` numérico.

IMPORTANTE: el ground truth TAMBIÉN contiene erratas. En algunos casos la grafía correcta es la que produjo Whisper, no la del ground truth (ej.: ground truth "mouseti" pero el tenista se escribe "Musetti"; "vasconia" pero el equipo es "Baskonia"; "letour" pero el pueblo de Albacete es "Letur").

Un nombre común (aunque venga capitalizado) es NO_ENTIDAD: "urgencias", "metrología", "luciérnagas", "paradas" no son entidades.

{_CAMPS_VALIDACIO}

{_REGLA_NO_INVENTAR}

{_REGLA_NO_SUSTITUIR}"""

SYSTEM_VALIDACIO_B = f"""{_CAPCALERA}

Recibes spans detectados automáticamente por un NER (spaCy `es_core_news_md`) sobre un corpus de noticias en bruto. Cada entrada lleva un `id` numérico. A diferencia de un ground truth humano, aquí NO hay ninguna garantía de que el span sea una entidad real: el NER genera bastante ruido (fragmentos que arrastran la palabra siguiente, sintagmas comunes, verbos/pronombres/adverbios colados en el span, palabras sueltas mal etiquetadas por ir en mayúscula al principio de frase). `label_ner` es la etiqueta que le puso spaCy (PER/ORG/LOC), pero puede estar equivocada.

Si el span no es una entidad real (fragmento de frase, sintagma común, verbo/pronombre suelto, span mal cortado), devuelve la entrada tal cual con tipo=NO_ENTIDAD.

{_CAMPS_VALIDACIO}

{_REGLA_NO_INVENTAR}

{_REGLA_NO_SUSTITUIR}"""


SCHEMA_CONTEXT = {
    "type": "object",
    "properties": {
        "entidades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Identificador numèric rebut."},
                    "contexto": {"type": "string",
                                 "description": "Tipo de entidad y una frase de contexto."},
                },
                "required": ["id", "contexto"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entidades"],
    "additionalProperties": False,
}

SYSTEM_CONTEXT = """Eres un documentalista de informativos de televisión.

Para cada entidad que recibas devuelve UNA sola línea de contexto que le sirva a un
guionista para escribir noticias verosímiles sobre ella: empieza por el tipo (Persona,
Organización, Lugar, Evento, Marca, Tecnología...) y sigue con una frase breve que diga
qué o quién es y en qué ámbito aparece.

Ejemplo: "Persona. Futbolista del FC Barcelona y de la selección española."

Devuelve exactamente un resultado por entrada, con el mismo id numérico y en el mismo
orden. Si no reconoces la entidad, dilo en el propio contexto en vez de inventarte datos."""


SCHEMA_FRASES = {
    "type": "object",
    "properties": {
        "frases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "texto": {"type": "string",
                              "description": "Frase con ortografía real. Es el ground truth."},
                },
                "required": ["texto"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["frases"],
    "additionalProperties": False,
}


def system_frases(idioma: str = "Castellano") -> str:
    """Prompt de generacio de frases."""
    return f"""Eres un guionista de informativos de RTVE (Telediario) que escribe en {idioma}.

Entrega el campo "texto": la frase con ortografía real y correcta. Es la transcripción de
referencia (ground truth) del dataset, así que los nombres propios deben estar bien
escritos y acentuados.

Reglas de las frases:
- Genera exactamente el número de frases que se te pida, independientes entre sí.
- La entidad objetivo debe aparecer en todas, escrita EXACTAMENTE como se te da. Respeta
  la concordancia gramatical del resto de la frase, pero NUNCA sustituyas la entidad por
  otra palabra, ni la abrevies, ni la traduzcas.
- Escribe ortografía normal: nada de alfabeto fonético ni de separación en sílabas.
- Evita las cifras cuando puedas escribir el número en palabras; si pones una, que sea
  un año o una cantidad normal.
"""


SCHEMA_ESCENARIS = {
    "type": "object",
    "properties": {
        "escenaris": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Situacions o contextos, no les intervencions en si.",
        }
    },
    "required": ["escenaris"],
    "additionalProperties": False,
}


def system_escenaris(idioma: str = "Castellano", domini: str = "informativos de televisión y radio") -> str:
    """Prompt de sistema per generar escenaris."""
    return f"""Eres un experto generando escenarios para ampliar datos de entrenamiento de un
sistema de reconocimiento de voz (ASR) en {idioma}, dentro del ámbito de {domini}.

Genera situaciones o contextos DISTINTOS en los que alguien podría hablar dentro de este
ámbito. No generes las intervenciones en sí, solo la situación o el contexto en el que
tendrían lugar. Ejemplo válido: "Un corresponsal informa en directo desde el lugar de un
incendio". Ejemplo NO válido: "El fuego ha arrasado ya diez hectáreas" -- eso ya es una
intervención, no un escenario.

Sé lo más diverso posible: varía el tema (política, deportes, cultura, sucesos, economía,
meteorología, sociedad, ciencia...), el tipo de intervención (titular, crónica, entrevista,
declaración, entradilla, noticia breve...) y el registro.

Cada escenario debe tener entre 5 y 200 caracteres. No repitas escenarios ni añadas
anotaciones, numeración ni explicaciones."""


SCHEMA_FRASES_ESCENARI = {
    "type": "object",
    "properties": {
        "frases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Frases generadas para el escenario dado.",
        }
    },
    "required": ["frases"],
    "additionalProperties": False,
}


def system_frases_escenari(idioma: str = "Castellano", domini: str = "informativos de televisión y radio") -> str:
    """Prompt de sistema per generar frases d'un escenari."""
    return f"""Eres un guionista de {domini} escribiendo en {idioma}, generando datos
sintéticos para entrenar un sistema de reconocimiento de voz (ASR).

Para el escenario que se te indique, genera frases independientes, verosímiles y
gramaticalmente correctas, como las diría un locutor o periodista real. Usa nombres
propios, lugares, cifras y términos tan DIVERSOS como sea posible entre las frases: no
repitas la misma estructura sintáctica ni el mismo vocabulario de una frase a otra.

Cada frase debe tener entre 5 y 200 caracteres. No uses alfabeto fonético internacional
(IPA) ni separación en sílabas con guiones. No repitas frases ni añadas anotaciones,
numeración ni explicaciones."""

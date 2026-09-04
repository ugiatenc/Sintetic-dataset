#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae las entidades (nombres propios) del Ground Truth RNE que el modelo
Whisper large-v3 (default) transcribe mal, comparando cada transcripción
contra su ground truth mediante alineamiento de palabras (jiwer).

Para cada entidad fallida se cuenta cuántas veces aparece mal transcrita:
  - Si el modelo la sustituye por otra palabra -> se registra la palabra errónea.
  - Si el modelo la omite por completo -> se registra como "[OMITIDA]".

Salida: JSON con la lista de entidades erróneas, agregada sobre los 4 audios.

Uso:
  python3 evaluate.py
  python3 evaluate.py --output entidades_erroneas.json
"""
import argparse
import difflib
import json
import re
from pathlib import Path
from collections import Counter

import jiwer
from rapidfuzz.distance import JaroWinkler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS_DIR = ROOT / "lab/inputs/RNE"
SALIDA_POR_DEFECTO = ROOT / "lab/entitats/entitats_fallades/entidades_erroneas.json"
MODEL_STEM = "large-v3"
OMITIDA = "[OMITIDA]"
MAX_REGION = 6  # tokens del REF en una región de desalineamiento; por encima es
                # desincronización grande (careta, música), no un fallo de entidad puntual

PAIRS = [
    ("24_HORAS",       "r1_24_horas",                    INPUTS_DIR / "groundtruths/ground_truth_R1_24HORAS.json"),
    ("MEDIODIA",       "R1_MEDIODIA_EN_RNE_2025-10-09",  INPUTS_DIR / "groundtruths/ground_truthMEDIODIA.txt"),
    ("NEUDC",          "R1_NEUDC_2025-10-12",            INPUTS_DIR / "groundtruths/ground_truthNEUDC.json"),
    ("EL_ULTIMO_TREN", "r1_el_ultimo_tren",              INPUTS_DIR / "groundtruths/ground_truth_R1_EL-ULTIMO-TREN.json"),
]

# ============================================================
# NORMALIZACIÓN Y CARGA
# ============================================================
ACC = str.maketrans("áéíóúüàèìòù", "aeiouuaeiou")
STOP = set("""el la los las un una y pero que como cuando donde porque para por con sin sobre entre
desde hasta este esta estos estas ese esa esos esas su sus mi mis tu tus no si ya aqui alli ahi
ahora antes despues bueno pues claro hola adios gracias buenas buenos hoy manana ayer nos me te se
le les lo al del es son era fue hay muy mas menos tambien tanto todo toda todos todas nada algo cada
otro otra otros otras mucho mucha muchos muchas poco vamos voy va van eh ah oh vale mira oye bien
mal senor senora don dona asi quien cual tras han hemos he has habia a en de o u su lo aún más""".split())


def norm(text: str) -> str:
    """Minúsculas + limpieza de puntuación. Mantiene tildes y 'ñ' a propósito:
    así los fallos de acentuación (pais/país) también aparecen como sustituciones."""
    text = text.lower().strip()
    text = re.sub(r'\[.*?\]', '', text)
    for c in [",", ".", "!", "¡", "?", "¿", ";", ":", '"', "'", "«", "»", "…", "—", "-", "(", ")"]:
        text = text.replace(c, " ")
    return " ".join(text.split())


def strip_accents(word: str) -> str:
    return word.translate(ACC)


def fonetiza(palabra: str) -> str:
    """Aproximación barata de grafía -> fonema del castellano: colapsa dígrafos y
    letras que representan el mismo sonido para poder comparar sustituciones que
    Whisper escribe distinto pero suenan igual ('krasznahorkai' / 'krasná jorkaj',
    'climent' / 'kilming')."""
    s = strip_accents(palabra.lower())
    s = re.sub(r"[^a-zñ ]", "", s)
    s = s.replace("ch", "C").replace("ll", "Y").replace("qu", "k").replace("gu", "g")
    s = re.sub(r"c([ei])", r"s\1", s)
    s = s.replace("c", "k").replace("z", "s").replace("v", "b")
    s = re.sub(r"g([ei])", r"x\1", s)
    s = s.replace("j", "x").replace("h", "").replace("ñ", "N").replace("w", "b")
    s = s.replace("y", "i").replace("Y", "y")
    return re.sub(r"(.)\1+", r"\1", s)  # dobles


def similitud(correcta: str, variante: str) -> float:
    """Similitud entidad <-> transcripción errónea, robusta a dos cosas que
    Jaro-Winkler crudo no soporta: que Whisper fragmente/una la palabra distinto que
    el GT ('radiogaceta' -> 'radio gaceta': se comparan sin espacios) y que la
    sustitución sea fonética con grafía lejana ('viñas' -> 'víñez'): se toma el
    máximo entre la similitud gráfica y la fonética."""
    a, b = correcta.replace(" ", ""), variante.replace(" ", "")
    if not a or not b:
        return 0.0
    return max(JaroWinkler.similarity(a, b), JaroWinkler.similarity(fonetiza(a), fonetiza(b)))


def similitud_grafica(a: str, b: str) -> float:
    """Similitud SOLO de grafía (Jaro-Winkler), sin el máximo fonético de `similitud()`.

    `similitud()` existe para el problema contrario: decidir si Whisper ha ACERTADO una
    entidad aunque la escriba distinto ('viñas' -> 'víñez' suena igual). Aquí la pregunta
    es la opuesta -- si un LLM ha REESCRITO una entidad por otra que solo se le parece --,
    así que una coincidencia fonética con grafía distinta es precisamente la señal de
    alarma que hay que conservar, no camuflar con un máximo."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


def diagnosticar_cambio(entrada: str, correcta: str) -> dict:
    """Diagnóstico de qué le ha hecho el LLM a `entrada` para producir `correcta`.

    Sustituye a la antigua `similitud_del_cambio`, que solo miraba los tramos
    SUSTITUIDOS (algo por algo) y de paso llamaba a `similitud()`, la métrica del
    problema contrario (ver `similitud_grafica`). Se devuelven tres piezas porque cada
    una se verifica con una regla distinta y ninguna sirve para las otras dos:

      - sim_sustitucion: peor similitud GRÁFICA entre los tramos que se sustituyen de
        verdad. Sigue cazando 'Japoel Tel Aviv' -> 'Maccabi Tel Aviv' (0.44, dos clubes
        distintos) sin dejar pasar sustituciones fonéticamente parecidas pero mal
        escritas ('Svereb' -> 'Sverev' da 0.93 igualmente: eso no lo resuelve una
        métrica de grafía, hace falta una segunda opinión).
      - insertados: palabras que trae `correcta` y no tenía `entrada`, en un tramo donde
        NO hay nada sustituido (`entrada` no tenía ninguna palabra ahí). No hay nada con
        que compararlas por grafía -- la única forma de verificarlas es comprobar que
        el ground truth las respalda, y eso lo tiene que hacer quien conozca el GT.
      - eliminados: palabras que `entrada` tenía y `correcta` quita sin sustituirlas por
        nada. Puede ser limpieza legítima ('Pepa Millán Vox' -> 'Pepa Millán') o pérdida
        de contenido real; tampoco es verificable por grafía sola.

    Importante: un tramo `replace` (algo por algo, los dos lados no vacíos) cuenta SOLO
    para `sim_sustitucion`, nunca para `insertados`/`eliminados` -- 'vasconia' ->
    'Baskonia' es una sustitución de una palabra por otra, no borra 'vasconia' Y añade
    'Baskonia' a la vez. Tratar cada `replace` como inserción+eliminación disparaba el
    guardarraíl de "elimina contenido" en casi cualquier corrección de una sola palabra.
    """
    a, b = entrada.lower().split(), correcta.lower().split()
    sim = 1.0
    insertados, eliminados = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        trozo_a, trozo_b = a[i1:i2], b[j1:j2]
        if tag == "replace":
            sim = min(sim, similitud_grafica(" ".join(trozo_a), " ".join(trozo_b)))
        elif tag == "insert":
            insertados += trozo_b
        elif tag == "delete":
            eliminados += trozo_a
    return {"sim_sustitucion": round(sim, 3), "insertados": insertados, "eliminados": eliminados}


def load_raw(p: Path) -> str:
    if p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8")
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, dict) and "text" in d:
        return d["text"]
    if isinstance(d, list):
        return " ".join(s["text"] for s in d if s.get("text"))
    raise ValueError(f"Formato no reconocido: {p}")


def entity_set(raw: str):
    """Nombres propios del ground truth: palabras capitalizadas que no abren frase."""
    ent = set()
    for sent in re.split(r'(?<=[.!?])\s+', raw):
        words = sent.split()
        for i, w in enumerate(words):
            if i == 0:
                continue
            tok = w.strip(',.;:¿?¡!()"\'«»…—-')
            if re.match(r'^[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,}', tok):
                n = norm(tok)
                if n and strip_accents(n) not in STOP:
                    ent.add(n)
    return ent


# ============================================================
# ALINEAMIENTO
# ============================================================
def align(ref: str, hyp: str):
    """Agrupa el alineamiento por REGIÓN, no por token. jiwer da chunks 'substitute'/
    'delete' de igual longitud a ambos lados (N ref <-> N hyp) y alinearlos posición
    a posición parte una palabra que Whisper fragmenta ('radiogaceta' -> 'radio
    gaceta') dejando la entidad emparejada solo con el último fragmento ('gaceta',
    similitud 0 con la entidad completa). Aquí se agrupa toda racha de chunks no-
    'equal' consecutivos y se devuelve el span REF completo junto al span HYP
    completo, para comparar 'radiogaceta' contra 'radio gaceta' en vez de 'gaceta'."""
    measures = jiwer.process_words(ref, hyp)
    rw, hw = ref.split(), hyp.split()
    chunks = measures.alignments[0]
    regiones, i = [], 0
    while i < len(chunks):
        if chunks[i].type == "equal":
            i += 1
            continue
        j = i
        while j < len(chunks) and chunks[j].type != "equal":
            j += 1
        r0, r1 = chunks[i].ref_start_idx, chunks[j - 1].ref_end_idx
        h0, h1 = chunks[i].hyp_start_idx, chunks[j - 1].hyp_end_idx
        regiones.append((rw[r0:r1], hw[h0:h1]))
        i = j
    return regiones


def _reparto_posicional(hyp_tokens: list, n: int) -> list:
    """Divide `hyp_tokens` en `n` bloques contiguos lo más parejos posible,
    preservando el orden. Sirve para cuando una región mezcla varias entidades
    ('camp nou' -> 'can know'): el span entero diluye la señal de cada una
    ('camp' vs 'can know' = 0.59), pero la entidad de turno suele corresponder
    a su bloque por posición ('camp' vs 'can' = 0.78)."""
    base, extra = divmod(len(hyp_tokens), n)
    grupos, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        grupos.append(hyp_tokens[i:i + size])
        i += size
    return grupos


def find_hypothesis(stem: str) -> Path:
    transcripciones = INPUTS_DIR / "transcriptions"
    for candidate in (transcripciones / f"{stem}_{MODEL_STEM}.json", transcripciones / f"{stem}_{MODEL_STEM}_words.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encuentra la transcripción del modelo {MODEL_STEM} para '{stem}'")


def to_json(resultado) -> str:
    """JSON legible: cada entidad en su bloque, pero 'mal_transcrita' en una sola línea."""
    lines = ["["]
    for i, e in enumerate(resultado):
        comma = "," if i < len(resultado) - 1 else ""
        variantes = ", ".join(json.dumps(v, ensure_ascii=False) for v in e["mal_transcrita"])
        lines.append("  {")
        lines.append(f'    "bien_escrita": {json.dumps(e["bien_escrita"], ensure_ascii=False)},')
        lines.append(f'    "mal_transcrita": [{variantes}]')
        lines.append(f"  }}{comma}")
    lines.append("]")
    return "\n".join(lines)


# ============================================================
# LÓGICA PRINCIPAL
# ============================================================
def run(output: Path):
    errores = Counter()

    for name, stem, gt_path in PAIRS:
        if not gt_path.exists():
            print(f"[!] Falta ground truth: {gt_path}")
            continue
        try:
            hyp_path = find_hypothesis(stem)
        except FileNotFoundError as e:
            print(f"[!] {e}")
            continue

        raw = load_raw(gt_path)
        ref = norm(raw)
        hyp = norm(load_raw(hyp_path))
        ent = entity_set(raw)

        n_fallos = 0
        for ref_tokens, hyp_tokens in align(ref, hyp):
            if len(ref_tokens) > MAX_REGION:
                continue  # desincronización grande, no un fallo de entidad puntual
            afectadas_idx = [i for i, t in enumerate(ref_tokens) if t in ent]
            if not afectadas_idx:
                continue
            amplia = " ".join(hyp_tokens)
            posicional = _reparto_posicional(hyp_tokens, len(ref_tokens))
            for i in afectadas_idx:
                candidatos = [c for c in (amplia, " ".join(posicional[i])) if c]
                variante = max(candidatos, key=lambda c: similitud(ref_tokens[i], c)) if candidatos else OMITIDA
                errores[(ref_tokens[i], variante)] += 1
            n_fallos += len(afectadas_idx)

        print(f"[{name}] {len(ent)} entidades en GT, {n_fallos} fallos de entidad")

    por_palabra = {}
    for (r, h), n in errores.items():
        por_palabra.setdefault(r, []).append([h, n])

    resultado = [
        {"bien_escrita": r, "mal_transcrita": sorted(variantes, key=lambda v: -v[1])}
        for r, variantes in por_palabra.items()
    ]
    resultado.sort(key=lambda e: -sum(n for _, n in e["mal_transcrita"]))

    output.write_text(to_json(resultado), encoding="utf-8")
    print(f"\n>>> {len(resultado)} entidades erróneas distintas ({sum(errores.values())} fallos totales)")
    print(f">>> Guardado en: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Por defecto escribe donde lo LEE `create_entities_list.ipynb`. Antes el default
    # era `src/entidades_erroneas.json`, así que una ejecución limpia dejaba el fichero
    # en un sitio y el notebook seguía leyendo la copia vieja de la otra carpeta.
    parser.add_argument("--output", type=Path, default=SALIDA_POR_DEFECTO,
                        help=f"Fichero JSON de salida (default: {SALIDA_POR_DEFECTO})")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output)

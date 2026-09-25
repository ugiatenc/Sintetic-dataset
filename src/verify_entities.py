#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verificador round-trip: TTS -> entorn acustic -> Whisper -> mesura d'error per entitat."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import phonetics

DEFAULT_INPUT = ROOT / "data/es/entitats/entidades_fuente_b_validadas.json"
DEFAULT_OUTPUT = ROOT / "data/es/entitats/entidades_verificades_roundtrip.json"
DEFAULT_REF_AUDIO = ROOT / "data/es/audio/referencia/audio_referencia.wav"
DEFAULT_DICCIONARI = phonetics.ruta_diccionari("castellano", "treball")

TTS_MODEL_ID = "k2-fsa/OmniVoice"
ASR_MODEL_ID = "openai/whisper-large-v3"
SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_ASR = 16000

FRASES_PORTADORES = [
    "Hoy hemos hablado de {E} en el informativo.",
    "La noticia sobre {E} ha ocupado gran parte del boletín.",
    "Todo gira en torno a {E} esta semana.",
    "Los datos disponibles sobre {E} son objeto de debate.",
]

ENTORNS_DEFECTE = ["carrer", "telefon"]

VEUS_DEFECTE = [
    {"id": "female_middle", "instruct": "female, middle-aged, moderate pitch"},
    {"id": "male_middle", "instruct": "male, middle-aged, moderate pitch"},
    {"id": "female_young", "instruct": "female, young adult, high pitch"},
]

ANCORES = ["Madrid", "Barcelona", "Valencia", "Europa", "Pedro Sánchez"]


def clau(text: str) -> str:
    """Clau de comparacio, compartida amb el diccionari (`phonetics.clau`)."""
    return phonetics.clau(text)


def variants_entitat(entitat: str) -> list[str]:
    """Formes en que una transcripcio CORRECTA pot escriure l'entitat."""
    base = clau(entitat)
    if not base:
        return []
    variants = {base}
    lletres = base.replace(" ", "")
    net = entitat.replace(".", "").replace(" ", "")

    if 2 <= len(net) <= 6 and net.isupper():
        variants.add(" ".join(lletres))
        variants.add(lletres)
    elif " " in base and len(lletres) <= 6:
        variants.add(lletres)

    return sorted(variants, key=len, reverse=True)


def sense_h(text: str) -> str:
    """La `h` castellana es muda: no hi ha cap diferencia acustica entre "Ormuz" i "Hormuz"."""
    return re.sub(r"h", "", text)


def entitat_a_la_transcripcio(entitat: str, transcripcio: str) -> bool:
    """Encert estricte: alguna forma valida d'escriure l'entitat apareix literal dins de la transcripcio."""
    t = clau(transcripcio)
    for variant in variants_entitat(entitat):
        for text, patro in ((t, variant), (sense_h(t), sense_h(variant))):
            if patro and re.search(rf"(?<!\w){re.escape(patro)}(?!\w)", text):
                return True
    return False


def _levenshtein(a: str, b: str) -> int:
    """Distancia d'edicio entre dues cadenes."""
    if len(a) < len(b):
        a, b = b, a
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        actual = [i]
        for j, cb in enumerate(b, start=1):
            actual.append(min(anterior[j] + 1, actual[j - 1] + 1, anterior[j - 1] + (ca != cb)))
        anterior = actual
    return anterior[-1]


def distancia_entitat(entitat: str, transcripcio: str) -> float:
    """CER entre l'entitat i el tros de transcripcio que mes se li assembla (0 = exacte)."""
    e = clau(entitat)
    mots = clau(transcripcio).split()
    if not e or not mots:
        return 1.0
    n = len(e.split())
    millor = 1.0
    for amplada in {max(1, n - 1), n, n + 1}:
        for i in range(len(mots) - amplada + 1):
            candidat = " ".join(mots[i:i + amplada])
            millor = min(millor, _levenshtein(e, candidat) / len(e))
    return round(min(millor, 1.0), 3)


def carregar_entitats(path: Path) -> list[str]:
    """Llista d'entitats a verificar."""
    dades = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(dades, dict) and ("validadas" in dades or "rechazadas" in dades):
        registres = list(dades.get("validadas", []))
        registres += [r for r in dades.get("rechazadas", []) if r.get("tipo") != "NO_ENTIDAD"]
    elif isinstance(dades, dict):
        registres = dades.get("resultats", [])
    else:
        registres = dades
    if not isinstance(registres, list):
        raise ValueError(f"{path} no conte una llista d'entitats reconeixible")

    entitats, forma_larga_de = [], {}
    for r in registres:
        if isinstance(r, str):
            entitats.append(r)
        elif isinstance(r, dict):
            entrada = r.get("grafia_correcta") or r.get("entrada") or r.get("entidad") or r.get("entitat")
            entitats.append(entrada)
            if entrada and r.get("forma_larga"):
                forma_larga_de[entrada] = r["forma_larga"]
    entitats = [e for e in dict.fromkeys(entitats) if e]

    presents = set(entitats)
    curtes = {e for e in entitats
              if forma_larga_de.get(e) not in (None, e) and forma_larga_de[e] in presents}
    if curtes:
        print(f"  {len(curtes)} formes curtes descartades per tenir la seva forma_larga a la mateixa llista: "
              f"{', '.join(sorted(curtes)[:5])}{'...' if len(curtes) > 5 else ''}")
    entitats = [e for e in entitats if e not in curtes]

    no_verificables = [(e, phonetics.verificable_round_trip(e)[1])
                       for e in entitats if not phonetics.verificable_round_trip(e)[0]]
    if no_verificables:
        print(f"  {len(no_verificables)} entitats no mesurables pel round-trip, excloses: "
              f"{', '.join(f'{e} ({m})' for e, m in no_verificables[:5])}"
              f"{'...' if len(no_verificables) > 5 else ''}")
        exclosos = {e for e, _ in no_verificables}
        entitats = [e for e in entitats if e not in exclosos]

    return entitats


def carregar_diccionari(path: Path) -> phonetics.Diccionari:
    """Diccionari fonetic {entitat_raw: forma_per_al_tts} de `dictionary.ipynb`."""
    return phonetics.carregar(path)


def triar_dispositiu(preferit: str = "auto") -> str:
    """Primera GPU que aquesta build de PyTorch pugui fer servir de debo."""
    import torch
    if preferit != "auto":
        return preferit
    if not torch.cuda.is_available():
        return "cpu"
    for i in range(torch.cuda.device_count()):
        try:
            (torch.zeros(8, device=f"cuda:{i}") + 1).sum().item()
            print(f"Dispositiu: cuda:{i} ({torch.cuda.get_device_name(i)})")
            return f"cuda:{i}"
        except Exception as e:
            print(f"  cuda:{i} ({torch.cuda.get_device_name(i)}) descartada: {type(e).__name__}")
    return "cpu"


def carregar_models(device: str, asr_model_id: str = ASR_MODEL_ID):
    """Importa i carrega TTS + ASR."""
    try:
        import torch
        from omnivoice import OmniVoice
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Falten dependencies (torch/omnivoice/transformers). Activa l'entorn del "
            "projecte: source /home/ugiat/.virtualenvs/sintetic/bin/activate"
        ) from exc

    print(f"Carregant {TTS_MODEL_ID}...")
    tts = OmniVoice.from_pretrained(TTS_MODEL_ID, device_map=device, dtype=torch.float16)
    print(f"Carregant {asr_model_id}...")
    asr = pipeline("automatic-speech-recognition", model=asr_model_id,
                   dtype=torch.float16, device=device)
    return tts, asr


def sintetitzar_lot(tts, textos: list[str], veu: dict, ref_audio: str) -> list:
    """Una sola crida de TTS per a tots els `textos`, del lot sencer o de multiples entitats alhora."""
    kwargs = {"text": textos, "language": "es"}
    if veu.get("instruct"):
        kwargs["instruct"] = veu["instruct"]
    else:
        kwargs["ref_audio"] = [veu.get("ref_audio") or ref_audio] * len(textos)
    return [a.astype("float32") for a in tts.generate(**kwargs)]


def entorn_amb_snr_fixa(entorn_id: str, snr_db: float | None):
    """L'entorn del pipeline pero amb la SNR clavada en comptes de sortejada."""
    import dataclasses

    from acoustic_sim import ENTORNS

    entorn = ENTORNS[entorn_id]
    if entorn.snr_db is None:
        return entorn
    valor = snr_db if snr_db is not None else (entorn.snr_db[0] + entorn.snr_db[1]) / 2
    return dataclasses.replace(entorn, snr_db=(valor, valor))


def degradar(waveform, entorn_id: str, llavor: int, snr_db: float | None = None):
    """Aplica un entorn acustic del pipeline principal (mateix codi que `generate_voices_environments.ipynb`) i."""
    import numpy as np
    from acoustic_sim import mesclar_amb_entorn

    rng = np.random.default_rng(llavor)
    entorn = entorn_amb_snr_fixa(entorn_id, snr_db)
    return mesclar_amb_entorn(waveform, SAMPLE_RATE_TTS, entorn, rng)


def transcriure_lot(asr, waveforms_24k: list, asr_batch_size: int) -> list[str]:
    """Una sola crida d'ASR per a tots els `waveforms_24k`."""
    import soxr

    entrades = [
        {"raw": soxr.resample(w, SAMPLE_RATE_TTS, SAMPLE_RATE_ASR), "sampling_rate": SAMPLE_RATE_ASR}
        for w in waveforms_24k
    ]
    sortides = asr(entrades, batch_size=asr_batch_size,
                   generate_kwargs={"language": "es", "task": "transcribe"})
    return [s["text"].strip() for s in sortides]


def verificar_lot(tts, asr, entitats: list[str], n_frases: int, entorns: list[str],
                  ref_audio: str, asr_batch_size: int, diccionari,
                  snr_db: float | None = None, veus: list[dict] | None = None) -> dict[str, dict]:
    """Verifica moltes entitats alhora: una crida de TTS i una d'ASR per a tot el lot."""
    import hashlib

    plantilles = FRASES_PORTADORES[:max(1, min(n_frases, len(FRASES_PORTADORES)))]

    veus = veus or VEUS_DEFECTE
    tasques, audios_nets = [], []
    for i_veu, veu in enumerate(veus):
        subtasques = [(ent, i, veu["id"], p.format(E=diccionari.get(ent, ent)))
                      for ent in entitats for i, p in enumerate(plantilles)]
        tasques += subtasques
        audios_nets += sintetitzar_lot(tts, [f for _, _, _, f in subtasques], veu, ref_audio)
        if i_veu < len(veus) - 1:
            alliberar_vram()

    condicions = ["net"] + list(entorns)
    items = []
    for (entitat, idx_plantilla, veu_id, frase), audio_net in zip(tasques, audios_nets):
        for condicio in condicions:
            if condicio == "net":
                senyal, meta = audio_net, {"entorn_id": "net"}
            else:
                llavor = int.from_bytes(
                    hashlib.sha256(f"{condicio}|{idx_plantilla}".encode()).digest()[:8], "big"
                )
                senyal, meta = degradar(audio_net, condicio, llavor, snr_db)
            items.append((entitat, frase, veu_id, condicio, senyal, meta))

    transcripcions = transcriure_lot(asr, [it[4] for it in items], asr_batch_size)

    per_entitat: dict[str, list[dict]] = {ent: [] for ent in entitats}
    condicions_meta: dict[str, dict] = {}
    for (entitat, frase, veu_id, condicio, _audio, meta), transcripcio in zip(items, transcripcions):
        encert = entitat_a_la_transcripcio(entitat, transcripcio)
        condicions_meta.setdefault(condicio, meta)
        per_entitat[entitat].append({
            "condicio": condicio, "veu": veu_id, "frase": frase,
            "transcripcio": transcripcio, "encert": encert,
            "distancia": 0.0 if encert else distancia_entitat(entitat, transcripcio),
        })

    resultats = {}
    for entitat, files in per_entitat.items():
        per_condicio = {}
        for condicio in condicions:
            d_cond = [f for f in files if f["condicio"] == condicio]
            per_condicio[condicio] = round(1 - sum(f["encert"] for f in d_cond) / len(d_cond), 3)
        n_fallats = sum(1 for f in files if not f["encert"])
        resultats[entitat] = {
            "entitat": entitat,
            "n_frases": len(files),
            "tasa_error": round(n_fallats / len(files), 3) if files else None,
            "distancia_mitjana": round(sum(f["distancia"] for f in files) / len(files), 3)
                                 if files else None,
            "tasa_error_per_condicio": per_condicio,
            "tasa_error_per_veu": {
                v: round(1 - sum(f["encert"] for f in fv) / len(fv), 3)
                for v in {f["veu"] for f in files}
                for fv in [[f for f in files if f["veu"] == v]]
            },
            "condicions_aplicades": {
                c: {k: v for k, v in m.items() if v is not None}
                for c, m in condicions_meta.items()
            },
            "exemples_fallats": [f for f in files if not f["encert"]],
        }
    return resultats


def alliberar_vram() -> None:
    """Buida la cache de PyTorch despres d'una OOM."""
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def processar_lot_resilient(tts, asr, lot: list[str], n_frases: int, entorns: list[str],
                            ref_audio: str, asr_batch_size: int, diccionari,
                            snr_db: float | None = None,
                            veus: list[dict] | None = None) -> dict[str, dict]:
    """Verifica un lot i, si peta per falta de memoria, el parteix per la meitat i reintenta cada mitat."""
    try:
        return verificar_lot(tts, asr, lot, n_frases, entorns, ref_audio,
                             asr_batch_size, diccionari, snr_db, veus)
    except Exception as e:
        es_oom = "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"
        alliberar_vram()

        if es_oom and len(lot) == 1:
            veus_actuals = veus or VEUS_DEFECTE
            if len(veus_actuals) > 1:
                veus_reduides = veus_actuals[:-1]
                print(f"  OOM amb 1 entitat i {len(veus_actuals)} veus -> reintent amb "
                      f"{len(veus_reduides)} veus ({', '.join(v['id'] for v in veus_reduides)})")
                resultat = processar_lot_resilient(tts, asr, lot, n_frases, entorns, ref_audio,
                                                   asr_batch_size, diccionari, snr_db, veus_reduides)
                for r in resultat.values():
                    if "error" not in r and "veus_reduides" not in r:
                        r["veus_reduides"] = [v["id"] for v in veus_reduides]
                return resultat

        if len(lot) == 1 or not es_oom:
            motiu = "OOM amb una sola entitat, fins i tot amb 1 veu" if es_oom else type(e).__name__
            print(f"  ERROR irrecuperable ({motiu}): {str(e)[:120]}")
            return {ent: {"entitat": ent, "error": str(e)} for ent in lot}

        mig = len(lot) // 2
        print(f"  OOM amb {len(lot)} entitats -> reintent en 2 lots de {mig} i {len(lot)-mig}")
        resultats: dict[str, dict] = {}
        for sublot in (lot[:mig], lot[mig:]):
            resultats.update(processar_lot_resilient(
                tts, asr, sublot, n_frases, entorns, ref_audio,
                max(1, asr_batch_size * len(sublot) // len(lot)), diccionari, snr_db, veus,
            ))
        return resultats


def calibracio(resultats: dict[str, dict]) -> dict:
    """Resum de com han anat les entitats ancora: la linia base de la condicio."""
    files = {a: resultats[a] for a in ANCORES
             if a in resultats and resultats[a].get("tasa_error") is not None}
    if not files:
        return {}
    vals = [r["tasa_error"] for r in files.values()]
    per_condicio: dict[str, float] = {}
    for r in files.values():
        for cond, v in r.get("tasa_error_per_condicio", {}).items():
            per_condicio.setdefault(cond, []).append(v)
    return {
        "n": len(files),
        "tasa_error_mitjana": round(sum(vals) / len(vals), 3),
        "per_entitat": {a: r["tasa_error"] for a, r in files.items()},
        "per_condicio": {c: round(sum(v) / len(v), 3) for c, v in per_condicio.items()},
    }


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"JSON amb les entitats a verificar (per defecte: {DEFAULT_INPUT.name})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diccionari", type=Path, default=DEFAULT_DICCIONARI,
                        help="diccionari fonetic de dictionary.ipynb (entitat -> forma pel TTS)")
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--n-frases", type=int, default=len(FRASES_PORTADORES),
                        help=f"frases portadores per entitat (max {len(FRASES_PORTADORES)})")
    parser.add_argument("--entorns", nargs="*", default=ENTORNS_DEFECTE,
                        help="entorns degradats a provar, a mes de 'net' (veure ENTORNS a acoustic_sim.py)")
    parser.add_argument("--veus", type=int, default=len(VEUS_DEFECTE),
                        help=f"quantes veus de VEUS_DEFECTE fer servir (max {len(VEUS_DEFECTE)}). "
                             f"Amb 1 no es pot distingir una entitat dificil d'una veu que la "
                             f"pronuncia malament; cada veu multiplica el cost de GPU.")
    parser.add_argument("--sense-ancores", action="store_true",
                        help="no afegeixis les entitats ancora de calibracio")
    parser.add_argument("--snr", type=float, default=None,
                        help="SNR fixa en dB per a tots els entorns degradats. Per defecte, "
                             "el punt mig de la forquilla de cada entorn. Mai es sorteja: "
                             "una SNR aleatoria per entitat les fa incomparables entre si.")
    parser.add_argument("--asr-model", default=ASR_MODEL_ID,
                        help=f"model d'ASR (per defecte {ASR_MODEL_ID}, el mateix que evaluate.py)")
    parser.add_argument("--device", default="auto",
                        help="'auto' (defecte) tria la primera GPU utilitzable; "
                             "'cuda:1'/'cpu' la forcen")
    parser.add_argument("--limit", type=int, default=None, help="nomes les N primeres entitats (proves)")
    parser.add_argument("--overwrite", action="store_true",
                        help="torna a mesurar les entitats d'aquest --input encara que ja "
                             "siguin a l'output; la resta de l'output es conserva")
    parser.add_argument("--batch-entities", type=int, default=4,
                        help="entitats processades juntes per crida de TTS/ASR (puja-ho si sobra "
                             "VRAM). El defecte era 8 i en aquesta RTX 4060 Ti de 16 GiB feia OOM "
                             "en el 100%% dels lots; ara a mes hi ha bisexio automatica si passa.")
    parser.add_argument("--asr-batch-size", type=int, default=4,
                        help="mida de lot interna de la pipeline d'ASR. El defecte era 16: mesurat "
                             "en aquesta RTX 4060 Ti, transcriure NOMES 4 audios curts (Whisper "
                             "els omple a 30s cada un, siguin del que siguin) ja consumeix ~4.8 GiB "
                             "-- un lot de 16 en necessitaria ~19 GiB, mes que tota la targeta. "
                             "Aixo, no els altres parametres, era la causa real de l'OOM persistent.")
    args = parser.parse_args()

    entitats = carregar_entitats(args.input)
    if args.limit:
        entitats = entitats[: args.limit]
    if not entitats:
        raise SystemExit(f"Cap entitat trobada a {args.input}")

    resultats: dict[str, dict] = {}
    if args.output.exists():
        previ = json.loads(args.output.read_text(encoding="utf-8"))
        resultats = {r["entitat"]: r for r in previ.get("resultats", [])}
        fallides = {e for e, r in resultats.items() if "error" in r}
        if args.overwrite:
            print(f"{len(resultats)} entitats al fitxer previ; es tornen a mesurar les "
                  f"{len(entitats)} de {args.input.name} i la resta es conserva")
        else:
            pendents = [e for e in entitats if e not in resultats or e in fallides]
            n_ok = len(resultats) - len(fallides)
            print(f"{n_ok} entitats ja verificades a {args.output.name}, {len(pendents)} pendents"
                  + (f" (de les quals {len(fallides & set(entitats))} reintents d'errors previs)"
                     if fallides else ""))
            entitats = pendents

    if not entitats:
        print("Res per fer: totes les entitats ja estaven verificades (--overwrite per repetir-les).")
        return 0

    veus = VEUS_DEFECTE[: max(1, min(args.veus, len(VEUS_DEFECTE)))]
    ancores_injectades: set[str] = set()
    if not args.sense_ancores:
        claus_entitats = {clau(e) for e in entitats}
        ja_candidates = [a for a in ANCORES if clau(a) in claus_entitats]
        noves = [a for a in ANCORES if clau(a) not in claus_entitats
                 and (a not in resultats or "error" in resultats[a])]
        ancores_injectades = set(noves)
        entitats = noves + entitats
        print(f"{len(noves)} ancores injectades per calibrar la condicio: {', '.join(noves) or '-'}")
        if ja_candidates:
            print(f"  {len(ja_candidates)} ja eren a la llista de candidates i s'hi queden "
                  f"com a tals: {', '.join(ja_candidates)}")
    print(f"Veus: {len(veus)} ({', '.join(v['id'] for v in veus)}) | "
          f"{args.n_frases} frases x {1 + len(args.entorns)} condicions "
          f"= {args.n_frases * len(veus) * (1 + len(args.entorns))} mesures per entitat")

    diccionari = carregar_diccionari(args.diccionari)

    expansions = [(e, f) for e in entitats
                  if (f := diccionari.get(e)) and phonetics.es_expansio(e, f)]
    if expansions:
        print(f"  {len(expansions)} entitats amb override d'expansió, no mesurables pel "
              f"round-trip: {', '.join(f'{e} -> {f}' for e, f in expansions[:5])}"
              f"{'...' if len(expansions) > 5 else ''}")
        exclosos = {e for e, _f in expansions}
        entitats = [e for e in entitats if e not in exclosos]

    if diccionari:
        print(f"Diccionari fonetic dispers: {len(diccionari)} overrides "
              f"({args.diccionari.name}); la resta usa l'ortografia raw")
    else:
        print("Diccionari fonetic sense overrides: totes les entitats usen l'ortografia raw")

    args.device = triar_dispositiu(args.device)
    tts, asr = carregar_models(args.device, args.asr_model)

    def desar():
        """Escriu els resultats acumulats a disc."""
        sortida = {
            "_meta": {
                "generat": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tts_model": TTS_MODEL_ID, "asr_model": args.asr_model,
                "n_frases": args.n_frases, "entorns": args.entorns,
                "snr_db": args.snr if args.snr is not None else "punt_mig_de_lentorn",
                "batch_entities": args.batch_entities, "asr_batch_size": args.asr_batch_size,
                "diccionari": str(args.diccionari) if diccionari else None,
                "diccionari_entrades": len(diccionari),
                "veus": [v["id"] for v in veus],
                "ancores": calibracio(resultats),
            },
            "resultats": sorted(
                (r for r in resultats.values() if "tasa_error" in r and r["tasa_error"] is not None),
                key=lambda r: (-r["tasa_error"], -(r.get("distancia_mitjana") or 0)),
            ) + [r for r in resultats.values() if "error" in r],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sortida, ensure_ascii=False, indent=1), encoding="utf-8")

    t0 = time.time()
    lots = [entitats[i:i + args.batch_entities] for i in range(0, len(entitats), args.batch_entities)]
    for i, lot in enumerate(lots, start=1):
        print(f"[lot {i}/{len(lots)}] {len(lot)} entitats: {', '.join(lot[:3])}"
              f"{'...' if len(lot) > 3 else ''}", flush=True)
        resultats_lot = processar_lot_resilient(
            tts, asr, lot, args.n_frases, args.entorns, args.ref_audio,
            args.asr_batch_size, diccionari, args.snr, veus,
        )
        resultats.update(resultats_lot)
        for entitat, r in resultats_lot.items():
            r["font_input"] = args.input.name
            if entitat in ancores_injectades:
                r["ancora"] = True
            if "error" in r:
                print(f"  {entitat:30} ERROR: {r['error'][:80]}")
            else:
                print(f"  {entitat:30} tasa_error={r['tasa_error']}  {r['tasa_error_per_condicio']}")

        desar()

    vals = [r["tasa_error"] for r in resultats.values() if r.get("tasa_error") is not None]
    print(f"\n{len(vals)} entitats verificades en {time.time()-t0:.0f}s")
    if vals:
        print(f"tasa d'error mitjana: {sum(vals)/len(vals):.2f}  "
              f"(min {min(vals):.2f}, max {max(vals):.2f})")

    cal = calibracio(resultats)
    if cal:
        print(f"\nCALIBRACIO ({cal['n']} ancores): tasa d'error {cal['tasa_error_mitjana']:.2f} "
              f"| per condicio {cal['per_condicio']}")
        print(f"  {cal['per_entitat']}")
        if cal["tasa_error_mitjana"] >= 0.2:
            print("  AVIS: les ancores haurien de rondar 0. Una tasa alta apunta a la "
                  "condicio (SNR massa baixa) o al TTS, no a les entitats: revisa-ho abans "
                  "d'aplicar cap llindar.")
        else:
            print(f"  Les ancores encerten: el senyal per sobre de {cal['tasa_error_mitjana']:.2f} "
                  f"es atribuible a l'entitat.")
    print(f"Escrit a {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

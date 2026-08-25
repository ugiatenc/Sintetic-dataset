#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador round-trip: TTS -> entorn acustic -> Whisper -> mesura d'error real.

Per que existeix aquest script
-------------------------------
`util_dataset` (si una entitat val la pena reforçar-la al dataset sintetic) avui el
decideix un LLM endevinant, sense haver sentit mai com sona l'entitat en boca d'un TTS
ni com la transcriu Whisper. Aquest script substitueix l'endevinalla per una mesura:
sintetitza l'entitat dins de frases portadores neutres (amb la seva forma FONETICA, la
que genera `dictionary.ipynb` -- no l'ortografia raw, veure mes avall), la passa per les
condicions acustiques reals del pipeline (`src/acoustic_sim.py`) i la transcriu amb el
mateix Whisper large-v3 que fa servir `evaluate.py`. Si la transcripcio no conte
l'entitat (en la seva ortografia RAW, la de veritat), es un fallo real, no una suposicio.

Per que la forma fonetica i no la raw
---------------------------------------
Si OmniVoice rep "Lamine Yamal" tal qual, ja el llegeix malament (accentuacio i vocals
equivocades) abans que Whisper hi entri en joc -- el resultat barreja "el TTS no sap
llegir-ho" amb "Whisper no ho reconeixeria ni ben pronunciat", dues coses diferents.
Es va comprovar empiricament: sintetitzant amb la fonetica ja corregida ("Lamín Yamal")
Whisper SEGUEIX fallant ("la Minyamal"), confirmant que el fallo es real i no un artefacte
de pronunciacio -- pero nomes es pot saber la diferencia si es sintetitza amb la forma
que un TTS ben configurat faria servir de veritat. Per aixo aquest script llegeix
`lab/entitats/diccionaris/diccionari_fonetic_castellano.json` (generat per
`dictionary.ipynb`, que ara agafa Font A + Font B com a input) i compara sempre contra
la forma RAW, que es l'etiqueta real que el dataset ha d'ensenyar a transcriure. Si una
entitat encara no te entrada al diccionari, sintetitza amb la seva ortografia tal qual i
ho avisa -- degradacio, no error.

Que NO fa aquest script
------------------------
No decideix `util_dataset`. Escriu `tasa_error` per entitat i condicio; la decisio de
llindar es queda fora. Barrejar mesura i decisio en un sol lloc fa mes dificil canviar
de criteri mes endavant sense tornar a executar tot el round-trip.

Cost de computo (mesurat en aquesta maquina, RTX 4060 Ti)
-----------------------------------------------------------
La primera versio d'aquest script cridava TTS i ASR una entitat cada cop (`batch_size=1`
a la GPU). Mesurat en directe sobre la Font B (98 entitats): ~13s/entitat, GPU al 96%
d'us pero mal aprofitada -- transformers ho avisa literalment ("using the pipelines
sequentially on GPU"). A 1.000 entitats aixo son ~3,7 hores.

Aquesta versio agrupa moltes entitats en cada crida (`--batch-entities`, per defecte 8):
una sola trucada de TTS per a totes les frases del lot, i una sola trucada d'ASR (amb el
seu propi `--asr-batch-size` intern) per a tots els audios del lot. La mesura NO canvia
-- son les mateixes frases, els mateixos entorns, la mateixa comparacio -- nomes com
s'envien a la GPU. Ajusta `--batch-entities` a la baixa si hi ha poca VRAM lliure.

Exemple:
    python3 src/verify_entities.py --limit 10                    # prova rapida
    python3 src/verify_entities.py                                # Font B sencera
    python3 src/verify_entities.py --input entidades_candidatas.json --overwrite
    python3 src/verify_entities.py --batch-entities 16 --asr-batch-size 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_INPUT = ROOT / "lab/entitats/entidades_fuente_b_validadas.json"
DEFAULT_OUTPUT = ROOT / "lab/entitats/entidades_verificades_roundtrip.json"
DEFAULT_REF_AUDIO = ROOT / "datasets/audios/referencia/audio_referencia.wav"
DEFAULT_DICCIONARI = ROOT / "lab/entitats/diccionaris/diccionari_fonetic_castellano.json"

TTS_MODEL_ID = "k2-fsa/OmniVoice"
ASR_MODEL_ID = "openai/whisper-large-v3"
SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_ASR = 16000

# Mateixa taula que evaluate.py: cada script del projecte la duplica expressament en
# comptes de compartir-la, per no acoblar-se entre si.
ACC = str.maketrans("áéíóúüàèìòù", "aeiouuaeiou")

# Frases portadores neutres: no exigeixen flexio de genere/nombre sobre l'entitat
# (a diferencia de generate_sentences.ipynb) perque aqui no hi ha LLM que ho resolgui.
# Es tria deliberadament NO fer-les generar per un LLM: aquest script mesura Whisper,
# no vol introduir soroll addicional del generador de frases.
FRASES_PORTADORES = [
    "Hoy hemos hablado de {E} en el informativo.",
    "La noticia sobre {E} ha ocupado gran parte del boletín.",
    "Todo gira en torno a {E} esta semana.",
    "Los datos disponibles sobre {E} son objeto de debate.",
]

# Entorns degradats per defecte: carrer i telefon son condicions reals on les
# entitats fallen mes sovint que en estudi net (veure ENTORNS a acoustic_sim.py).
ENTORNS_DEFECTE = ["carrer", "telefon"]


def norm(text: str) -> str:
    text = text.lower().strip()
    for c in [",", ".", "!", "¡", "?", "¿", ";", ":", '"', "'", "«", "»", "…", "—", "-", "(", ")"]:
        text = text.replace(c, " ")
    return " ".join(text.split())


def clau(text: str) -> str:
    return norm(text).translate(ACC)


def entitat_a_la_transcripcio(entitat: str, transcripcio: str) -> bool:
    """Encert simple i estricte: la forma normalitzada de l'entitat apareix literal
    dins de la transcripcio normalitzada. Sense marge fuzzy: aquest script existeix
    per detectar fallades, i ser tolerant aqui les amagaria."""
    e, t = clau(entitat), clau(transcripcio)
    return bool(e) and re.search(rf"(?<!\w){re.escape(e)}(?!\w)", t) is not None


def carregar_entitats(path: Path) -> list[str]:
    """Llista d'entitats a verificar. Accepta:
    - un JSON amb "validadas"/"rechazadas" (format d'`entidades_fuente_*_validadas.json`)
    - una llista plana de strings o de dicts amb "entidad"/"entrada"/"grafia_correcta"

    De les "rechazadas" NOMES s'agafen les que el LLM va tipificar com a entitat real
    (tipo != NO_ENTIDAD): son exactament el cas que aquest script existeix per resoldre.
    El LLM va rebutjar `Lamine Yamal`, `OSCE` i `Washington` de la Font B com a
    "no val la pena" sense cap evidencia de com sona Whisper amb elles -- son la mostra
    mes valuosa per verificar, no soroll a descartar.
    """
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
    entitats = [e for e in dict.fromkeys(entitats) if e]  # deduplicat preservant ordre

    # Si 'Cristiano Ronaldo' i 'Ronaldo' son alhora a la llista, es descarta la forma
    # curta: verificar-la a part gasta una altra crida de TTS+ASR sencera per una
    # entitat que ja queda coberta per la llarga, i en un corpus gran la GPU es el
    # recurs que s'acaba abans (veure capçalera del fitxer).
    #
    # Guarda anti-autoreferencia: aquest diccionari es clau per `grafia_correcta`, no
    # per l'span cru del NER, i el LLM sovint YA expandeix la forma curta a la llarga
    # en corregir la grafia ('Ronaldo' -> grafia_correcta 'Cristiano Ronaldo'). Sense
    # aquest guard, `forma_larga_de['Cristiano Ronaldo'] == 'Cristiano Ronaldo'`
    # (autoreferencia) feia que es descartés la forma LLARGA en comptes de la curta.
    presents = set(entitats)
    curtes = {e for e in entitats
              if forma_larga_de.get(e) not in (None, e) and forma_larga_de[e] in presents}
    if curtes:
        print(f"  {len(curtes)} formes curtes descartades per tenir la seva forma_larga a la mateixa llista: "
              f"{', '.join(sorted(curtes)[:5])}{'...' if len(curtes) > 5 else ''}")
    return [e for e in entitats if e not in curtes]


def carregar_diccionari(path: Path) -> dict[str, str]:
    """Diccionari fonetic {entitat_raw: forma_per_al_tts}, generat per `dictionary.ipynb`.

    Sense aixo el round-trip sintetitza l'ortografia tal qual, i el TTS ja la llegeix
    malament (accentuacio, grups consonantics impossibles) abans que Whisper hi entri
    en joc -- confonent "l'entitat es dificil" amb "OmniVoice no la sap llegir". Es va
    comprovar empiricament amb `Lamine Yamal`: fins i tot amb la fonetica ja corregida
    ("Lamín Yamal") Whisper segueix fallant, cosa que confirma que el fallo es real i no
    nomes de pronunciacio -- pero cal la fonetica per poder distingir-ho cas a cas.

    Si el fitxer no existeix encara, es torna buit i cada entitat sintetitza amb la seva
    ortografia raw (el mateix comportament que abans de connectar aquest pas).
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def carregar_models(device: str):
    """Importa i carrega TTS + ASR. Es fa dins d'una funcio (no al top-level) perque
    `--help` i els tests de `carregar_entitats`/`entitat_a_la_transcripcio` no
    requereixin GPU ni els pesos descarregats."""
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
    print(f"Carregant {ASR_MODEL_ID}...")
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID,
                   dtype=torch.float16, device=device)
    return tts, asr


def sintetitzar_lot(tts, textos: list[str], ref_audio: str) -> list:
    """Una sola crida de TTS per a tots els `textos`, del lot sencer o de multiples
    entitats alhora. `ref_audio` es repeteix explicitament (no es confia en el
    broadcasting implicit d'OmniVoice) perque la mida hagi de coincidir sempre."""
    audios = tts.generate(text=textos, ref_audio=[ref_audio] * len(textos))
    return [a.astype("float32") for a in audios]


def degradar(waveform, entorn_id: str, llavor: int):
    """Aplica un entorn acustic del pipeline principal (mateix codi que fa servir
    `generate_voices_environments.ipynb`), amb llavor derivada de l'entitat i la frase
    perque reprendre l'execucio doni sempre les mateixes condicions."""
    import numpy as np
    from acoustic_sim import mesclar_amb_entorn

    rng = np.random.default_rng(llavor)
    audio, _meta = mesclar_amb_entorn(waveform, SAMPLE_RATE_TTS, entorn_id, rng)
    return audio


def transcriure_lot(asr, waveforms_24k: list, asr_batch_size: int) -> list[str]:
    """Una sola crida d'ASR per a tots els `waveforms_24k`. `transformers` agrupa
    internament segons `batch_size` (veure `Pipeline.__call__`); sense aixo processa
    un audio cada cop encara que li passis una llista, que es exactament l'avis que
    la propia libreria treu quan ho detecta ("using the pipelines sequentially")."""
    import soxr

    entrades = [
        {"raw": soxr.resample(w, SAMPLE_RATE_TTS, SAMPLE_RATE_ASR), "sampling_rate": SAMPLE_RATE_ASR}
        for w in waveforms_24k
    ]
    sortides = asr(entrades, batch_size=asr_batch_size,
                   generate_kwargs={"language": "es", "task": "transcribe"})
    return [s["text"].strip() for s in sortides]


def verificar_lot(tts, asr, entitats: list[str], n_frases: int, entorns: list[str],
                  ref_audio: str, asr_batch_size: int, diccionari: dict[str, str]) -> dict[str, dict]:
    """Verifica moltes entitats alhora: una crida de TTS i una d'ASR per a tot el lot,
    enlloc d'una parella de crides per entitat. El resultat per entitat es identic al
    d'una versio seqüencial -- nomes canvia com s'agrupen les crides a la GPU."""
    import hashlib

    plantilles = FRASES_PORTADORES[:max(1, min(n_frases, len(FRASES_PORTADORES)))]

    # 1. Totes les frases de totes les entitats del lot, en un unic TTS batch.
    #    OmniVoice sintetitza la forma FONETICA (`tts_text`), mai la raw: si li donessim
    #    "Lamine Yamal" tal qual, el TTS ja el llegiria malament abans que Whisper hi
    #    entri en joc. La comparacio final es sempre contra `ent` (la forma raw, que es
    #    l'etiqueta real del dataset) -- aixo no canvia.
    tasques = [(ent, p.format(E=diccionari.get(ent, ent))) for ent in entitats for p in plantilles]
    audios_nets = sintetitzar_lot(tts, [frase for _, frase in tasques], ref_audio)

    # 2. Expandim cada frase a les seves condicions (net + entorns degradats). La
    #    degradacio es processament de senyal per CPU, no cal ni val la pena batejar-la.
    condicions = ["net"] + list(entorns)
    items = []  # (entitat, frase, condicio, audio)
    for (entitat, frase), audio_net in zip(tasques, audios_nets):
        for condicio in condicions:
            if condicio == "net":
                senyal = audio_net
            else:
                llavor = int.from_bytes(
                    hashlib.sha256(f"{entitat}|{condicio}|{frase}".encode()).digest()[:8], "big"
                )
                senyal = degradar(audio_net, condicio, llavor)
            items.append((entitat, frase, condicio, senyal))

    # 3. Totes les transcripcions del lot, en un unic ASR batch.
    transcripcions = transcriure_lot(asr, [it[3] for it in items], asr_batch_size)

    # 4. Reagrupem per entitat i calculem les mateixes metriques que abans.
    per_entitat: dict[str, list[dict]] = {ent: [] for ent in entitats}
    for (entitat, frase, condicio, _audio), transcripcio in zip(items, transcripcions):
        encert = entitat_a_la_transcripcio(entitat, transcripcio)
        per_entitat[entitat].append({
            "condicio": condicio, "frase": frase,
            "transcripcio": transcripcio, "encert": encert,
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
            "tasa_error_per_condicio": per_condicio,
            "exemples_fallats": [f for f in files if not f["encert"]],
        }
    return resultats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"JSON amb les entitats a verificar (per defecte: {DEFAULT_INPUT.name})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diccionari", type=Path, default=DEFAULT_DICCIONARI,
                        help="diccionari fonetic de dictionary.ipynb (entitat -> forma pel TTS)")
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--n-frases", type=int, default=3,
                        help=f"frases portadores per entitat (max {len(FRASES_PORTADORES)})")
    parser.add_argument("--entorns", nargs="*", default=ENTORNS_DEFECTE,
                        help="entorns degradats a provar, a mes de 'net' (veure ENTORNS a acoustic_sim.py)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="nomes les N primeres entitats (proves)")
    parser.add_argument("--overwrite", action="store_true", help="reverifica entitats ja presents a l'output")
    parser.add_argument("--batch-entities", type=int, default=8,
                        help="entitats processades juntes per crida de TTS/ASR (puja-ho si sobra VRAM)")
    parser.add_argument("--asr-batch-size", type=int, default=16,
                        help="mida de lot interna de la pipeline d'ASR")
    args = parser.parse_args()

    entitats = carregar_entitats(args.input)
    if args.limit:
        entitats = entitats[: args.limit]
    if not entitats:
        raise SystemExit(f"Cap entitat trobada a {args.input}")

    resultats: dict[str, dict] = {}
    if args.output.exists() and not args.overwrite:
        previ = json.loads(args.output.read_text(encoding="utf-8"))
        resultats = {r["entitat"]: r for r in previ.get("resultats", [])}
        pendents = [e for e in entitats if e not in resultats]
        print(f"{len(resultats)} entitats ja verificades a {args.output.name}, "
              f"{len(pendents)} pendents")
        entitats = pendents

    if not entitats:
        print("Res per fer: totes les entitats ja estaven verificades (--overwrite per repetir-les).")
        return 0

    diccionari = carregar_diccionari(args.diccionari)
    if diccionari:
        sense_fonetica = [e for e in entitats if e not in diccionari]
        print(f"Diccionari fonetic: {len(diccionari)} entrades ({args.diccionari.name})")
        if sense_fonetica:
            print(f"  AVIS: {len(sense_fonetica)}/{len(entitats)} entitats sense entrada al "
                  f"diccionari, sintetitzaran amb la seva ortografia raw: "
                  f"{', '.join(sense_fonetica[:5])}{'...' if len(sense_fonetica) > 5 else ''}")
    else:
        print(f"AVIS: no hi ha diccionari fonetic a {args.diccionari}; totes les entitats "
              f"sintetitzaran amb la seva ortografia raw (genera'l amb dictionary.ipynb).")

    tts, asr = carregar_models(args.device)

    def desar():
        sortida = {
            "_meta": {
                "generat": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tts_model": TTS_MODEL_ID, "asr_model": ASR_MODEL_ID,
                "n_frases": args.n_frases, "entorns": args.entorns,
                "batch_entities": args.batch_entities, "asr_batch_size": args.asr_batch_size,
                "diccionari": str(args.diccionari) if diccionari else None,
                "diccionari_entrades": len(diccionari),
            },
            "resultats": sorted(
                (r for r in resultats.values() if "tasa_error" in r and r["tasa_error"] is not None),
                key=lambda r: -r["tasa_error"],
            ) + [r for r in resultats.values() if "error" in r],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sortida, ensure_ascii=False, indent=1), encoding="utf-8")

    t0 = time.time()
    lots = [entitats[i:i + args.batch_entities] for i in range(0, len(entitats), args.batch_entities)]
    for i, lot in enumerate(lots, start=1):
        print(f"[lot {i}/{len(lots)}] {len(lot)} entitats: {', '.join(lot[:3])}"
              f"{'...' if len(lot) > 3 else ''}", flush=True)
        try:
            resultats_lot = verificar_lot(tts, asr, lot, args.n_frases, args.entorns,
                                          args.ref_audio, args.asr_batch_size, diccionari)
            resultats.update(resultats_lot)
            for entitat, r in resultats_lot.items():
                print(f"  {entitat:30} tasa_error={r['tasa_error']}  {r['tasa_error_per_condicio']}")
        except Exception as e:
            # Un lot sencer que peta (p.ex. OOM) no s'ha de perdre en silenci ni tombar
            # tota l'execucio: es marca cada entitat del lot com a error i es continua.
            print(f"  ERROR al lot: {e}")
            for entitat in lot:
                resultats[entitat] = {"entitat": entitat, "error": str(e)}

        desar()  # despres de cada lot, no de cada entitat: si es talla, nomes es reprocessa el lot en curs

    vals = [r["tasa_error"] for r in resultats.values() if r.get("tasa_error") is not None]
    print(f"\n{len(vals)} entitats verificades en {time.time()-t0:.0f}s")
    if vals:
        print(f"tasa d'error mitjana: {sum(vals)/len(vals):.2f}  "
              f"(min {min(vals):.2f}, max {max(vals):.2f})")
    print(f"Escrit a {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

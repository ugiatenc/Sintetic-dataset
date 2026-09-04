#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador round-trip: TTS -> entorn acustic -> Whisper -> mesura d'error real.

Sintetitza cada entitat (amb la seva forma FONETICA) en frases portadores neutres, les
passa pels entorns acustics del pipeline (`src/acoustic_sim.py`) i les transcriu amb
Whisper large-v3. Si la transcripcio no conte l'entitat en la seva ortografia RAW, es
un fallo mesurat, no un LLM endevinant si "val la pena" reforçar-la.

Per entitat: `--n-frases` frases x `--veus` veus x (net + `--entorns`) condicions,
totes amb la mateixa SNR/RIR/soroll perque les tases d'error siguin comparables entre
si. Inclou entitats ANCORA trivials (Madrid, Barcelona...) com a linia base: si elles
tambe fallen, el problema es la condicio acustica, no l'entitat.

Abans d'executar-ho cal
------------------------
Diccionari de TREBALL generat (`lab/dictionary.ipynb` amb `ETAPA='treball'`), amb
fonetica per a totes les entitats a verificar. Sense aixo cada entitat sintetitza amb
la seva ortografia crua i la mesura barreja "el TTS ho llegeix malament" amb "Whisper
no ho reconeixeria ni ben pronunciat". L'script avisa quantes entitats no en tenen.

Que NO fa: no decideix el llindar de tall. Escriu `tasa_error` i `distancia_mitjana`
per entitat; `create_entities_list.ipynb` es qui aplica `UMBRAL_TASA_ERROR`.

Us
--
    python3 src/verify_entities.py --limit 10             # prova rapida
    python3 src/verify_entities.py                         # Font B sencera
    python3 src/verify_entities.py --veus 1 --n-frases 2   # barat, menys fiable
    python3 src/verify_entities.py --snr 6                 # condicio mes dura

`--batch-entities` (per defecte 4) agrupa entitats per crida de TTS/ASR; puja-ho si
sobra VRAM. Si peta per OOM, es biseca el lot (i, si nomes queda 1 entitat, es
redueixen les veus) i es reintenta sol; una entitat que quedi en `error` es reintenta
automaticament a la propera execucio.

El coll d'ampolla real de VRAM sol ser `--asr-batch-size` (per defecte 4), no
`--batch-entities` ni `--veus`: Whisper omple cada audio a 30s per al seu encoder
independentment de la seva durada real, i en una RTX 4060 Ti de 16 GiB transcriure
NOMES 4 audios curts ja consumeix ~4.8 GiB. Si torna a petar, abans de tocar
`--batch-entities` o `--veus`, prova a baixar `--asr-batch-size` a 2 o 1.
"""

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

# Cache de HuggingFace dins del projecte. En aquesta maquina `~/.cache/huggingface`
# pertany a root i qualsevol descarrega hi peta amb PermissionError. Passar `cache_dir=`
# (el que fa `create_entities_list.ipynb` amb el tokenizer) aqui no serveix: OmniVoice
# resol el model amb `snapshot_download()` sense `cache_dir`, de manera que nomes fa cas
# de la variable d'entorn. Ha d'estar posada ABANS d'importar `huggingface_hub`, que
# llegeix les rutes en importar-se -- per aixo es aqui i no dins de `carregar_models()`.
# `setdefault`: si algu ja te el seu propi HF_HOME, es respecta.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import phonetics

DEFAULT_INPUT = ROOT / "lab/entitats/entidades_fuente_b_validadas.json"
DEFAULT_OUTPUT = ROOT / "lab/entitats/entidades_verificades_roundtrip.json"
DEFAULT_REF_AUDIO = ROOT / "datasets/audios/referencia/audio_referencia.wav"
# Diccionari de TREBALL: cobreix tots els candidats, inclosos els que el llindar
# descartara despres. El diccionari FINAL nomes te els seleccionats i encara no
# existeix quan s'executa el round-trip (veure src/phonetics.py).
DEFAULT_DICCIONARI = phonetics.ruta_diccionari("castellano", "treball")

TTS_MODEL_ID = "k2-fsa/OmniVoice"
# El MATEIX model que `evaluate.py` fa servir per detectar els errors de Font A. Abans
# aqui hi havia `large-v3-turbo`, un model destil·lat diferent: la Font B es mesurava
# contra un model que no era el que havia fallat, i les dues fonts no eren comparables.
# `--asr-model openai/whisper-large-v3-turbo` el recupera si cal anar mes rapid.
ASR_MODEL_ID = "openai/whisper-large-v3"
SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_ASR = 16000

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

# Veus de la mesura. El vocabulari `instruct` d'OmniVoice es tancat (veure VEUS a
# `generate_voices_environments.ipynb`); aquestes tres son un subconjunt de les que
# fara servir el dataset. Amb una sola veu no es pot distingir "Whisper no reconeix
# aquesta entitat" de "no la reconeix EN AQUESTA veu", i el dataset final en tindra moltes.
VEUS_DEFECTE = [
    {"id": "female_middle", "instruct": "female, middle-aged, moderate pitch"},
    {"id": "male_middle", "instruct": "male, middle-aged, moderate pitch"},
    {"id": "female_young", "instruct": "female, young adult, high pitch"},
]

# Entitats ancora: frequents, ben conegudes i de grafia trivial. Whisper les hauria
# d'encertar gairebe sempre, aixi que la seva tasa d'error mesura la DIFICULTAT DE LA
# CONDICIO, no la de l'entitat. Sense aquesta referencia no hi ha manera de saber si un
# 30% d'error vol dir "entitat dificil" o "hem posat el soroll massa alt": es mesurava
# en una escala sense zero.
#
# Totes han d'encaixar a FRASES_PORTADORES SENSE article. Amb "el Gobierno" la plantilla
# dona "hablado de el Gobierno", que Whisper transcriu correctament com a "del Gobierno";
# la comparacio literal no hi trobava "el gobierno" i l'ancora fallava el 100% sempre,
# fent creure que la condicio acustica estava trencada quan el trencat era el control.
# S'hi inclou una de multiparaula per comprovar que l'alineament multiparaula funciona.
ANCORES = ["Madrid", "Barcelona", "Valencia", "Europa", "Pedro Sánchez"]


def clau(text: str) -> str:
    """Clau de comparacio, compartida amb el diccionari (`phonetics.clau`).

    Abans hi havia aqui una taula `str.maketrans` propia que nomes cobria vocals
    accentuades castellanes i catalanes; `phonetics.clau` normalitza qualsevol
    diacritic amb `unicodedata` i preserva la `ñ`. Tenir-ne dues versions divergents
    es el que va deixar passar el fallo de cerca d'`AEMET`.
    """
    return phonetics.clau(text)


def variants_entitat(entitat: str) -> list[str]:
    """Formes en que una transcripcio CORRECTA pot escriure l'entitat.

    Nomes cobreix diferencies d'escriptura de la mateixa cosa, no aproximacions: Whisper
    escriu les sigles de maneres que la comparacio literal donava per fallides tot i ser
    encerts. `OSCE` pot sortir com "OSCE" o "O.S.C.E." (que normalitzat es "o s c e"), i
    `EE.UU.` com "EEUU". Comptar-ho com a error inflava la `tasa_error` justament de les
    sigles, que es el grup que mes amunt apareixia al ranking de dificultat.
    """
    base = clau(entitat)
    if not base:
        return []
    variants = {base}
    lletres = base.replace(" ", "")
    net = entitat.replace(".", "").replace(" ", "")

    # Sigla curta: accepta-la deletrejada ("o s c e") i compactada ("osce").
    if 2 <= len(net) <= 6 and net.isupper():
        variants.add(" ".join(lletres))
        variants.add(lletres)
    # Sigla que ja porta punts o espais al seu nom ("EE.UU." -> "eeuu").
    elif " " in base and len(lletres) <= 6:
        variants.add(lletres)

    return sorted(variants, key=len, reverse=True)


def sense_h(text: str) -> str:
    """La `h` castellana es muda: no hi ha cap diferencia acustica entre "Ormuz" i
    "Hormuz", aixi que escriure-la o no es una convencio ortografica, no una fallada
    del model. Sense aixo `estrecho de Ormuz` puntuava `tasa_error = 1.0` fins i tot
    en audio net perque Whisper escrivia sempre "Hormuz" -- i entrava a la llista
    final com si fos de les entitats mes dificils del corpus.

    Nomes afecta la `h`. Les diferencies que SI porten so (`Bizkaia`/`Vizcaya`,
    `Urkiola`/`Urquiola`) segueixen comptant com a fallada: son justament les
    grafies oficials que el dataset ha de reforçar.
    """
    return re.sub(r"h", "", text)


def entitat_a_la_transcripcio(entitat: str, transcripcio: str) -> bool:
    """Encert estricte: alguna forma valida d'escriure l'entitat apareix literal dins de
    la transcripcio normalitzada. Sense marge fuzzy -- aquest script existeix per detectar
    fallades i ser tolerant aqui les amagaria; les variants de `variants_entitat` i la
    `h` muda no son aproximacions sino ortografies alternatives de la mateixa entitat."""
    t = clau(transcripcio)
    for variant in variants_entitat(entitat):
        for text, patro in ((t, variant), (sense_h(t), sense_h(variant))):
            if patro and re.search(rf"(?<!\w){re.escape(patro)}(?!\w)", text):
                return True
    return False


def _levenshtein(a: str, b: str) -> int:
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
    """CER entre l'entitat i el tros de transcripcio que mes se li assembla (0 = exacte).

    El boolea `encert` no distingeix "Pete Hegset" (una lletra) de soroll sencer, i amb
    3 frases x 3 condicions la `tasa_error` nomes te 10 valors possibles: massa gruixut
    per ordenar candidats. Aquesta distancia es una segona lectura de les MATEIXES dades,
    sense cost de GPU, i dona un ranking continu de "com de lluny queda".
    """
    e = clau(entitat)
    mots = clau(transcripcio).split()
    if not e or not mots:
        return 1.0
    n = len(e.split())
    millor = 1.0
    # Es prova amb una paraula menys i una mes perque Whisper parteix i ajunta entitats
    # ("radiogaceta" -> "radio gaceta"), el mateix motiu pel qual `evaluate.py` alinea
    # per regio i no token a token.
    for amplada in {max(1, n - 1), n, n + 1}:
        for i in range(len(mots) - amplada + 1):
            candidat = " ".join(mots[i:i + amplada])
            millor = min(millor, _levenshtein(e, candidat) / len(e))
    return round(min(millor, 1.0), 3)


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
    entitats = [e for e in entitats if e not in curtes]

    # Simbols i unitats (`%`, `ºC`, `km/h`) no es poden mesurar amb aquest metode:
    # el TTS diu la lectura i Whisper escriu el simbol o la lectura segons li convingui,
    # aixi que la comparacio literal contra la grafia raw dona `tasa_error = 1.0`
    # sempre. Deixar-los passar els col·locava al capdamunt del ranking de dificultat.
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
    """Diccionari fonetic {entitat_raw: forma_per_al_tts} de `dictionary.ipynb`.

    Sense aixo el round-trip sintetitza l'ortografia tal qual i el TTS ja la llegeix
    malament abans que Whisper hi entri en joc, confonent "l'entitat es dificil" amb
    "OmniVoice no la sap llegir".

    La cerca la fa `phonetics.Diccionari`, insensible a majuscules i accents: amb el
    `dict.get` directe d'abans, `AEMET` no trobava l'entrada `aemet` del diccionari i
    se sintetitzava en cru sense que res ho indiques.
    """
    return phonetics.carregar(path)


def triar_dispositiu(preferit: str = "auto") -> str:
    """Primera GPU que aquesta build de PyTorch pugui fer servir de debo.

    El defecte era `cuda:0` a cegues i en aquesta maquina `cuda:0` es la GTX 1080 Ti,
    que es Pascal (sm_61): les builds de torch amb CUDA recent ja no la compilen i peta
    amb "no kernel image is available for execution on the device" quan ja s'han carregat
    els pesos. En comptes de comparar capacitats a ma, es prova una operacio petita a cada
    dispositiu i es fa servir el primer que respon -- el mateix que ja feia
    `generate_voices_environments.ipynb`, que era l'unic dels dos que arrencava sol.
    """
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
    print(f"Carregant {asr_model_id}...")
    asr = pipeline("automatic-speech-recognition", model=asr_model_id,
                   dtype=torch.float16, device=device)
    return tts, asr


def sintetitzar_lot(tts, textos: list[str], veu: dict, ref_audio: str) -> list:
    """Una sola crida de TTS per a tots els `textos`, del lot sencer o de multiples
    entitats alhora.

    `veu` es un dels dicts de VEUS_DEFECTE. Amb `instruct` es fa servir el mateix
    mecanisme que `generate_voices_environments.ipynb`; amb `ref_audio` es clona la veu
    d'un .wav i la ruta es repeteix explicitament (sense confiar en el broadcasting
    implicit d'OmniVoice) perque la mida hagi de coincidir sempre.
    """
    kwargs = {"text": textos, "language": "es"}
    if veu.get("instruct"):
        kwargs["instruct"] = veu["instruct"]
    else:
        kwargs["ref_audio"] = [veu.get("ref_audio") or ref_audio] * len(textos)
    return [a.astype("float32") for a in tts.generate(**kwargs)]


def entorn_amb_snr_fixa(entorn_id: str, snr_db: float | None):
    """L'entorn del pipeline pero amb la SNR clavada en comptes de sortejada.

    `ENTORNS["carrer"]` te `snr_db=(8.0, 18.0)` i `mesclar_amb_entorn` en treu un valor
    uniforme a cada crida. Amb la llavor derivada de l'entitat, cada entitat s'acabava
    mesurant a una SNR diferent dins d'una forquilla de 10 dB -- una variacio de
    dificultat mes gran que la que separa una entitat facil d'una de dificil, cosa que
    feia que les `tasa_error` no fossin comparables entre si.

    Amb `snr_db=None` es fa servir el punt mig de la forquilla de l'entorn.
    """
    import dataclasses

    from acoustic_sim import ENTORNS

    entorn = ENTORNS[entorn_id]
    if entorn.snr_db is None:
        return entorn
    valor = snr_db if snr_db is not None else (entorn.snr_db[0] + entorn.snr_db[1]) / 2
    return dataclasses.replace(entorn, snr_db=(valor, valor))


def degradar(waveform, entorn_id: str, llavor: int, snr_db: float | None = None):
    """Aplica un entorn acustic del pipeline principal (mateix codi que
    `generate_voices_environments.ipynb`) i retorna (audio, meta).

    `llavor` NO depen de l'entitat a proposit: totes les entitats han de rebre la
    mateixa RIR, el mateix retall de soroll i el mateix nivell per a una condicio i
    plantilla donades. Si no, la diferencia entre dues `tasa_error` barreja "aquesta
    entitat es mes dificil" amb "a aquesta li ha tocat pitjor sala".

    La `meta` (SNR aplicada, RIR, canal) es guarda al resultat: abans es descartava amb
    `audio, _meta = ...` i no hi havia manera de saber en quines condicions s'havia
    mesurat una entitat, tot i que el pipeline principal si que ho registra al manifest.
    """
    import numpy as np
    from acoustic_sim import mesclar_amb_entorn

    rng = np.random.default_rng(llavor)
    entorn = entorn_amb_snr_fixa(entorn_id, snr_db)
    return mesclar_amb_entorn(waveform, SAMPLE_RATE_TTS, entorn, rng)


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
                  ref_audio: str, asr_batch_size: int, diccionari,
                  snr_db: float | None = None, veus: list[dict] | None = None) -> dict[str, dict]:
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
    veus = veus or VEUS_DEFECTE
    tasques, audios_nets = [], []
    for i_veu, veu in enumerate(veus):
        subtasques = [(ent, i, veu["id"], p.format(E=diccionari.get(ent, ent)))
                      for ent in entitats for i, p in enumerate(plantilles)]
        tasques += subtasques
        audios_nets += sintetitzar_lot(tts, [f for _, _, _, f in subtasques], veu, ref_audio)
        if i_veu < len(veus) - 1:
            # Buidar cache ENTRE veus, no nomes quan ja ha petat una OOM. En aquesta
            # maquina els dos models (OmniVoice + Whisper large-v3 fp16) ja ocupen
            # ~15 GiB d'una GPU de 15.57 -- 3 crides seguides de `generate()` sense
            # alliberar res van fragmentar prou memoria per fer OOM intentant reservar
            # nomes 60 MiB, fins i tot amb un sol entitat per lot (`--batch-entities 1`).
            alliberar_vram()

    # 2. Expandim cada frase a les seves condicions (net + entorns degradats). La
    #    degradacio es processament de senyal per CPU, no cal ni val la pena batejar-la.
    #
    #    La llavor surt nomes de (condicio, index de plantilla): identica per a totes les
    #    entitats. Abans hi entrava l'entitat i la frase, de manera que cada entitat
    #    rebia una sala i una SNR diferents i les taxes d'error no es podien comparar.
    condicions = ["net"] + list(entorns)
    items = []  # (entitat, frase, veu, condicio, audio, meta)
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

    # 3. Totes les transcripcions del lot, en un unic ASR batch.
    transcripcions = transcriure_lot(asr, [it[4] for it in items], asr_batch_size)

    # 4. Reagrupem per entitat i calculem les mateixes metriques que abans.
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
            # Senyal continua: amb 9 mesures la `tasa_error` nomes pot valer 0, 0.111,
            # 0.222... i empata massa candidats. La distancia mitjana els desempata.
            "distancia_mitjana": round(sum(f["distancia"] for f in files) / len(files), 3)
                                 if files else None,
            "tasa_error_per_condicio": per_condicio,
            # Per veu: si una entitat nomes falla amb una veu, el problema es de
            # pronunciacio d'aquella veu, no de l'entitat.
            "tasa_error_per_veu": {
                v: round(1 - sum(f["encert"] for f in fv) / len(fv), 3)
                for v in {f["veu"] for f in files}
                for fv in [[f for f in files if f["veu"] == v]]
            },
            # Condicions exactes en que s'ha mesurat (SNR, RIR, canal). Son identiques
            # per a totes les entitats, i tenir-les escrites permet comprovar-ho.
            "condicions_aplicades": {
                c: {k: v for k, v in m.items() if v is not None}
                for c, m in condicions_meta.items()
            },
            "exemples_fallats": [f for f in files if not f["encert"]],
        }
    return resultats


def alliberar_vram() -> None:
    """Buida la cache de PyTorch despres d'una OOM.

    Sense aixo la memoria reservada pel lot que ha petat queda fragmentada i el
    lot seguent torna a petar encara que sigui mes petit: es la cadena que va
    convertir una OOM en 56 entitats fallides de 56."""
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
    """Verifica un lot i, si peta per falta de memoria, el parteix per la meitat
    i reintenta cada mitat. Nomes s'acaba marcant com a error una entitat que
    falla tota sola, que ja no es un problema de mida de lot.

    L'`asr_batch_size` baixa en proporcio al lot: si el que no cabia era el lot
    d'ASR, mantenir-lo alt tornaria a petar amb menys entitats.

    La biseccio nomes redueix el nombre d'ENTITATS per lot -- mai el de veus. En una
    GPU on els dos models ja ocupen ~15 GiB d'una de 15.57 (aquesta maquina), 3 veus
    poden fer OOM fins i tot amb una unica entitat per lot; sense mes marge de biseccio
    per entitats, `len(lot) == 1` es donava per irrecuperable. Abans de rendir-se,
    doncs, es reintenta amb menys veus.
    """
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
                # Es marca perque qui llegeixi el resultat sapiga que aquesta entitat
                # es va mesurar amb menys veus que la resta -- no es directament
                # comparable amb una `tasa_error` calculada amb totes les veus.
                #
                # `if "veus_reduides" not in r`: si la recursio ha degradat mes d'un
                # cop (3 veus -> 2 -> 1), cada nivell torna aqui i, sense aquest guard,
                # el nivell mes extern (2 veus) sobreescriuria l'anotacio del nivell
                # que REALMENT ha tingut exit (1 veu) amb un numero de veus mes gran
                # del que s'ha fet servir de veritat.
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
    """Resum de com han anat les entitats ancora: la linia base de la condicio.

    Una `tasa_error` nomes es interpretable comparada amb aixo. Si `Madrid` falla el 20%
    en `carrer`, un 25% en una entitat qualsevol no vol dir que sigui dificil -- vol dir
    que la condicio ho es. I si les ancores fallen molt, el que cal revisar es la SNR o
    el TTS, no la llista d'entitats.
    """
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

    # El fitxer previ es carrega SEMPRE, tambe amb `--overwrite`. Abans `--overwrite`
    # entrava amb `resultats` buit i, com que el fitxer es reescriu sencer al final,
    # s'enduia tot el que no fos a l'`--input` d'aquella crida: una re-mesura de Font B
    # esborrava les entitats mesurades amb
    # `--input entidades_fuente_a_omitidas.json`, que es justament el cas que el README
    # descriu com a acumulatiu. `--overwrite` vol dir "torna a mesurar les d'aquest
    # input", no "oblida la resta".
    resultats: dict[str, dict] = {}
    if args.output.exists():
        previ = json.loads(args.output.read_text(encoding="utf-8"))
        resultats = {r["entitat"]: r for r in previ.get("resultats", [])}
        # Una entitat amb `error` (tipicament OOM) NO esta verificada: nomes va
        # petar. Abans entrava igualment a `resultats` i quedava exclosa de
        # `pendents`, de manera que un lot fallit es donava per fet per sempre i
        # nomes `--overwrite` (que reprocessa TOT) el podia rescatar. Amb 56 de 56
        # entitats en OOM, aixo va deixar un output sencer d'errors que cap
        # reexecucio normal hauria arreglat.
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

    # Ancores de calibracio al capdavant, per veure aviat si la condicio es raonable
    # sense esperar-se a que acabi tota l'execucio.
    veus = VEUS_DEFECTE[: max(1, min(args.veus, len(VEUS_DEFECTE)))]
    ancores_injectades: set[str] = set()
    if not args.sense_ancores:
        # Nomes les que encara no s'han mesurat: en una represa no cal repetir-les
        # (la calibracio es llegeix igual del que ja hi ha a `resultats`).
        # Per clau normalitzada: `entidades_fuente_a_omitidas.json` porta "Pedro Sánchez",
        # que tambe es ancora, i una diferencia de caixa o accent l'hauria mesurat dues
        # vegades -- una com a candidata i una altra com a instrumentacio.
        claus_entitats = {clau(e) for e in entitats}
        ja_candidates = [a for a in ANCORES if clau(a) in claus_entitats]
        noves = [a for a in ANCORES if clau(a) not in claus_entitats
                 and (a not in resultats or "error" in resultats[a])]
        ancores_injectades = set(noves)
        entitats = noves + entitats
        print(f"{len(noves)} ancores injectades per calibrar la condicio: {', '.join(noves) or '-'}")
        if ja_candidates:
            # Aquestes NO es marquen com a ancora: son candidates de debo i han de poder
            # entrar al dataset si resulta que fallen. Serveixen igualment de calibracio.
            print(f"  {len(ja_candidates)} ja eren a la llista de candidates i s'hi queden "
                  f"com a tals: {', '.join(ja_candidates)}")
    print(f"Veus: {len(veus)} ({', '.join(v['id'] for v in veus)}) | "
          f"{args.n_frases} frases x {1 + len(args.entorns)} condicions "
          f"= {args.n_frases * len(veus) * (1 + len(args.entorns))} mesures per entitat")

    diccionari = carregar_diccionari(args.diccionari)

    # Un override que LLEGEIX l'entitat en comptes de reescriure-la (`EE.UU.` ->
    # "Estados Unidos") fa que el TTS digui la lectura i Whisper l'escrigui: comparat
    # literalment contra la grafia crua dona `tasa_error = 1.0` garantida i col·loca
    # l'entitat al capdamunt del ranking de dificultat sense cap fallada real.
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
                # Linia base de la condicio: si les ancores fallen, el llindar de
                # `tasa_error` s'ha de llegir a partir d'aqui, no de zero.
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
            # Provinença de l'`--input`: `create_entities_list.ipynb` la fa servir per
            # etiquetar l'entitat amb la font real (Font A omesa / Font B) en comptes de
            # donar per fet que tot el que hi ha en aquest fitxer ve de Font B nomes
            # perque aquest n'es el defecte habitual.
            r["font_input"] = args.input.name
            if entitat in ancores_injectades:
                # Marca perque `create_entities_list.ipynb` les pugui excloure: son
                # instrumentacio de mesura, no candidates a entrar al dataset. Nomes
                # les INJECTADES: una ancora que ja venia de la llista de candidates es
                # una candidata de debo i excloure-la seria perdre-la sense dir-ho.
                r["ancora"] = True
            if "error" in r:
                print(f"  {entitat:30} ERROR: {r['error'][:80]}")
            else:
                print(f"  {entitat:30} tasa_error={r['tasa_error']}  {r['tasa_error_per_condicio']}")

        desar()  # despres de cada lot, no de cada entitat: si es talla, nomes es reprocessa el lot en curs

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

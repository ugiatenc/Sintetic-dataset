#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compara l'aranes REAL amb dues versions sintetiques i prepara una carpeta per escoltar-les."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ROOT / "src"))

import respelling

MANIFEST = ROOT / "datasets/aranes/test/manifest.jsonl"
CLIPS = ROOT / "datasets/aranes/test"
DIR_TTS = ROOT / "lab/proves_inicials/aranes/audios/validacio_tts"
DIR_ESCOLTA = ROOT / "lab/proves_inicials/aranes/audios/escolta_hibrid"
FRASES = AQUI / "frases_escolta.txt"
CORPUS = AQUI / "corpus_aranes.txt"
PLURALS = ROOT / "data/aranes/entitats/plurals_aranes.json"
RESULTATS = AQUI / "resultats_escolta_hibrid.json"
CACHE = AQUI / "cache_llm"
TTS_MODEL_ID = "k2-fsa/OmniVoice"
IDIOMA_TTS = "ca"
LLAVOR = 1234


def _carrega(nom, fitxer):
    """Importa un modul pel seu fitxer."""
    spec = importlib.util.spec_from_file_location(nom, AQUI / fitxer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w03 = _carrega("whisper03", "03_provar_whisper.py")
v04 = _carrega("validar04", "04_validar_tts.py")

PROMPT_PLURALS = """Ets un expert en gramatica de l'aranes (occita gascó de la Val d'Aran).

Et dono mots aranesos acabats en -ns. Per a cadascun, digues si la -s final es una MARCA
DE PLURAL (un nom o adjectiu en plural, com `camins` = camins, `vesins` = veïns) o no ho
es (adverbis, preposicions i altres mots que simplement acaben en -ns, com `mens` = menys,
`abans` = abans, `sens` = sense).

Respon NOMES un objecte JSON, sense text al voltant:
{{"mot": true, "mot": false, ...}}
true = es plural. false = no ho es. Inclou TOTS els mots de la llista.

Mots:
{llista}"""


def claude(prompt: str, model: str) -> str:
    """Crida al CLI de Claude Code (mode headless), amb cache a disc."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{model}_{hashlib.sha256((model + prompt).encode()).hexdigest()[:16]}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["result"]
    r = subprocess.run(["claude", "-p", prompt, "--output-format", "json", "--model", model],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"claude ha fallat: {r.stderr[:300]}")
    d = json.loads(r.stdout)
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d["result"]


def json_de_resposta(text: str) -> dict:
    """Extreu el JSON de la resposta del model, tolerant blocs de codi."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        raise ValueError(f"resposta sense JSON: {text[:200]}")
    return json.loads(text[i:j + 1])


def classificar_plurals(textos, model: str, refer: bool = False) -> dict:
    """{mot: es_plural} per als mots acabats en -ns."""
    desat = json.loads(PLURALS.read_text(encoding="utf-8")) if PLURALS.exists() else {}
    decisions = {} if refer else {k: v for k, v in desat.items() if not k.startswith("_")}
    candidats = respelling.candidats_plural(textos)
    nous = [m for m in candidats if m not in decisions]
    print(f"mots acabats en -ns: {len(candidats)} | ja decidits: {len(candidats) - len(nous)} | "
          f"a preguntar: {len(nous)}")
    if nous:
        d = json_de_resposta(claude(PROMPT_PLURALS.format(llista="\n".join(nous)), model))
        falten = [m for m in nous if m not in d]
        if falten:
            print(f"  avís: el model no ha tornat {len(falten)} mots: {falten[:5]}")
        decisions.update({m: bool(v) for m, v in d.items() if m in set(nous)})
        plurals = sorted(m for m, v in decisions.items() if v)
        PLURALS.parent.mkdir(parents=True, exist_ok=True)
        PLURALS.write_text(json.dumps(
            {"_meta": {"descripcio": "Mots aranesos acabats en -ns: true = la -s es marca de "
                                     "plural i la -n emmudeix (camins -> camís). Decidit un cop "
                                     "per mot amb un LLM; el codi nomes aplica la decisio.",
                       "model": model, "plurals": len(plurals), "no_plurals": len(decisions) - len(plurals)},
             **dict(sorted(decisions.items()))}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  desat a {PLURALS}")
    return decisions


def ruta_variant(frase: dict, i: int, veus: list[dict], variant: str) -> Path:
    """Ruta del wav d'una variant per a una frase."""
    clip = Path(frase["audio"]).stem.removeprefix("common_voice_oc_")
    if variant == "cru":
        return v04.ruta_sintetic(frase, i, veus, IDIOMA_TTS)
    return DIR_TTS / variant / f"cv{clip}_{variant}_{veus[i % len(veus)]['id']}.wav"


def sintetitzar(items, veus, variant, dispositiu, mida_lot):
    """items: [(i_global, frase, text_a_llegir)]."""
    import numpy as np, soundfile as sf, torch
    pendents = [x for x in items if not ruta_variant(x[1], x[0], veus, variant).exists()]
    print(f"[TTS {variant}] {len(items) - len(pendents)} ja fets, {len(pendents)} per generar")
    if not pendents:
        return
    (DIR_TTS / variant).mkdir(parents=True, exist_ok=True)
    from omnivoice import OmniVoice
    tts = OmniVoice.from_pretrained(TTS_MODEL_ID, device_map=dispositiu, dtype=torch.float16)
    per_veu = collections.defaultdict(list)
    for x in pendents:
        per_veu[x[0] % len(veus)].append(x)
    fets = 0
    for idx_veu, llista in per_veu.items():
        veu = veus[idx_veu]
        for inici in range(0, len(llista), mida_lot):
            lot = llista[inici:inici + mida_lot]
            torch.manual_seed(LLAVOR + idx_veu + inici)
            ones = tts.generate(text=[t for _, _, t in lot], language=IDIOMA_TTS,
                                ref_audio=[veu["ref_audio"]] * len(lot),
                                ref_text=[veu["ref_text"]] * len(lot))
            for (i, f, _), ona in zip(lot, ones):
                sf.write(ruta_variant(f, i, veus, variant),
                         np.asarray(ona, dtype="float32"), tts.sampling_rate, subtype="PCM_16")
            fets += len(lot)
            print(f"  {fets}/{len(pendents)}")
    del tts
    torch.cuda.empty_cache()


def detectar_idioma(rutes, dispositiu):
    """P(catala) i idioma mes probable segons Whisper, sense transcriure."""
    import librosa, torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(w03.MODEL_DEFECTE)
    model = WhisperForConditionalGeneration.from_pretrained(
        w03.MODEL_DEFECTE, dtype=torch.float16).to(dispositiu).eval()
    tok = proc.tokenizer
    ids = sorted({v for k, v in tok.get_vocab().items()
                  if k.startswith("<|") and k.endswith("|>") and k[2:-2].isalpha()
                  and k[2:-2].islower() and 2 <= len(k[2:-2]) <= 3})
    noms = [tok.convert_ids_to_tokens(v)[2:-2] for v in ids]
    sot = model.config.decoder_start_token_id
    top, p_ca = collections.Counter(), 0.0
    for i in range(0, len(rutes), 16):
        ones = [librosa.load(r, sr=w03.SR_ASR, mono=True)[0] for r in rutes[i:i + 16]]
        f = proc(ones, sampling_rate=w03.SR_ASR, return_tensors="pt").input_features.to(dispositiu, torch.float16)
        with torch.no_grad():
            lg = model(input_features=f,
                       decoder_input_ids=torch.full((len(ones), 1), sot, device=dispositiu)).logits[:, -1, :].float()
        probs = lg[:, ids].softmax(-1)
        for row in probs:
            top[noms[int(row.argmax())]] += 1
            p_ca += float(row[noms.index("ca")])
    del model
    torch.cuda.empty_cache()
    return {"top": dict(top.most_common(4)), "p_catala": round(p_ca / len(rutes), 3)}


CATEGORIES_ESCOLTA = {
    "nh": lambda m: "nh" in m,
    "lh": lambda m: "lh" in m,
    "sh": lambda m: "sh" in m,
    "o_a_u": lambda m: ("o" in m or "ó" in m) and "ò" not in m,
    "ns_plural": lambda m: m.endswith("ns"),
}


def tria_escolta(items, n):
    """Frases amb mes fonetica problematica primer, despres la resta fins a `n`."""
    puntuades = []
    for idx, (i, m, tts_text) in enumerate(items):
        ms = respelling.mots(m["text"])
        cats = {c for c, prova in CATEGORIES_ESCOLTA.items() if any(prova(w) for w in ms)}
        puntuades.append((len(cats), cats, idx))
    ordenades = sorted(puntuades, key=lambda x: -x[0])

    triats, vistes = [], set()
    for _, cats, idx in ordenades:
        if cats - vistes:
            triats.append(idx)
            vistes |= cats
    for _, cats, idx in ordenades:
        if len(triats) >= n:
            break
        if idx not in triats:
            triats.append(idx)
    return sorted(triats[:n])


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frases", type=Path, default=FRASES)
    p.add_argument("--totes", action="store_true",
                   help="fa servir les 115 frases del split test, no nomes frases_escolta.txt "
                        "(numeros fiables; frases_escolta.txt nomes en dona 12, massa poc "
                        "per treure'n cap conclusio, com ja va passar-hi)")
    p.add_argument("--escolta", type=int, default=12,
                   help="amb --totes, quantes frases copiar a la carpeta d'escolta (defecte: 12)")
    p.add_argument("--model", default="sonnet", help="model per classificar plurals (defecte: sonnet)")
    p.add_argument("--classificar-corpus", action="store_true",
                   help="refer el diccionari de plurals amb tot el corpus i sortir")
    p.add_argument("--nomes-textos", action="store_true", help="sense TTS ni Whisper")
    p.add_argument("--dispositiu", default="auto")
    p.add_argument("--lot-tts", type=int, default=4)
    p.add_argument("--mida-lot", type=int, default=8)
    args = p.parse_args()

    warnings.filterwarnings("ignore", message="(?s).*1080.*")

    tot = [json.loads(l)["text"] for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    if CORPUS.exists():
        tot += CORPUS.read_text(encoding="utf-8").splitlines()
    decisions = classificar_plurals(tot, args.model)
    plurals = {m for m, v in decisions.items() if v}
    print(f"plurals: {sorted(plurals)}")
    print(f"no plurals: {sorted(m for m, v in decisions.items() if not v)}\n")
    if args.classificar_corpus:
        return 0

    man = v04.carregar_manifest(MANIFEST)
    test = [m for m in man if m["split"] == "test"]
    per_text = {m["text"]: (i, m) for i, m in enumerate(test)}
    if args.totes:
        frases_txt = [m["text"] for m in test]
    else:
        frases_txt = [l for l in args.frases.read_text(encoding="utf-8").splitlines() if l.strip()]
        falten = [f for f in frases_txt if f not in per_text]
        if falten:
            raise SystemExit(f"{len(falten)} frases no son al split test: {falten[:2]}")

    items = []
    traca: dict = {}
    for f in frases_txt:
        i, m = per_text[f]
        items.append((i, m, respelling.reescriu(f, traca=traca, plurals=plurals)))
    print(f"{len(items)} frases | regles disparades: {dict(sorted(traca.items(), key=lambda kv: -kv[1]))}\n")
    for i, m, tts_text in items[:4]:
        print(f"  RAW     {m['text']}\n  HIBRID  {tts_text}\n")
    if args.nomes_textos:
        return 0

    from transformers.utils import logging as tlog
    tlog.set_verbosity_error(); tlog.disable_progress_bar()
    dispositiu = w03.triar_dispositiu(args.dispositiu)
    veus = v04.triar_veus(man, 4)

    sintetitzar(items, veus, "hibrid", dispositiu, args.lot_tts)

    import librosa, torch
    from transformers import pipeline
    print(f"\nCarregant {w03.MODEL_DEFECTE}...")
    asr = pipeline("automatic-speech-recognition", model=w03.MODEL_DEFECTE,
                   dtype=torch.float16 if dispositiu.startswith("cuda") else torch.float32,
                   device=dispositiu)
    refs = [m["text"] for _, m, _ in items]
    jocs = {"real": [m["audio"] for _, m, _ in items],
            "cru": [str(ruta_variant(m, i, veus, "cru")) for i, m, _ in items],
            "hibrid": [str(ruta_variant(m, i, veus, "hibrid")) for i, m, _ in items]}
    resultats = {}
    for nom, rutes in jocs.items():
        ones = [librosa.load(r, sr=w03.SR_ASR, mono=True)[0] for r in rutes]
        hip = w03.transcriure_lot(asr, ones, "oc", args.mida_lot)
        resultats[nom] = {"metriques": w03.metriques(refs, hip), "hipotesis": hip, "rutes": rutes}
        print(f"[ASR] {nom}: WER {resultats[nom]['metriques']['WER']:.3f} "
              f"CER {resultats[nom]['metriques']['CER']:.3f}")
    del asr
    torch.cuda.empty_cache()

    print("\n[Deteccio d'idioma]")
    for nom, joc in resultats.items():
        joc["idioma"] = detectar_idioma(joc["rutes"], dispositiu)
        print(f"  {nom:<7} P(catala)={joc['idioma']['p_catala']:.3f}  top={joc['idioma']['top']}")
    objectiu = resultats["real"]["idioma"]["p_catala"]
    print(f"\n  {'variant':<8} {'WER':>6} {'CER':>6} {'P(ca)':>7} {'|dif. amb el real|':>20}")
    for nom, joc in resultats.items():
        print(f"  {nom:<8} {joc['metriques']['WER']:>6.3f} {joc['metriques']['CER']:>6.3f} "
              f"{joc['idioma']['p_catala']:>7.3f} {abs(joc['idioma']['p_catala'] - objectiu):>20.3f}")

    idxs_escolta = tria_escolta(items, args.escolta) if len(items) > args.escolta else range(len(items))

    if DIR_ESCOLTA.exists():
        shutil.rmtree(DIR_ESCOLTA)
    DIR_ESCOLTA.mkdir(parents=True)
    files = []
    for n, idx in enumerate(idxs_escolta, 1):
        i, m, tts_text = items[idx]
        shutil.copy(m["audio"], DIR_ESCOLTA / f"f{n:02d}_0_real.wav")
        shutil.copy(ruta_variant(m, i, veus, "cru"), DIR_ESCOLTA / f"f{n:02d}_1_cru.wav")
        shutil.copy(ruta_variant(m, i, veus, "hibrid"), DIR_ESCOLTA / f"f{n:02d}_2_hibrid.wav")
        files.append({"id": f"f{n:02d}", "aranes_real": m["text"], "text_hibrid": tts_text,
                      **{f"whisper_{k}": resultats[k]["hipotesis"][idx] for k in resultats}})
    with (DIR_ESCOLTA / "frases.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(files[0]) + ["quina_sona_millor", "comentari"])
        w.writeheader()
        w.writerows(files)

    taula = "".join(f"| {n} | {r['metriques']['WER']:.3f} | {r['metriques']['CER']:.3f} | "
                    f"{r['idioma']['p_catala']:.3f} | {abs(r['idioma']['p_catala'] - objectiu):.3f} |\n"
                    for n, r in resultats.items())
    n_metriques = len(items)
    n_escolta = len(idxs_escolta)
    ns_plural_actiu = traca.get("ns_plural", 0)
    text_plural = (
        f"La regla de la `-ns` de plural s'ha aplicat {ns_plural_actiu} vegades en aquest "
        f"conjunt (mots com `camins`->`camís`, `vesins`->`vesís`). Nomes es toquen els mots "
        f"que el classificador de plurals marca com a tals; `abans`, `mens` i similars es "
        f"queden intactes."
        if ns_plural_actiu else
        "La regla de la `-ns` de plural no s'ha activat en aquest conjunt: els unics mots "
        "acabats en -ns que hi surten (`abans`, `mens`...) **no** son plurals. Es justament "
        "el cas que una regla automatica s'equivocaria (els convertiria en `abàs`/`més`), "
        "i per aixo la decisio de quins mots son plurals la pren un classificador, no una "
        "regla de grafia."
    )
    avis_soroll = "" if n_metriques >= 100 else f"""
> ### Atencio: amb nomes {n_metriques} frases aquesta taula pot no ser fiable
>
> La deteccio d'idioma sobre pocs clips es sorollosa: el mateix audio real pot variar
> mes de 0.1 de P(catala) nomes canviant la mostra. Per a un numero fiable, torna a
> executar amb `--totes` (les 115 frases del split test)."""

    (DIR_ESCOLTA / "README.md").write_text(f"""# Escolta: el text reescrit sona mes aranes?

Metriques calculades sobre **{n_metriques} frases** del split test; d'aquestes,
**{n_escolta}** s'han copiat aqui per escoltar (les que concentren mes fonetica
problematica), amb 3 audios cadascuna:

| Fitxer | Que es |
|---|---|
| `fNN_0_real.wav` | **Persona real** (Common Voice). La referencia: aixi ha de sonar |
| `fNN_1_cru.wav` | OmniVoice (`language="ca"`) llegint el text aranes TAL QUAL |
| `fNN_2_hibrid.wav` | OmniVoice llegint el text REESCRIT amb les normes del Conselh |

Les dues versions sintetiques fan servir la MATEIXA veu clonada per frase: l'unica
diferencia es el text que llegeix el TTS. Els textos son a `frases.csv`, amb dues columnes
buides (`quina_sona_millor`, `comentari`) per anotar-hi.

## Que s'ha reescrit

Regles aplicades sobre les {n_metriques} frases mesurades: {dict(sorted(traca.items(), key=lambda kv: -kv[1]))}

`nh`->`ny`, `lh`->`ll`, `sh`->`x`/`ix` i `o`/`ó`->`u`/`ú`. NO s'hi toca la `th`, la `h`, la
`-n` final ni la `u`: en catala ja sonen prou be o no hi ha manera de representar-ho.

{text_plural}

## Mesures automatiques
{avis_soroll}

| Versio | WER | CER | P(catala) | dif. amb el real |
|---|---|---|---|---|
{taula}
**P(catala)** es a quin punt Whisper diu que l'audio sona catala. L'objectiu es
acostar-se al valor del real ({objectiu:.3f}), no superar-lo.

**El WER/CER d'aquesta taula** compara sempre amb el text aranes original, encara que
`cru` i `hibrid` llegeixin textos diferents: mesura si el que diu cada versio, un cop
transcrit, s'assembla a l'ortografia aranesa real. No es una mesura de qualitat fonetica.
Qui ho decideix es l'orella.
""", encoding="utf-8")

    RESULTATS.write_text(json.dumps(
        {"frases": refs, "hibrid": [t for _, _, t in items], "traca": traca,
         "plurals_detectats": sorted(plurals),
         "resultats": {k: {kk: vv for kk, vv in r.items() if kk != "rutes"} for k, r in resultats.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCarpeta d'escolta: {DIR_ESCOLTA}")
    print(f"Resultats: {RESULTATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

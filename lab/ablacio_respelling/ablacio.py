#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablacio de les regles de reescriptura fonetica aranes -> grafia catalana."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AQUI))

import regles as R
import respelling

PROVES = ROOT / "lab/proves_inicials/aranes"
MANIFEST = ROOT / "datasets/aranes/test/manifest.jsonl"
PLURALS = ROOT / "data/aranes/entitats/plurals_aranes.json"
DIR_AUDIOS = AQUI / "audios"
RESULTATS = AQUI / "resultats_ablacio.json"
TTS_MODEL_ID = "k2-fsa/OmniVoice"
IDIOMA_TTS = "ca"
LLAVOR = 1234

REUTILITZATS = {
    "cru":   ROOT / "lab/proves_inicials/aranes/audios/validacio_tts/ca",
    "totes": ROOT / "lab/proves_inicials/aranes/audios/validacio_tts/hibrid",
}

VARIANTS = {
    "cru":              (frozenset(),                              "text aranes tal qual"),
    "totes":            (R.REGLES_ACTUALS,                         "les 5 regles d'avui"),
    "digrafs":          (frozenset({"nh", "lh", "sh"}),            "nomes nh/lh/sh"),
    "o_tot":            (frozenset({"o_tot"}),                     "nomes o->u, tota la paraula"),
    "o_tonica":         (frozenset({"o_tonica"}),                  "nomes o->u de la tonica"),
    "ng":               (frozenset({"ng"}),                        "nomes -n final -> -ng"),
    "ts":               (frozenset({"ts"}),                        "nomes -tz final -> -ts"),
    "proposta":         (frozenset({"o_tonica", "ng", "ts", "ns_plural"}),
                         "o tonica + ng + ts + plurals, SENSE digrafs"),
    "proposta_digrafs": (frozenset({"o_tonica", "ng", "ts", "ns_plural", "nh", "lh", "sh"}),
                         "la proposta amb els digrafs, per aillar-los"),
}


def _carrega(nom, fitxer):
    """Importa un modul de proves pel seu fitxer."""
    spec = importlib.util.spec_from_file_location(nom, PROVES / fitxer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w03 = _carrega("whisper03", "03_provar_whisper.py")
v04 = _carrega("validar04", "04_validar_tts.py")


def carregar_plurals() -> set[str]:
    """Plurals decidits (mots en -ns), sense les claus de metadades."""
    d = json.loads(PLURALS.read_text(encoding="utf-8"))
    return {k for k, v in d.items() if not k.startswith("_") and v}


def comprovar_equivalencia(frases, plurals) -> bool:
    """`regles.REGLES_ACTUALS` ha de donar el mateix text que `src/respelling.py`."""
    diff = [f for f in frases
            if respelling.reescriu(f, plurals=plurals)
            != R.reescriu(f, R.REGLES_ACTUALS, plurals=plurals)]
    print(f"[comprovacio] {len(frases) - len(diff)}/{len(frases)} frases identiques a respelling.py")
    for f in diff[:3]:
        print(f"    RAW         {f}")
        print(f"    respelling  {respelling.reescriu(f, plurals=plurals)}")
        print(f"    regles      {R.reescriu(f, R.REGLES_ACTUALS, plurals=plurals)}")
    return not diff


def ruta(variant: str, frase: dict, i: int, veus: list[dict]) -> Path:
    """Ruta del wav d'una variant per a una frase i la veu que li toca."""
    clip = Path(frase["audio"]).stem.removeprefix("common_voice_oc_")
    veu = veus[i % len(veus)]["id"]
    if variant == "cru":
        return REUTILITZATS["cru"] / f"cv{clip}_tts-ca_{veu}.wav"
    if variant == "totes":
        return REUTILITZATS["totes"] / f"cv{clip}_hibrid_{veu}.wav"
    return DIR_AUDIOS / variant / f"cv{clip}_{variant}_{veu}.wav"


def sintetitzar(variant, items, veus, dispositiu, mida_lot, tts=None):
    """items: [(i_global, frase, text_a_llegir)]."""
    import numpy as np, soundfile as sf, torch
    pendents = [x for x in items if not ruta(variant, x[1], x[0], veus).exists()]
    print(f"[TTS {variant}] {len(items) - len(pendents)} ja fets, {len(pendents)} per generar")
    if not pendents:
        return tts
    (DIR_AUDIOS / variant).mkdir(parents=True, exist_ok=True)
    if tts is None:
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
                sf.write(ruta(variant, f, i, veus), np.asarray(ona, dtype="float32"),
                         tts.sampling_rate, subtype="PCM_16")
            fets += len(lot)
            print(f"  {fets}/{len(pendents)}", end="\r", flush=True)
    print()
    return tts


class DetectorIdioma:
    """P(idioma) per clip amb Whisper, sense transcriure."""

    def __init__(self, dispositiu):
        """Carrega Whisper al dispositiu indicat."""
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch, self.dispositiu = torch, dispositiu
        self.proc = WhisperProcessor.from_pretrained(w03.MODEL_DEFECTE)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            w03.MODEL_DEFECTE, dtype=torch.float16).to(dispositiu).eval()
        tok = self.proc.tokenizer
        self.ids = sorted({v for k, v in tok.get_vocab().items()
                           if k.startswith("<|") and k.endswith("|>") and k[2:-2].isalpha()
                           and k[2:-2].islower() and 2 <= len(k[2:-2]) <= 3})
        self.noms = [tok.convert_ids_to_tokens(v)[2:-2] for v in self.ids]
        self.sot = self.model.config.decoder_start_token_id

    def __call__(self, rutes, mida_lot=16):
        """Transcriu una llista de wavs per lots."""
        import librosa
        torch = self.torch
        fora = []
        for i in range(0, len(rutes), mida_lot):
            ones = [librosa.load(r, sr=w03.SR_ASR, mono=True)[0] for r in rutes[i:i + mida_lot]]
            f = self.proc(ones, sampling_rate=w03.SR_ASR, return_tensors="pt"
                          ).input_features.to(self.dispositiu, torch.float16)
            with torch.no_grad():
                lg = self.model(input_features=f,
                                decoder_input_ids=torch.full((len(ones), 1), self.sot,
                                                             device=self.dispositiu)
                                ).logits[:, -1, :].float()
            fora += self.torch.softmax(lg[:, self.ids], -1).cpu().tolist()
        return fora

    def alliberar(self):
        """Descarrega el model de la GPU."""
        del self.model
        self.torch.cuda.empty_cache()


def bootstrap_aparellat(deltes, n=10000, llavor=0):
    """IC95% de la mitjana de les diferencies aparellades, i P(delta < 0)."""
    import random
    if not deltes:
        return {"n": 0}
    rnd = random.Random(llavor)
    mitjana = sum(deltes) / len(deltes)
    mostres = []
    for _ in range(n):
        m = [deltes[rnd.randrange(len(deltes))] for _ in range(len(deltes))]
        mostres.append(sum(m) / len(m))
    mostres.sort()
    return {"n": len(deltes),
            "mitjana": round(mitjana, 4),
            "ic95": [round(mostres[int(0.025 * n)], 4), round(mostres[int(0.975 * n)], 4)],
            "p_baixa": round(sum(1 for x in mostres if x < 0) / n, 3)}


def jensen_shannon(p, q):
    """Distancia JS entre dos perfils d'idioma (arrel de la divergencia, en base 2)."""
    import math
    m = [(a + b) / 2 for a, b in zip(p, q)]
    def kl(x, y):
        """Divergencia KL en bits entre dues distribucions."""
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0 and b > 0)
    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


def perfil_mitja(vectors):
    """Mitjana component a component d'una llista de vectors."""
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(len(vectors[0]))]


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprovar", action="store_true",
                   help="nomes comprova l'equivalencia amb respelling.py i surt")
    p.add_argument("--nomes-textos", action="store_true", help="sense TTS ni Whisper")
    p.add_argument("--variants", nargs="+", help="nomes aquestes variants")
    p.add_argument("--dispositiu", default="auto")
    p.add_argument("--lot-tts", type=int, default=4)
    p.add_argument("--mida-lot", type=int, default=8)
    args = p.parse_args()

    man = v04.carregar_manifest(MANIFEST)
    test = [m for m in man if m["split"] == "test"]
    frases = [m["text"] for m in test]
    plurals = carregar_plurals()

    ok = comprovar_equivalencia(frases, plurals)
    if args.comprovar:
        return 0 if ok else 1
    if not ok:
        raise SystemExit("El text de `totes` ja no coincideix amb respelling.py: no es "
                         "poden reaprofitar els audios `hibrid`.")

    noms = args.variants or list(VARIANTS)
    textos, traces, tocades = {}, {}, {}
    for nom in noms:
        rr, _ = VARIANTS[nom]
        tr = {}
        textos[nom] = [R.reescriu(f, rr, traca=tr, plurals=plurals) for f in frases]
        traces[nom] = tr
        tocades[nom] = [i for i, (a, b) in enumerate(zip(frases, textos[nom])) if a != b]

    print(f"\n{'variant':<18} {'frases tocades':>15}  regles disparades")
    for nom in noms:
        print(f"{nom:<18} {len(tocades[nom]):>9}/{len(frases)}  "
              f"{dict(sorted(traces[nom].items(), key=lambda kv: -kv[1]))}")
    if args.nomes_textos:
        for nom in noms:
            if nom == "cru":
                continue
            print(f"\n=== {nom} — {VARIANTS[nom][1]}")
            for i in tocades[nom][:3]:
                print(f"   RAW {frases[i]}\n   NOU {textos[nom][i]}")
        return 0

    from transformers.utils import logging as tlog
    tlog.set_verbosity_error(); tlog.disable_progress_bar()
    dispositiu = w03.triar_dispositiu(args.dispositiu)
    veus = v04.triar_veus(man, 4)
    print(f"\nveus: {[v['id'] for v in veus]}")

    import torch
    tts = None
    for nom in noms:
        items = [(i, m, textos[nom][i]) for i, m in enumerate(test)]
        tts = sintetitzar(nom, items, veus, dispositiu, args.lot_tts, tts)
    if tts is not None:
        del tts
        torch.cuda.empty_cache()

    jocs = {nom: [str(ruta(nom, m, i, veus)) for i, m in enumerate(test)] for nom in noms}
    jocs["real"] = [m["audio"] for m in test]
    ids_veus = {v["client_id"] for v in veus}
    clips_veus = [m for m in man if m["client_id"] in ids_veus]
    print(f"\nreferencia dels 4 locutors clonats: {len(clips_veus)} clips reals")

    print("\n[Deteccio d'idioma]")
    det = DetectorIdioma(dispositiu)
    posteriors = {nom: det(rutes) for nom, rutes in jocs.items()}
    posteriors["real_4_veus"] = det([m["audio"] for m in clips_veus])
    det.alliberar()
    i_ca = det.noms.index("ca")
    p_ca = {nom: [v[i_ca] for v in vs] for nom, vs in posteriors.items()}
    for nom, vs in p_ca.items():
        print(f"  {nom:<18} P(ca)={sum(vs) / len(vs):.3f}  n={len(vs)}")

    print("\n[ASR]")
    import librosa
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=w03.MODEL_DEFECTE,
                   dtype=torch.float16 if dispositiu.startswith("cuda") else torch.float32,
                   device=dispositiu)
    metriques, hipotesis = {}, {}
    for nom, rutes in jocs.items():
        ones = [librosa.load(r, sr=w03.SR_ASR, mono=True)[0] for r in rutes]
        hip = w03.transcriure_lot(asr, ones, "oc", args.mida_lot)
        hipotesis[nom] = hip
        metriques[nom] = w03.metriques(frases, hip)
        print(f"  {nom:<18} WER {metriques[nom]['WER']:.3f}  CER {metriques[nom]['CER']:.3f}")
    del asr
    torch.cuda.empty_cache()

    perfil_real = perfil_mitja(posteriors["real_4_veus"])
    objectiu = sum(p_ca["real_4_veus"]) / len(p_ca["real_4_veus"])
    taula = {}
    for nom in noms:
        deltes_tot = [a - b for a, b in zip(p_ca[nom], p_ca["cru"])]
        deltes_toc = [deltes_tot[i] for i in tocades[nom]]
        mitjana = sum(p_ca[nom]) / len(p_ca[nom])
        taula[nom] = {
            "descripcio": VARIANTS[nom][1],
            "regles": sorted(VARIANTS[nom][0]),
            "frases_tocades": len(tocades[nom]),
            "edicions": dict(sorted(traces[nom].items(), key=lambda kv: -kv[1])),
            "p_catala": round(mitjana, 4),
            "dif_amb_real_4_veus": round(mitjana - objectiu, 4),
            "js_amb_real_4_veus": round(jensen_shannon(perfil_mitja(posteriors[nom]), perfil_real), 4),
            "delta_vs_cru_115": bootstrap_aparellat(deltes_tot),
            "delta_vs_cru_tocades": bootstrap_aparellat(deltes_toc),
            "WER": round(metriques[nom]["WER"], 5),
            "CER": round(metriques[nom]["CER"], 5),
        }

    ref = {}
    for nom in ("real", "real_4_veus"):
        m = sum(p_ca[nom]) / len(p_ca[nom])
        ref[nom] = {"p_catala": round(m, 4), "n": len(p_ca[nom]),
                    "js_amb_real_4_veus": round(
                        jensen_shannon(perfil_mitja(posteriors[nom]), perfil_real), 4)}
        if nom in metriques:
            ref[nom].update({"WER": round(metriques[nom]["WER"], 5),
                             "CER": round(metriques[nom]["CER"], 5)})

    RESULTATS.write_text(json.dumps({
        "objectiu_p_catala": round(objectiu, 4),
        "referencia": ref,
        "variants": taula,
        "frases": frases,
        "textos": textos,
        "p_catala_per_clip": {k: [round(x, 5) for x in v] for k, v in p_ca.items()},
        "idiomes": det.noms,
        "hipotesis": hipotesis,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'variant':<18}{'P(ca)':>7}{'dif.real':>9}{'JS':>7}{'Δ vs cru (IC95%)':>26}{'WER':>7}{'CER':>7}")
    for nom in sorted(noms, key=lambda n: taula[n]["p_catala"]):
        t, d = taula[nom], taula[nom]["delta_vs_cru_115"]
        ic = f"{d['mitjana']:+.3f} [{d['ic95'][0]:+.3f},{d['ic95'][1]:+.3f}]" if d.get("n") else "—"
        print(f"{nom:<18}{t['p_catala']:>7.3f}{t['dif_amb_real_4_veus']:>+9.3f}"
              f"{t['js_amb_real_4_veus']:>7.3f}{ic:>26}{t['WER']:>7.3f}{t['CER']:>7.3f}")
    print(f"\n  objectiu (real, 4 locutors clonats): P(ca) = {objectiu:.3f}")
    print(f"  real, 59 locutors del test:          P(ca) = {ref['real']['p_catala']:.3f}")
    print(f"\nResultats: {RESULTATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

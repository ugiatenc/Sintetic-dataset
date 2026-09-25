#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ng` sobre les regles d'avui: mesura sobre les 115 i carpeta per escoltar."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AQUI))

import ablacio as A
import regles as R
import respelling

DESTI = AQUI / "escolta/ng"
REGLES = R.REGLES_ACTUALS | {"ng"}


def g_de_mes(referencies, hipotesis):
    """Quantes `g` escriu Whisper que el text aranes no tenia."""
    import re
    n_frases = extra = 0
    exemples = []
    for ref, hip in zip(referencies, hipotesis):
        d = len(re.findall("g", hip.lower())) - len(re.findall("g", ref.lower()))
        if d > 0:
            n_frases += 1
            extra += d
            exemples.append({"aranes": ref, "whisper": hip})
    return {"frases": n_frases, "de": len(referencies), "g_extra": extra,
            "exemples": exemples[:6]}


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--escolta", type=int, default=12)
    p.add_argument("--dispositiu", default="auto")
    p.add_argument("--lot-tts", type=int, default=4)
    p.add_argument("--mida-lot", type=int, default=8)
    args = p.parse_args()

    man = A.v04.carregar_manifest(A.MANIFEST)
    test = [m for m in man if m["split"] == "test"]
    frases = [m["text"] for m in test]
    plurals = A.carregar_plurals()

    traca: dict = {}
    textos = [R.reescriu(f, REGLES, traca=traca, plurals=plurals) for f in frases]
    print(f"totes+ng | regles disparades: {dict(sorted(traca.items(), key=lambda kv: -kv[1]))}")

    from transformers.utils import logging as tlog
    tlog.set_verbosity_error(); tlog.disable_progress_bar()
    import torch
    dispositiu = A.w03.triar_dispositiu(args.dispositiu)
    veus = A.v04.triar_veus(man, 4)

    A.VARIANTS["totes_ng"] = (REGLES, "les regles d'avui + ng")
    tts = A.sintetitzar("totes_ng", [(i, m, textos[i]) for i, m in enumerate(test)],
                        veus, dispositiu, args.lot_tts, None)
    if tts is not None:
        del tts
        torch.cuda.empty_cache()

    jocs = {
        "cru":      [str(A.ruta("cru", m, i, veus)) for i, m in enumerate(test)],
        "totes":    [str(A.ruta("totes", m, i, veus)) for i, m in enumerate(test)],
        "ng":       [str(A.ruta("ng", m, i, veus)) for i, m in enumerate(test)],
        "totes_ng": [str(A.ruta("totes_ng", m, i, veus)) for i, m in enumerate(test)],
    }
    det = A.DetectorIdioma(dispositiu)
    post = {n: det(r) for n, r in jocs.items()}
    i_ca = det.noms.index("ca")
    det.alliberar()
    p_ca = {n: [v[i_ca] for v in vs] for n, vs in post.items()}

    import librosa
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=A.w03.MODEL_DEFECTE,
                   dtype=torch.float16 if dispositiu.startswith("cuda") else torch.float32,
                   device=dispositiu)
    met, hip = {}, {}
    for n, rutes in jocs.items():
        ones = [librosa.load(r, sr=A.w03.SR_ASR, mono=True)[0] for r in rutes]
        hip[n] = A.w03.transcriure_lot(asr, ones, "oc", args.mida_lot)
        met[n] = A.w03.metriques(frases, hip[n])
    del asr
    torch.cuda.empty_cache()

    def delta(a, b):
        """Diferencia aparellada entre dues variants, amb IC del 95 %."""
        return A.bootstrap_aparellat([x - y for x, y in zip(p_ca[a], p_ca[b])])

    def mostra(d):
        """Formata una diferencia amb el seu interval."""
        return f"{d['mitjana']:+.3f} [{d['ic95'][0]:+.3f},{d['ic95'][1]:+.3f}]" if d.get("n") else "—"

    print(f"\n{'joc':<10}{'P(ca)':>7}{'Δ vs cru':>24}{'Δ vs totes':>24}{'WER':>7}{'CER':>7}")
    for n in jocs:
        print(f"{n:<10}{sum(p_ca[n]) / len(p_ca[n]):>7.3f}"
              f"{mostra(delta(n, 'cru')):>24}{mostra(delta(n, 'totes')):>24}"
              f"{met[n]['WER']:>7.3f}{met[n]['CER']:>7.3f}"
              f"   g de mes: {g_de_mes(frases, hip[n])['frases']:>3}/{len(frases)}")

    import json
    (AQUI / "resultats_ng.json").write_text(json.dumps({
        "p_catala": {n: round(sum(v) / len(v), 4) for n, v in p_ca.items()},
        "metriques": {n: {"WER": round(m["WER"], 5), "CER": round(m["CER"], 5)}
                      for n, m in met.items()},
        "deltes": {f"{n}_vs_{b}": delta(n, b)
                   for b in ("cru", "totes") for n in jocs if n != b},
        "p_catala_per_clip": {n: [round(x, 5) for x in v] for n, v in p_ca.items()},
        "g_de_mes": {n: g_de_mes(frases, hip[n]) for n in jocs},
        "textos_totes_ng": textos, "hipotesis": hip,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    punt = sorted(range(len(frases)),
                  key=lambda i: -R.reescriu(frases[i], {"ng"}, traca={}).count("ng"))
    tria = sorted(punt[:args.escolta])

    if DESTI.exists():
        shutil.rmtree(DESTI)
    DESTI.mkdir(parents=True)
    files = []
    for n, i in enumerate(tria, 1):
        m = test[i]
        for etiqueta, joc in (("0_real", None), ("1_actual", "totes"), ("2_actual-ng", "totes_ng")):
            orig = Path(m["audio"]) if joc is None else Path(jocs[joc][i])
            shutil.copy(orig, DESTI / f"f{n:02d}_{etiqueta}.wav")
        files.append({"id": f"f{n:02d}", "aranes": m["text"],
                      "text_actual": respelling.reescriu(m["text"], plurals=plurals),
                      "text_actual_ng": textos[i],
                      "whisper_actual": hip["totes"][i], "whisper_actual_ng": hip["totes_ng"][i],
                      "s_hi_sent_una_g": "", "quin_sona_mes_aranes": "", "comentari": ""})
    with (DESTI / "frases.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(files[0]))
        w.writeheader(); w.writerows(files)

    g_base, g_ng = g_de_mes(frases, hip["totes"]), g_de_mes(frases, hip["totes_ng"])
    mostra_g = "\n".join(f"- ARANES `{e['aranes']}`\n  WHISPER `{e['whisper']}`"
                         for e in g_ng["exemples"][:4])
    (DESTI / "README.md").write_text(f"""# Escolta: la `-ng` final s'hi sent?

`ng` converteix la `-n` final en `-ng` per obtenir [ŋ]. Baixa P(catala) {sum(p_ca['totes_ng']) / len(p_ca['totes_ng']) - sum(p_ca['totes']) / len(p_ca['totes']):+.3f} sobre les
regles d'avui, pero l'interval creua el zero i, sobretot, **sembla que per un motiu dolent**.

## La [g] s'hi sent

Whisper escriu una `g` que el text aranes no porta en **{g_ng['frases']}/{g_ng['de']}** frases amb `ng`
({g_ng['g_extra']} `g` de mes), contra **{g_base['frases']}/{g_base['de']}** sense la regla. Si Whisper l'escriu, l'ha sentida:

{mostra_g}

Es exactament la por amb que `normes_ortografiques_aranes.json` va descartar la regla, i
el WER no ho detectava perque ja es de 0,7 i una [g] de mes s'hi perd. Si aixo es
confirma escoltant-ho, la baixada de P(catala) **no es que soni mes aranes**: es que soni
espatllat, que tambe allunya del catala.

Dues preguntes per a l'orella, i son diferents:

1. **S'hi sent una [g]?** (columna `s_hi_sent_una_g`)
2. **Quina sona mes aranesa?** (columna `quin_sona_mes_aranes`) Si no s'hi sent cap [g]
   pero tampoc cap diferencia, la regla tampoc no fa falta.

| Fitxer | Que es |
|---|---|
| `fNN_0_real.wav` | persona real (Common Voice) |
| `fNN_1_actual.wav` | el text que genera `respelling.py` avui |
| `fNN_2_actual-ng.wav` | el mateix, amb `-n` final -> `-ng` |

Mateixa veu i mateixa llavor a `_1` i `_2`: l'unic que canvia es la `-n` final.
""", encoding="utf-8")
    print(f"\nCarpeta d'escolta: {DESTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 10: auditoria del dataset generat (tts_text recalculat, cap frase ni locutor del test)."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import numpy as np

import config as C
import corpus

DIR_DATASET = C.DIR_DATASET


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    import importlib.util as u
    import phonetics
    sp = u.spec_from_file_location("r5", AQUI / "05_respelling.py"); m5 = u.module_from_spec(sp); sp.loader.exec_module(m5)
    m = [x for x in corpus.llegeix_jsonl(DIR_DATASET / "manifest.jsonl") if x.get("status") == "ok"]
    veus = corpus.llegeix_jsonl(C.DIR_AUDIO / "veus/manifest.jsonl")
    if not m:
        raise SystemExit("Cap audio al manifest.")
    fallades, linies = [], []

    frases_corpus = [r["raw_text"] for r in corpus.llegeix_jsonl(C.DIR_TEXT / "corpus.jsonl")]
    plurals, prot = m5.context_proteccio(frases_corpus)  # el mateix context (sigles, majuscules, noms) que el pas 5
    dif = [x["id"] for x in m if m5.reescriu_protegit(x["text"], plurals, prot, {}) != x["tts_text"]]
    linies.append(f"1. tts_text reproduible: {len(m) - len(dif):,}/{len(m):,} identics" + (f"  **FALLA** ({dif[:3]})" if dif else "  ok"))
    if dif: fallades.append("tts_text")

    frases_dev = C.frases_dev()
    dev_al_train = [x["id"] for x in m if x.get("split", "train") == "train" and corpus.clau_dedup(x["text"]) in frases_dev]
    linies.append(f"1b. frases del dev al train: {len(dev_al_train)}" + (f"  **FALLA** ({dev_al_train[:3]})" if dev_al_train else "  ok"))
    if dev_al_train: fallades.append("dev al train")
    cv = [json.loads(l) for l in C.MANIFEST_CV.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_txt = {corpus.clau_dedup(c["text"]) for c in cv if c["split"] == "test"}
    colades = [x["id"] for x in m if corpus.clau_dedup(x["text"]) in test_txt]
    linies.append(f"2. frases del test de Common Voice al dataset: {len(colades)}" + ("  **FALLA**" if colades else "  ok"))
    if colades: fallades.append("frases_test")

    test_spk = {c["client_id"] for c in cv if c["split"] == "test"}
    veus_test = [v["id"] for v in veus if v.get("client_id") in test_spk]
    linies.append(f"3. locutors del test al banc de veus: {len(veus_test)}" + ("  **FALLA**" if veus_test else "  ok"))
    if veus_test: fallades.append("locutors_test")

    import soundfile as sf
    llargs = [x["id"] for x in m if x["durada_s"] > 30]
    random.seed(0)
    mostra = random.sample(m, min(200, len(m)))
    mal = []
    for x in mostra:
        try:
            y, sr = sf.read(ARREL / x["audio"], dtype="float32")
            if sr != 16000 or not np.isfinite(y).all() or (np.abs(y) >= 0.9999).mean() > 0.001:
                mal.append(x["id"])
        except Exception:
            mal.append(x["id"])
    linies.append(f"4. audios > 30 s: {len(llargs)}; fitxers dolents (mostra de {len(mostra)}): {len(mal)}" + ("  **FALLA**" if llargs or mal else "  ok"))
    if llargs or mal: fallades.append("audio")

    import collections
    cv_ = collections.Counter(x["veu_id"] for x in m)
    mitj = len(m) / max(1, len(cv_))
    desq = [k for k, n in cv_.items() if n > 2 * mitj]
    ce = collections.Counter(x["entorn_id"] for x in m)
    linies.append(f"5. veus: {len(cv_)} (mitjana {mitj:.0f} clips/veu, desequilibrades: {len(desq)}); entorns: {len(ce)}" + ("  avis" if desq else "  ok"))

    h = sum(x["durada_s"] for x in m) / 3600
    (DIR_DATASET / "AUDITORIA.md").write_text(
        f"# Auditoria del dataset\n\n{len(m):,} clips · {h:.2f} h · {len(cv_)} veus · {len(ce)} entorns\n\n" +
        "\n".join(f"- {l}" for l in linies) +
        f"\n\n**Resultat: {'FALLA (' + ', '.join(fallades) + ')' if fallades else 'tot correcte'}**\n", encoding="utf-8")
    print("\n".join(linies)); print(f"\n-> {DIR_DATASET / 'AUDITORIA.md'}")
    return 1 if fallades else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`banc.jsonl` -> carpetes per grup i per locutor + `README.md` amb les taules."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
BANC = AQUI / "banc.jsonl"
GRUPS = {
    "aranes_declarat": ("01_aranes_declarat", "Occitans amb accent aranès declarat (no són al test)"),
    "no_declarat": ("02_no_declarat", "Occitans sense accent declarat (no són al test)"),
    "test": ("03_test_no_usable", "Locutors del TEST: només per escoltar, la seva veu NO pot entrar al train"),
    "catala": ("04_catala", "Catalans de Common Voice `ca`, un per accent"),
}


def _mou(origen: Path, desti: Path) -> None:
    """Mou un fitxer si existeix i el desti no."""
    if origen.exists() and not desti.exists():
        desti.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origen), str(desti))


def fmt(v, dec=2):
    """Formata un valor per a la taula."""
    return "—" if v is None else (f"{v:.{dec}f}" if isinstance(v, float) else str(v))


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    files = [json.loads(l) for l in BANC.open(encoding="utf-8") if l.strip()]
    for f in files:
        carpeta = AQUI / GRUPS[f["categoria"]][0] / f["id"]
        carpeta.mkdir(parents=True, exist_ok=True)
        _mou(AQUI / f["ref_audio"], carpeta / "referencia.wav")
        f["ref_audio"] = str((carpeta / "referencia.wav").relative_to(AQUI))
        (carpeta / "referencia.txt").write_text(f["ref_text"] + "\n", encoding="utf-8")
        for j, s in enumerate(f.get("sint", []), 1):
            _mou(AQUI / s["audio"], carpeta / f"sintetic_{j}.wav")
            s["audio"] = str((carpeta / f"sintetic_{j}.wav").relative_to(AQUI))
            (carpeta / f"sintetic_{j}.txt").write_text(f"RAW  {s['raw_text']}\nTTS  {s['tts_text']}\n", encoding="utf-8")
        splits = ", ".join(f"{k} {v}" for k, v in sorted(f["splits"].items()))
        linies = [f"# {f['id']}", "",
                  f"- Grup: **{GRUPS[f['categoria']][1]}**",
                  f"- Common Voice `{f['locale']}` · splits: {splits} · {f['n_clips_audio']} clips amb àudio",
                  f"- Accent declarat: **{f['accents']}** · gènere: {f['gender']} · edat: {f['age']}",
                  f"- Al test: {'SÍ (no usable per al train)' if f['al_test'] else 'no'} · al banc actual (pas 6): {'sí' if f['al_banc'] else 'no'}",
                  f"- Referència ({f.get('origen_ref', '')}, split {f.get('ref_split', '?')}): nota {fmt(f.get('nota'))} · "
                  f"SNR {fmt(f.get('snr_db'), 1)} dB · parla {fmt(f.get('ratio_parla'))} · {fmt(f.get('durada_s'), 1)} s"
                  + (f" · ⚠ {f['avis']}" if f.get("avis") else ""),
                  f"- Text de la referència: «{f['ref_text']}»", ""]
        for j, s in enumerate(f.get("sint", []), 1):
            linies += [f"## sintetic_{j}.wav ({s['durada_s']} s)", f"- RAW: {s['raw_text']}", f"- TTS: {s['tts_text']}"]
            if f.get(f"whisper_{j}") is not None:
                linies += [f"- Whisper entén: «{f[f'whisper_{j}']}» · CER {f[f'cer_{j}']:.3f}"]
            linies.append("")
        if f.get("error_tts"):
            linies.append(f"⚠ error TTS: {f['error_tts']}")
        (carpeta / "info.md").write_text("\n".join(linies) + "\n", encoding="utf-8")
    for d in (AQUI / "audio" / "ref", AQUI / "audio" / "sint", AQUI / "audio"):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    with BANC.open("w", encoding="utf-8") as fh:
        for f in files:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    parts = ["# Banc de proves de veus (21/09)", "",
             "Common Voice `oc` (tots els locutors amb àudio: `dev`, `train`, `other` i el test) i 5 veus de Common "
             "Voice `ca`, cadascuna amb la seva referència i dues frases d'aranès sintetitzades amb OmniVoice clonant-la. "
             "Cada locutor té una carpeta amb `referencia.wav` + `referencia.txt`, `sintetic_1.wav`/`sintetic_2.wav` "
             "(+ `.txt` amb el text RAW i el TTS) i un `info.md`. Es genera amb `construeix_banc.py` "
             "(referències i síntesi), `avalua_whisper.py` (CER) i `fes_readme.py` (carpetes i aquest fitxer).", "",
             "Les dues frases, iguals per a tothom:", ""]
    if files and files[0].get("sint"):
        for s in files[0]["sint"]:
            parts += [f"- RAW «{s['raw_text']}»", f"  TTS «{s['tts_text']}»"]
    parts += ["", "**CER** = distància de caràcters entre el `tts_text` i el que Whisper large-v3-turbo (català) entén del "
              "clip sintètic, sense majúscules ni puntuació. Whisper catalanitza el que sent (`Consell General`, `2025`, "
              "`10%`), així que aquest CER porta un sòl d'uns 0,29 que no és culpa de la veu; per això hi ha també el "
              "**CER consens**: distància entre el que Whisper entén d'aquesta veu i la lectura més freqüent entre totes "
              "les veus. Mesura només el que la veu hi afegeix: ⚠ per damunt de 0,25 la veu diu una altra cosa (soroll, "
              "veu que no clona bé). **Nota** = puntuació de qualitat de la referència del pas 6 (0-1: SNR, proporció de "
              "parla, transitoris, durada, constància).", ""]
    for cat, (carp, titol) in GRUPS.items():
        fs = sorted((f for f in files if f["categoria"] == cat), key=lambda f: -(f.get("nota") or 0))
        if not fs:
            continue
        cers = sorted(f["cer"] for f in fs if f.get("cer") is not None)
        parts += [f"## {titol}", "",
                  f"{len(fs)} locutors → carpeta `{carp}/`" + (f" · CER mediana **{cers[len(cers) // 2]:.3f}**" if cers else ""), "",
                  "| Locutor | Splits (clips) | Accent | Gènere · edat | Nota ref. | SNR | CER | CER consens | Text de la referència |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for f in fs:
            splits = ", ".join(f"{k} {v}" for k, v in sorted(f["splits"].items()))
            marca = " 🟩banc" if f["al_banc"] else ""
            cc = f.get("cer_consens")
            cc_s = "—" if cc is None else (f"**{cc:.3f}** ⚠" if cc > 0.25 else f"{cc:.3f}")
            parts.append(f"| [`{f['id']}`]({carp}/{f['id']}/){marca} | {splits} | {f['accents']} | {f['gender']} · {f['age']} | "
                         f"{fmt(f.get('nota'))} | {fmt(f.get('snr_db'), 0)} | {fmt(f.get('cer'), 3)} | {cc_s} | {f['ref_text']} |")
        parts.append("")
    conclusions = AQUI / "CONCLUSIONS.md"
    if conclusions.exists():
        parts += ["---", "", conclusions.read_text(encoding="utf-8")]
    (AQUI / "README.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"-> {AQUI / 'README.md'} ({len(files)} locutors en {len(GRUPS)} carpetes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

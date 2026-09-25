#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pas 8b: mou els clips ja generats (i els reals de Common Voice) a `dataset_aranes/<font>_<bloc>/` amb un json al costat de cada wav, i reescriu les rutes dels manifests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARREL = AQUI.parents[1]
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(AQUI))

import config as C

MANIFEST = C.DIR_DATASET / "manifest.jsonl"
DIR_VELL = C.DIR_DATASET / "audio"
DIR_CV_VELL = C.DIR_CV_REAL / "clips"
FRASES_DEV = C.frases_dev()


def algu_te_obert(ruta: Path) -> bool:
    """Si algun proces te el fitxer obert (el pas 8 en marxa): no s'ha de tocar el manifest."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}") == str(ruta):
                    return True
        except OSError:
            continue
    return False


def reubica(registres: list[dict], amb_index: bool) -> dict:
    """Mou cada wav a la seva carpeta, escriu el json i actualitza `audio`; torna recomptes."""
    n = {"moguts": 0, "ja_al_lloc": 0, "falten": 0, "json": 0}
    carpetes = set()
    for i, r in enumerate(registres):
        if r.get("status") != "ok":
            continue
        r["split"] = r.get("split") or C.split_clip(r["id"], r.get("client_id"))
        vell = ARREL / r["audio"]
        if not amb_index and r["split"] == "train" and C.split_clip(r["id"], text=r["text"], frases_dev=FRASES_DEV) == "exclos":
            vell.unlink(missing_ok=True)
            vell.with_suffix(".json").unlink(missing_ok=True)
            r["status"], r["split"] = "exclos", "exclos"
            n["exclosos"] = n.get("exclosos", 0) + 1
            continue
        nou = C.ruta_clip(r["id"], index=i if amb_index else None, split=r["split"])
        if nou.parent not in carpetes:
            nou.parent.mkdir(parents=True, exist_ok=True)
            carpetes.add(nou.parent)
        if vell.resolve() != nou.resolve() and vell.exists():
            os.replace(vell, nou)
            vell.with_suffix(".json").unlink(missing_ok=True)  # el json vell no s'ha de quedar orfe
            n["moguts"] += 1
        elif nou.exists():
            n["ja_al_lloc"] += 1
        else:
            n["falten"] += 1
            continue
        r["audio"] = str(nou.relative_to(ARREL))
        nou.with_suffix(".json").write_text(json.dumps({**r, "transcript": r["text"]}, ensure_ascii=False, indent=1), encoding="utf-8")
        n["json"] += 1
    return n


def reescriu(ruta: Path, registres: list[dict]) -> None:
    """Escriu el manifest sencer de manera atomica."""
    tmp = ruta.with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in registres:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, ruta)


def main() -> int:
    """Punt d'entrada: comprova que el pas 8 no corre, mou, reescriu i verifica."""
    if algu_te_obert(MANIFEST):
        raise SystemExit("El pas 8 te el manifest obert: atura la generacio abans.")
    for nom, ruta, amb_index in (("sintetics", MANIFEST, False), ("reals (cv)", C.MANIFEST_CV_REAL, True)):
        if not ruta.exists():
            continue
        regs = [json.loads(l) for l in ruta.open(encoding="utf-8") if l.strip()]
        n = reubica(regs, amb_index)
        reescriu(ruta, regs)
        print(f"{nom}: {n['moguts']:,} moguts, {n['ja_al_lloc']:,} ja al lloc, {n['json']:,} json, {n['falten']} sense fitxer, {n.get('exclosos', 0)} exclosos (frase del dev)")
    for d in (DIR_VELL, DIR_CV_VELL):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            print(f"carpeta buida esborrada: {d.relative_to(ARREL)}")
        elif d.exists():
            print(f"AVIS: {d.relative_to(ARREL)} encara te {sum(1 for _ in d.iterdir()):,} fitxers")
    for arrel in (C.DIR_CLIPS, C.DIR_CLIPS_EVAL):
        if arrel.exists():
            for d in [p for p in arrel.iterdir() if p.is_dir() and not any(p.iterdir())]:
                d.rmdir()
            carpetes = sorted(p for p in arrel.iterdir() if p.is_dir())
            fitxers = sum(1 for p in carpetes for _ in p.iterdir())
            print(f"{arrel.relative_to(ARREL)}: {len(carpetes)} carpetes, {fitxers:,} fitxers (wav + json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

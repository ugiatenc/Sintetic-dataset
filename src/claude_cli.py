#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crides a Claude via el CLI de Claude Code, amb cache per prompt."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import time
from pathlib import Path

ARREL = Path(__file__).resolve().parent.parent
CACHE_DEFECTE = ARREL / ".cache_claude"
TIMEOUT_S = 600


class ErrorClaude(RuntimeError):
    """Error en una crida al CLI de Claude."""
    pass


def _ruta_cache(cache: Path, model: str, prompt: str) -> Path:
    """Ruta del fitxer de cache d'un prompt (hash de model + prompt)."""
    clau = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()[:20]
    return cache / f"{model}_{clau}.json"


def crida(prompt: str, model: str = "sonnet", cache: Path | None = None,
          max_reintents: int = 4, timeout: int = TIMEOUT_S) -> str:
    """Text de resposta de Claude."""
    cache = cache or CACHE_DEFECTE
    cache.mkdir(parents=True, exist_ok=True)
    fitxer = _ruta_cache(cache, model, prompt)
    if fitxer.exists():
        return json.loads(fitxer.read_text(encoding="utf-8"))["result"]

    ultim: Exception | None = None
    for intent in range(1, max_reintents + 1):
        try:
            r = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "json", "--model", model],
                capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0:
                raise ErrorClaude(f"claude ha sortit amb {r.returncode}: {r.stderr[:300]}")
            d = json.loads(r.stdout)
            if "result" not in d:
                raise ErrorClaude(f"resposta sense 'result': {r.stdout[:200]}")
            fitxer.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            return d["result"]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ErrorClaude) as e:
            ultim = e
            if intent == max_reintents:
                break
            espera = 2 ** (intent - 1) + random.uniform(0, 1)
            print(f"    avis: {type(e).__name__} (intent {intent}/{max_reintents}); "
                  f"reintent en {espera:.1f} s")
            time.sleep(espera)
    raise ErrorClaude(f"Claude ha fallat {max_reintents} vegades: {ultim}")


def json_de_resposta(text: str) -> dict:
    """El primer objecte JSON de la resposta, amb o sense tanques de codi."""
    net = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = net.find("{"), net.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"resposta sense JSON: {text[:200]}")
    return json.loads(net[i:j + 1])


def llista(prompt: str, clau: str, model: str = "sonnet", cache: Path | None = None,
           minim: int = 1, **kw) -> list[str]:
    """Crida que ha de tornar `{clau: [str, ...]}`, validada."""
    d = json_de_resposta(crida(prompt, model=model, cache=cache, **kw))
    if clau not in d or not isinstance(d[clau], list):
        raise ValueError(f"la resposta no te la llista '{clau}': {list(d)[:5]}")
    items = [str(x).strip() for x in d[clau] if str(x).strip()]
    if len(items) < minim:
        raise ValueError(f"nomes {len(items)} items, se'n demanaven {minim} com a minim")
    return items

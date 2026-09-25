#!/usr/bin/env python3
"""Estudi acustic: com pronuncien els locutors reals de Common Voice la lletra `o`?"""
import collections
import json
import re
import sys
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

try:
    import parselmouth
except ImportError:  # pragma: no cover
    parselmouth = None

ARREL = Path("/media/ugiat/dd2/projects/nerea/sintetic_dataset/Sintetic-dataset")
S = Path(__file__).resolve().parent
MANIFEST = ARREL / "datasets/aranes/test/manifest.jsonl"
PH_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SR = 16000
LLETRES = set("oóòuúa")

VOCALS = set("aeiouàáèéíïòóúü")
ACCENTUADES = set("àáèéíòóú")
FUNCIONALS = {"non", "tot", "tota", "toti", "totes", "coma", "vos", "jo", "lo", "on", "mos",
              "o", "son", "sons", "mon", "donc", "contra", "tostemp", "boni", "pro", "nosati",
              "vosati", "vòsti", "vòste", "vòsta", "com", "encoèra", "tanpòc", "quauquarren",
              "sonque", "sol", "solet", "solets", "dempús", "alavetz", "donques"}

U_LIKE = {"u", "ʊ", "ɯ", "ʉ", "w"}
O_LIKE = {"o", "ɔ", "ɤ", "õ", "ɵ"}
Y_LIKE = {"y", "ʏ", "ø", "œ", "ɥ"}
A_LIKE = {"a", "ɐ", "ɑ", "æ", "ɶ"}
E_LIKE = {"e", "ɛ", "ə", "ɜ", "ɘ"}
I_LIKE = {"i", "ɪ", "j"}


def base_fon(p: str) -> str:
    """Fonema espeak sense llargada ni diacritics: 'oː' -> 'o', 'ɔ̃' -> 'ɔ'."""
    p = p.replace("ː", "").replace("ˑ", "")
    return "".join(c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c))


def classe_fon(p: str) -> str:
    """Classe vocalica (U, O, Y, A, E) d'un fonema."""
    b = base_fon(p)
    for nom, s in (("U", U_LIKE), ("O", O_LIKE), ("Y", Y_LIKE), ("A", A_LIKE), ("E", E_LIKE), ("I", I_LIKE)):
        if b in s or (b and b[0] in s):
            return nom
    return "cons" if b else "-"


def rom_char(ch: str):
    """Una lletra -> [a-z'] per a MMS_FA, o None si no es lletra."""
    ch = ch.lower()
    if ch in "'’‘":
        return "'"
    base = unicodedata.normalize("NFD", ch)[0]
    if base == "ç":
        base = "c"
    return base if "a" <= base <= "z" else None


def grups_vocalics(mot: str) -> list[tuple[int, int]]:
    """Posicions dels grups de vocals d'un mot."""
    grups, i = [], 0
    while i < len(mot):
        if mot[i] in VOCALS:
            j = i
            while j < len(mot) and mot[j] in VOCALS:
                j += 1
            grups.append((i, j)); i = j
        else:
            i += 1
    return grups


def grup_tonic(mot: str) -> int | None:
    """Index del grup vocalic tonic, estimat amb la regla occitana: accent grafic si n'hi ha; si no."""
    g = grups_vocalics(mot)
    if not g:
        return None
    for k, (a, b) in enumerate(g):
        if any(c in ACCENTUADES for c in mot[a:b]):
            return k
    if len(g) == 1:
        return 0
    if mot[-1] in VOCALS or (mot[-1] == "s" and mot[-2] in VOCALS):
        return len(g) - 2
    return len(g) - 1


def carrega_audio(p: Path) -> np.ndarray:
    """Llegeix un wav a mono 16 kHz."""
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        d = gcd(sr, SR); x = resample_poly(x, SR // d, sr // d).astype(np.float32)
    return x


def tria_gpu() -> torch.device:
    """La RTX 4060 Ti si hi es; si no, la primera GPU o la CPU."""
    for i in range(torch.cuda.device_count()):
        if "4060" in torch.cuda.get_device_name(i):
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    dev = tria_gpu()
    print("dispositiu:", dev, torch.cuda.get_device_name(dev) if dev.type == "cuda" else "")
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
    rows = rows[:limit]

    bundle = torchaudio.pipelines.MMS_FA
    fa_model = bundle.get_model(with_star=False).to(dev).eval()
    fa_tok = bundle.get_tokenizer(); fa_al = bundle.get_aligner()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(PH_MODEL)
    vocab = json.loads(Path(hf_hub_download(PH_MODEL, "vocab.json")).read_text(encoding="utf-8"))
    id2fon = {i: t for t, i in vocab.items()}
    ph_model = Wav2Vec2ForCTC.from_pretrained(PH_MODEL).to(dev).eval()
    blank = vocab.get("<pad>", 0)
    print("models carregats")

    out = (S / "estudi_o_u.jsonl").open("w", encoding="utf-8")
    n_fet = n_err = 0
    for k, r in enumerate(rows):
        try:
            x = carrega_audio(ARREL / "datasets/aranes/test" / r["audio"])
            wave = torch.from_numpy(x)[None]
            paraules, roms, mapes = [], [], []
            for w in r["text"].split():
                w_net = w.strip("¡!¿?.,;:«»\"“”()[]—–-…")
                rom, mapa = [], []
                for idx, ch in enumerate(w_net):
                    rc = rom_char(ch)
                    if rc:
                        rom.append(rc); mapa.append((idx, ch))
                if rom:
                    paraules.append(w_net); roms.append("".join(rom)); mapes.append(mapa)
            if not roms:
                continue
            with torch.inference_mode():
                emis, _ = fa_model(wave.to(dev))
            spans = fa_al(emis[0], fa_tok(roms))
            fa_ratio = wave.size(1) / emis.size(1) / SR
            with torch.inference_mode():
                inp = extractor(x, sampling_rate=SR, return_tensors="pt")
                logits = ph_model(inp.input_values.to(dev)).logits[0]
            ids = logits.argmax(-1).cpu().numpy()
            ph_ratio = wave.size(1) / len(ids) / SR
            fons = []
            i = 0
            while i < len(ids):
                j = i
                while j < len(ids) and ids[j] == ids[i]:
                    j += 1
                if ids[i] != blank:
                    fons.append((id2fon.get(int(ids[i]), "?"), i * ph_ratio, j * ph_ratio))
                i = j
            snd = parselmouth.Sound(x.astype(np.float64), SR) if parselmouth else None
            form = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                                       maximum_formant=5500.0) if snd else None
            for wi, (w, mapa, wspans) in enumerate(zip(paraules, mapes, spans)):
                wl = w.lower()
                gt = grup_tonic(wl); grups = grups_vocalics(wl)
                for (idx, ch), sp in zip(mapa, wspans):
                    chl = ch.lower()
                    if chl not in LLETRES:
                        continue
                    t0, t1 = sp.start * fa_ratio, sp.end * fa_ratio
                    def solap(f, m=0.0):
                        """Solapament temporal entre un fonema i el tram del mot."""
                        return max(0.0, min(f[2], t1 + m) - max(f[1], t0 - m))
                    cands = [(solap(f), f) for f in fons if solap(f) > 0]
                    if not cands:
                        cands = [(solap(f, 0.03), f) for f in fons if solap(f, 0.03) > 0]
                    voc = [c for c in cands if classe_fon(c[1][0]) not in ("cons", "-")]
                    tria = max(voc or cands, key=lambda c: c[0])[1] if cands else None
                    fon = tria[0] if tria else "-"
                    f1 = f2 = None
                    a, b = (tria[1], tria[2]) if tria else (t0, t1)
                    if form is not None and b - a >= 0.02:
                        ts = np.linspace(a + 0.25 * (b - a), b - 0.25 * (b - a), 5)
                        v1 = [form.get_value_at_time(1, t) for t in ts]
                        v2 = [form.get_value_at_time(2, t) for t in ts]
                        v1 = [v for v in v1 if v and not np.isnan(v)]; v2 = [v for v in v2 if v and not np.isnan(v)]
                        f1 = float(np.median(v1)) if v1 else None; f2 = float(np.median(v2)) if v2 else None
                    gk = next((n for n, (a, b) in enumerate(grups) if a <= idx < b), None)
                    tonica = (gk == gt) if gk is not None and gt is not None else None
                    diftong = (idx > 0 and wl[idx - 1] in VOCALS) or (idx + 1 < len(wl) and wl[idx + 1] in VOCALS)
                    propi = w[:1].isupper() and wi > 0
                    out.write(json.dumps({
                        "clip": r["audio"], "loc": r["client_id"][:8], "mot": w, "lletra": chl, "idx": idx,
                        "fon": fon, "classe": classe_fon(fon), "f1": f1, "f2": f2,
                        "t0": round(t0, 3), "t1": round(t1, 3), "score": round(float(sp.score), 3),
                        "fon_dur": round(b - a, 3),
                        "tonica": tonica, "diftong": diftong, "propi": propi,
                        "funcional": wl in FUNCIONALS, "final": idx == len(wl) - 1, "inicial": idx == 0,
                    }, ensure_ascii=False) + "\n")
            n_fet += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"  [err] {r['audio']}: {type(e).__name__}: {str(e)[:120]}")
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(rows)} clips")
    out.close()
    print(f"fet: {n_fet} clips, {n_err} errors -> {S / 'estudi_o_u.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

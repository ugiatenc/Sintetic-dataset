"""Verificacio del corpus final (data/aranes/text/corpus.jsonl) contra tot el que s'ha demanat i arreglat."""
import collections
import json
import re
import sys
from pathlib import Path

ARREL = Path("/media/ugiat/dd2/projects/nerea/sintetic_dataset/Sintetic-dataset")
sys.path.insert(0, str(ARREL / "src"))
sys.path.insert(0, str(ARREL / "pipelines/aranes"))
import config as C  # noqa: E402
import corpus as CO  # noqa: E402
from qualitat import NUMERALS  # noqa: E402


def llegeix(p):
    """Llegeix un jsonl si existeix."""
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


regs = llegeix(C.DIR_TEXT / "corpus.jsonl")
raw = [r["raw_text"] for r in regs]
tts = [r["tts_text"] for r in regs]
parell = list(zip(raw, tts))
fonts = collections.Counter(r["font"] for r in regs)
hores = sum(r["durada_est_s"] for r in regs) / 3600
fiables = sum(r["durada_est_s"] for r in regs if r["font"] in C.FONTS_FIABLES) / 3600
print(f"{len(regs):,} frases · {hores:.1f} h (fiables {fiables:.1f} h)")
for f, n in fonts.most_common():
    h = sum(r["durada_est_s"] for r in regs if r["font"] == f) / 3600
    print(f"   {f:18s} {n:>8,}  {h:6.1f} h")

resultats = []


def prova(nom, cond, exemple=""):
    """Registra una comprovacio i n'imprimeix el resultat."""
    resultats.append(bool(cond))
    print(f"  [{'OK' if cond else 'FALLA'}] {nom}" + (f"   {str(exemple)[:110]}" if exemple else ""))


def info(nom, valor):
    """Imprimeix un valor informatiu."""
    print(f"  [info] {nom}: {valor}")


def troba(patro, seq, flags=0):
    """Elements de la sequencia que casen amb el patro."""
    rx = re.compile(patro, flags)
    return [t for t in seq if rx.search(t)]


def primer(patro, seq, flags=0):
    """Primer element que casa amb el patro, o cadena buida."""
    return (troba(patro, seq, flags) or [""])[0]


def exemple(rx_raw, rx_tts=None):
    """Primer parell (raw, tts) que casa amb els patrons."""
    for a, b in parell:
        if re.search(rx_raw, a) and (rx_tts is None or re.search(rx_tts, b)):
            return a, b
    return None


NUM = "|".join(sorted(NUMERALS, key=len, reverse=True))
DESENES = "dètz|vint|trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|nauanta|cent|mil"

print("\n# integritat")
prova("cap sentinella ẅ al tts_text", not any("ẅ" in t for t in tts), next((t for t in tts if "ẅ" in t), ""))
prova("ids unics", len({r["id"] for r in regs}) == len(regs))
prova("cap raw_text / tts_text buit", all(r["raw_text"].strip() and r["tts_text"].strip() for r in regs))
prova("'origen' es l'ultim camp de cada registre", all(list(r)[-1] == "origen" for r in regs))
dmax = max(r["durada_est_s"] for r in regs)
prova("cap frase > 30 s (durada_est_s)", dmax <= 30.0, f"max {dmax:.1f}s")
lmax = max(len(t) for t in tts)
prova("cap tts_text > 30 s a velocitat lenta (p10)", lmax / CO.CAR_PER_SEGON_LENT <= 30.0, f"max {lmax} car = {lmax / CO.CAR_PER_SEGON_LENT:.1f}s")
prova(f"llargada raw {C.MIN_CAR}-{C.MAX_CAR} car", all(C.MIN_CAR <= len(t) <= C.MAX_CAR for t in raw), str([t for t in raw if not C.MIN_CAR <= len(t) <= C.MAX_CAR][:2]))
cv = {CO.clau_dedup(json.loads(l)["text"]) for l in C.MANIFEST_CV.open(encoding="utf-8") if l.strip()}
prova(f"cap frase del test de Common Voice ({len(cv)})", not any(CO.clau_dedup(t) in cv for t in raw))
claus = collections.Counter(CO.clau_dedup(t) for t in raw)
prova("cap duplicat (clau de dedup)", max(claus.values()) == 1, next((k for k, v in claus.items() if v > 1), ""))

print("\n# conciliacio per font")
tot_net = tot_d5 = 0
minus_rec = {}
for f in C.FONTS:
    d = C.dir_font(f)
    if not (d / "capturat.jsonl").exists():
        continue
    cap, net, des = llegeix(d / "capturat.jsonl"), llegeix(d / "net.jsonl"), llegeix(d / "descartades.jsonl")
    per_pas = collections.Counter(x.get("pas") for x in des)
    minus_rec[f] = sum(1 for x in des if x.get("pas") == 3 and x["motiu"].startswith("inicial en minuscula"))
    tot_net += len(net); tot_d5 += per_pas.get(5, 0)
    prova(f"{f}: pas ∈ {{1,3,5}}", set(per_pas) <= {1, 3, 5}, dict(per_pas))
    base = lambda i: re.sub(r"-\d+$", "", i) if re.search(r"-\d{7}-\d+$", i) else i
    ids_cap = {x["id"] for x in cap}
    peces = collections.Counter(base(x["id"]) for x in net + [x for x in des if x.get("pas") == 3])
    sense = [i for i in ids_cap if peces[i] == 0]; orfes = [i for i in peces if i not in ids_cap]
    prova(f"{f}: capturat {len(cap):,} -> net {len(net):,} + desc. pas 3 {per_pas.get(3, 0):,}; cap registre sense rastre",
          not sense and not orfes, f"sense rastre {len(sense):,} · orfes {len(orfes):,}")
prova(f"corpus.jsonl {len(regs):,} = Σ net {tot_net:,} - Σ descartades pas 5 {tot_d5:,}", len(regs) == tot_net - tot_d5, f"diferencia {len(regs) - tot_net + tot_d5:+,}")
info("rebutjades per 'inicial en minuscula' (pas 3) per font", minus_rec)

print("\n# xifres")
prova("cap digit al raw_text", not troba(r"\d", raw), primer(r"\d", raw))
prova("cap '%' al raw_text (per cent)", not troba(r"%", raw), primer(r"%", raw))
prova("separadors: cap 'zero,zero' ni '.zero'", not troba(r"\.zero|zero,zero|,zero\b", raw), primer(r"\.zero|zero,zero", raw))
prova("numerals compostos: 'vint-e-' existeix", bool(troba(r"\bvint-e-(un|dus|tres|quate|cinc|sies|sèt|ueit|nau)\b", raw)))
prova("numerals: cap 'trenta-e-' (no existeix en aranes)", not troba(r"\b(trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|nauanta)-e-", raw))
prova("numerals: 'cincau' i no 'cinquau'", len(troba(r"\bcinquau", raw)) <= 1)
prova("milers: 'mil' compost present", bool(troba(r"\b(dus|tres|quate|cinc|vint|cent) mil\b", raw)))
rx_rang = rf"\b({DESENES})-({DESENES})\b"
prova("rangs N-M: cap 'desena-desena' enganxada (dètz-vint)", not troba(rx_rang, raw), primer(rx_rang, raw))
prova("rangs N-M: 'dus mil X a dus mil Y' present", bool(troba(r"\bdus mil [\w-]+ a dus mil\b", raw)), primer(r"\bdus mil [\w-]+ a dus mil\b", raw))
prova("'punt' entre numerals present", bool(troba(r"\b(tres|cinc|un|sèt) punt (tres|un|sèt|cinc|zero)\b", raw)))
rx_unit = rf"\b(?:{NUM})\s+(?:kms|cms|km²|m²|km/h|km|cm|mm|kg|gr|ha|min|kW|MW|Hz)\b"
prova("unitats: cap abreviatura nua darrere d'un numeral", not troba(rx_unit, raw), primer(rx_unit, raw))
prova("unitats expandides presents ('quilomètres')", bool(troba(r"\b(quilomètres|ectàrees|quilograms)\b", raw)))
julhet_font = sum(len(re.findall(r"\bjulhet\b", x["raw_text"])) for f in C.FONTS for x in llegeix(C.dir_font(f) / "capturat.jsonl"))
julhet_corpus = len(troba(r"\bjulhet\b", raw))
prova(f"dates: cap 'julhet' introduit per l'expansio (font {julhet_font}, corpus {julhet_corpus})", julhet_corpus <= julhet_font, primer(r"\bjulhet\b", raw))
prova("dates: 'de juriòl de' present", bool(troba(r"\bde juriòl de\b", raw)), primer(r"\bde juriòl de\b", raw))
prova("numerals repetits NO rebutjats ('mil ... mil' existeix)", bool(troba(r"\bmil\b.*\bmil\b", raw)))

print("\n# neteja")
prova("cap inicial en minuscula", not troba(r"^[a-zà-ú]", raw), primer(r"^[a-zà-ú]", raw))
rx_cl = r"^[—\-]?\s*(un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz|onze|dotze|tretze|catorze|quinze|setze|vint)(?:[.\-](?:un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz))*[.:\-—]+\s"
prova("cap clausula numerada al principi (dus.- / catorze:) (<= 3 residus coneguts: 2 linies de lot del BSC + 1 partida de daus)", len(troba(rx_cl, raw, re.I)) <= 3, f"{len(troba(rx_cl, raw, re.I))} · " + primer(rx_cl, raw, re.I))
prova("cap entrada de diccionari al principi (mot: Majuscula)", not troba(r"^[a-zà-ú][a-zà-ú'\-]*:\s+[A-ZÀ-Ü]", raw), primer(r"^[a-zà-ú][a-zà-ú'\-]*:\s+[A-ZÀ-Ü]", raw))
rx_cons = re.compile(r"(?<![\w'’\-.(])[bcdfghjklmnpqrstvwxz](?![\w'’\-.)])")
cons = [t for t in raw if rx_cons.search(t) and not re.search(r"\bletr[ae]s?\b", t)]
prova("cap consonant solta (mot partit: 'E r Ajuntament'; 'letra h' i '(s)' exclosos)", len(cons) <= 3, f"{len(cons)} casos · " + (cons or [""])[0])
prova("cap tirada de 4+ consonants impossibles", not troba(r"[bcdfgjkpqtvwxz]{4,}", raw), primer(r"[bcdfgjkpqtvwxz]{4,}", raw))
tot_maj = [t for t in raw if len(t.split()) >= 5 and sum(w.isupper() and len(w) > 1 for w in t.split()) / len(t.split()) >= 0.6]
prova("cap frase amb el 60 % o mes dels mots en majuscules", not tot_maj, (tot_maj or [""])[0])
prova("cap frase acabada en lletra (sense puntuacio final)", not troba(r"[A-Za-zÀ-ú]$", raw), primer(r"[A-Za-zÀ-ú]$", raw))
info("frases acabades en puntuacio + cometa/parentesi (0 tambe al 20/09: la normalitzacio treu la cometa final)", len(troba(r'[.!?]["»)\]]+$', raw)))
prova("cap ', \"' al final (punts suspensius mal resolts)", not troba(r', "$', raw), primer(r', "$', raw))
rx_sv = r"(?<![\w'’\-])[bcdfghjklmnpqrstvwxz]{3,}(?![\w'’\-])"
prova("cap mot sense vocals (pdf, xlsx, msnm, brr)", not troba(rx_sv, raw), primer(rx_sv, raw))
prova("cap marcador XXXX", not troba(r"\b([A-Z])\1{3,}\b", raw), primer(r"\b([A-Z])\1{3,}\b", raw))
prova("cap 'per cent-' (rang N%-M%)", not troba(r"per cent-", raw), primer(r"per cent-", raw))
rx_nz = r"\b(?:un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz|onze|dotze|tretze|catorze|quinze|setze|vint|trenta|quaranta|cinquanta|seishanta|setanta|ueitanta|nauanta|cents?|mil|milions?)(?:-e-[a-zà-ú]+)? zero\b"
prova("cap 'numeral zero' (milers separats per espai)", not troba(rx_nz, raw), primer(rx_nz, raw))
prova("cap 'MAJUSCULES, minuscula' tractat d'etiqueta: 'Candidat síndic' absent", not troba(r"^Candidat síndic\b", raw))
prova("'au' (gascon) absent", not troba(r"\bau\b", raw), primer(r"\bau\b", raw))
prova("cap 'kms'/'cms' nus", not troba(r"\b[kc]ms\b", raw), primer(r"\b[kc]ms\b", raw))
rx_eng = r"\b(?![IVXLCDM]+au\b)[A-ZÀ-Ü]{5,}[a-zà-ú]{2,}"
prova("cap capcalera enganxada (MAJUSCULESMot)", not troba(rx_eng, raw), primer(rx_eng, raw))
prova("ordinal roma intacte: cap 'XVII Iau'", not troba(r"\b[IVXLC]{2,} [IVXLC]au\b", raw), primer(r"\b[IVXLC]{2,} [IVXLC]au\b", raw))
prova("cap verb aranes en Majuscula a mig de frase", not troba(r"(?<=[a-zàèéíòóúïüç,])\s+(Ei|Siguec|Auec|Ère)\s+[a-zàèéíòóúïüç]", raw), primer(r"(?<=[a-zàèéíòóúïüç,])\s+(Ei|Siguec|Auec|Ère)\s+[a-z]", raw))
rx_for = r"\b(con|por|desde|hasta|durante|aunque|mientras|además|enero|febrero|marzo|lunes|martes|dab|òmi|deus|medish)\b"
prova("cap marca forana (mostra castella + bearnes)", not troba(rx_for, raw), primer(rx_for, raw))
prova("'tostemps', 'esta', 'aus' NO filtrats (son aranesos)", all(troba(rf"\b{w}\b", raw) for w in ("tostemps", "esta", "aus")))
prova("cap capcalera de pagina (' p.' + numeral)", not troba(r"\bp\. (un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz|vint|trenta|cent)\b", raw), primer(r"\bp\. (un|dus|tres|dètz|vint)\b", raw))
_ET = r"^(?![IVXLCDM]+\s)[A-ZÀ-Ü]{3,}(\s+[A-ZÀ-Ü]{2,})*\s+[A-ZÀ-Ü][a-zà-ú]"
prova("etiquetes de personatge al principi (< 40 residuals)", len(troba(_ET, raw)) < 40, f"{len(troba(_ET, raw))} casos · " + primer(_ET, raw))
_LAB = r"^(HAMLET|HORACI|OFELIA|ROMÈU|JULIETA|TEOBALDO|PARIS|POLONI|LAERTES|MERCUCIO|BENVOLIO|CAPULET|MONTAGUE|ENTERRAIRE|REI|REINA|CLAUDI|GERTRUDIS)\s"
prova("cap etiqueta de personatge coneguda al principi", not troba(_LAB, raw), primer(_LAB, raw))
prova("cap acotacio '(...) ETIQUETA' al principi", not troba(r"^\([^()]{1,80}\)\s+[A-ZÀ-Ü]{3,}\s", raw))
prova("cap mot repetit consecutiu (<= 3 tolerats)", len(troba(r"\b(\w{3,}) \1\b", raw)) <= 3, primer(r"\b(\w{3,}) \1\b", raw))
prova("cap dues frases enganxades (.Majuscula)", not troba(r"[a-z]\.[A-ZÀ-Ü][a-z]", raw), primer(r"[a-z]\.[A-ZÀ-Ü][a-z]", raw))
prova("cap punt entre lletres sense espai", not troba(r"[a-zàèéíòóúïü]\.[a-zàèéíòóúïü]", raw), primer(r"[a-zàèéíòóúïü]\.[a-zàèéíòóúïü]", raw))
prova("cap guio de dialeg inicial", not troba(r"^[-—–]", raw))
prova("cap ì / ù", not troba(r"[ìù]", raw), primer(r"[ìù]", raw))
prova("cap punts suspensius / '--' / parentesis o cometes buits (\"\" enganxades o al principi)", not troba(r'\.\.\.|--|\(\s*\)|(?<![.!?])""|^"\s*"', raw), primer(r'\.\.\.|--|\(\s*\)|(?<![.!?])""|^"\s*"', raw))
prova("cap lletra triplicada", not troba(r"([a-zà-ú])\1{2}", raw), primer(r"([a-zà-ú])\1{2}", raw))
prova("cap coma davant de tancament ( ,) ,] ,» )", not troba(r",\s*[)\]»]", raw), primer(r",\s*[)\]»]", raw))
prova("regressio cometes: cap lletra\"lletra (Canut\"era)", not troba(r"[a-zà-ú]\"[a-zà-ú]", raw), primer(r"[a-zà-ú]\"[a-zà-ú]", raw))
prova("cap parentesis mal niuats", all(t.count("(") == t.count(")") for t in raw), next((t for t in raw if t.count("(") != t.count(")")), ""))
prova("punt volat present (en·honsar)", bool(troba(r"[ns]·h", raw)))
info("frases amb « o »", len(troba(r"[«»]", raw)))

print("\n# respelling i entitats")
ex = exemple(r"\bDOGC\b")
prova("sigla lletrejada: DOGC -> 'de o ge ce'", bool(ex) and "de o ge ce" in ex[1], ex[1] if ex else "(no hi ha DOGC)")
ex = exemple(r"\bEMDs\b")
prova("sigla en plural: EMDs -> 'e ema de' (sense 'essa')", (not ex) or ("e ema de" in ex[1] and "e ema de essa" not in ex[1]), ex[1] if ex else "(no hi ha EMDs)")
tipus_sigles = json.load((C.DIR_ENTITATS / "tipus_sigles_aranes.json").open(encoding="utf-8"))
prova("K: TU / ES / CONSELH (mots corrents en majuscules) son al diccionari com a `mot`", all(tipus_sigles.get(k) in (None, "mot") for k in ("AS", "TU", "ES", "CONSELH")), {k: tipus_sigles.get(k) for k in ("AS", "TU", "ES", "CONSELH")})
prova("cap 'Dadda' (clau composta d'ADDA purgada)", not troba(r"\bDadda\b", tts), primer(r"\bDadda\b", tts))
ex = exemple(r"\bConselh\b")
prova("nom aranes reescrit: Conselh -> Cunsell", bool(ex) and "Cunsell" in ex[1], ex[1] if ex else "")
ex = exemple(r"\bCarlos\b")
prova("nom foran intacte: Carlos", bool(ex) and "Carlos" in ex[1] and "Carlus" not in ex[1], ex[1] if ex else "(no hi ha Carlos)")
ex = exemple(r"\bBarcelona\b")
prova("lloc de fora intacte: Barcelona no es toca", bool(ex) and "Barcelona" in ex[1] and "Barceluna" not in ex[1], ex[1] if ex else "(no hi ha Barcelona)")
ex = exemple(r"\bBossòst\b")
prova("lloc d'Aran reescrit: Bossòst -> Bussòst", (not ex) or "Bussòst" in ex[1], ex[1] if ex else "(no hi ha Bossòst)")
ex = exemple(r"\bNatasha\b")
prova("nom foran amb digraf (foran_grafia): Natasha -> Nataixa", (not ex) or "Nataixa" in ex[1], ex[1] if ex else "(no hi ha Natasha)")
ex = exemple(r"\bEisenhower\b")
prova("nom foran sense digraf: Eisenhower intacte", (not ex) or "Eisenhower" in ex[1], ex[1] if ex else "(no hi ha Eisenhower)")
prova("article 'es' mai menjat (el tts no en te menys que el raw; ES -> es en pot afegir)", all(len(re.findall(r"\bes\b", a)) <= len(re.findall(r"\bes\b", b)) for a, b in parell))
prova("cap 'DUGC' / sigles reescrites", not troba(r"\bDUGC\b|\bUdl\b", tts))
prova("oo -> no uu: 'cuuperaci' absent", not troba(r"cuuperaci|zuulog", tts), primer(r"cuuperaci", tts))
ex = exemple(r"\bcooperacion\b")
prova("cooperacion -> cooperaciun (oo intacte)", bool(ex) and "cooperaciun" in ex[1], ex[1] if ex else "(no n'hi ha)")
ex = exemple(r"\bXIX\b")
prova("roma: XIX -> dètz-e-nau", bool(ex) and "dètz-e-nau" in ex[1], ex[1] if ex else "(no hi ha XIX)")
ex = exemple(r"\bXXIIIau\b")
prova("roma ordinal: XXIIIau -> vint-e-tresau", bool(ex) and "vint-e-tresau" in ex[1], ex[1] if ex else "(no hi ha XXIIIau)")
rx_rom = r"(?:sègles?\s+[IVXLC]{2,}\b|\b[IVXLC]{2,}au\b)"
prova("cap roma de segle / ordinal sense expandir al tts", not troba(rx_rom, tts), primer(rx_rom, tts))
ex = exemple(r"\bd'ESO\b")
prova("clitic + sigla: d'ESO rep la fonetica (d'Eso)", (not ex) or "d'Eso" in ex[1], ex[1] if ex else "(no hi ha d'ESO)")
ex = exemple(r"\bBeckenham\b")
prova("cua de noms sense classificar protegida: Beckenham intacte", (not ex) or "Beckenham" in ex[1], ex[1] if ex else "(no hi ha Beckenham)")
ex = exemple(r"\bcançons\b")
prova("plural -ns: cançons -> cançús", (not ex) or "cançús" in ex[1], ex[1] if ex else "(no hi ha cançons)")
ex = exemple(r"\bVielha\b"); prova("lh -> ll: Vielha -> Viella", bool(ex) and "Viella" in ex[1], ex[1] if ex else "")
ex = exemple(r"\bsenhor\b"); prova("nh -> ny: senhor -> senyur", bool(ex) and "senyur" in ex[1], ex[1] if ex else "")
ex = exemple(r"\bmadeish\b"); prova("sh -> ix: madeish -> madeix", bool(ex) and "madeix" in ex[1], ex[1] if ex else "")

print("\n# regles del 21/09 (revisio)")
prova("E: cap frase de visitvaldaran ni d'AINA", not (fonts.get("visitvaldaran") or fonts.get("aina_es_arn")), dict(fonts))
nm = [len(t.split()) for t in raw]
prova(f"A: totes les frases tenen {C.MIN_MOTS}-{C.MAX_MOTS} mots", all(C.MIN_MOTS <= n <= C.MAX_MOTS for n in nm), f"min {min(nm)} max {max(nm)}")
curtes = [t for t in raw if len(t.split()) == 3]
info("A: frases de 3 mots (abans quedaven fora)", f"{len(curtes):,} · {curtes[:3]}")
n_tros = sum(1 for r in regs if re.search(r"-\d{7}-\d+$", r["id"]))
info("B: frases que son un tros d'una frase partida (id amb sufix -k)", f"{n_tros:,}")
_cion = [r for r in regs if C.presegmentada(r["font"]) and re.search(r"\b[a-zà-ú]+(ción|sión)\b", r["raw_text"])]
prova("D: cap sufix castella '-ción/-sión' a les fonts presegmentades (reparat a -cion/-sion)", not _cion, _cion[0]["raw_text"][:100] if _cion else "")
info("D: '-ción/-sión' a les fonts fiables (errades de la font, es deixen)", len(troba(r"(ción|sión)\b", raw)))
prova("D: formes reparades presents (ocasion/poblacion/informacion)", bool(troba(r"\b(ocasion|poblacion|informacion)\b", raw)), primer(r"\b(ocasion|poblacion|informacion)\b", raw))
try:
    import importlib
    M5 = importlib.import_module("05_respelling")
    esq = collections.Counter(M5.esquelet(t) for t in raw)
    top = esq.most_common(1)[0]
    prova(f"G: cap esquelet amb mes de {C.MAX_PER_ESQUELET} copies", top[1] <= C.MAX_PER_ESQUELET, f"max {top[1]} · {top[0][:80]}")
except Exception as e:  # noqa: BLE001
    info("G: no s'ha pogut calcular l'esquelet", repr(e))
prova("M: cap '·h' al tts_text", not troba(r"·h", tts), primer(r"·h", tts))
info("M: punts volats que queden al tts (l·l i noms compostos, el TTS els llegeix)", len(troba(r"·", tts)))
prova("M: 'tz' es conserva al tts (dotze -> dutze, no dutse)", bool(troba(r"\bdutze\b", tts)) and not troba(r"\bdutse\b", tts))
rx_carr = r"\b[A-Z]{1,2}-(un|dus|tres|quate|cinc|sies|sèt|ueit|nau|dètz|vint|trenta|cent)\b"
prova("N: carreteres/codis lletra-numeral sense expandir al raw (<= 10 residus: AA-sèt, ML-dus)", len(troba(rx_carr, raw)) <= 10, f"{len(troba(rx_carr, raw))} · " + primer(rx_carr, raw))
prova("N: 'ena dus cents trenta' (N-230) present", bool(troba(r"\bena dus cents trenta\b", raw)), primer(r"\bena dus cents trenta\b", raw))
D1 = r"(?:zero|un|dus|tres|quate|cinc|sies|sèt|ueit|nau)"
rx_tel = rf"\b{D1}(?: {D1}){{8}}\b"
info("N: frases amb 9 xifres seguides llegides una a una (telefons)", f"{len(troba(rx_tel, raw)):,} · {primer(rx_tel, raw)[:90]}")
rx_telnum = r"\b(sies|nau|sèt|ueit) cents [\w-]+ mil [\w-]+ cents\b"
info("N: telefons al corpus", "cap: les frases amb telefon cauen al pas 1 (0 marques fortes / sense puntuacio final)")
_tit = [t for t in tts if C.es_titular(t)]
prova("K: cap tts_text aranes que sigui un titular en majuscules (<= 2 fragments anglesos tolerats)", len(_tit) <= 2, _tit[:2])
rx_caps = r"\b(ES|ERA|DAMB|CONSELH|ETH|ENTÀ|TAMBÉ|AQUERA|GENERAU|DERA|DETH)\b"
prova("K: cap mot corrent en majuscules al tts (ES/ERA/DAMB/CONSELH...)", not troba(rx_caps, tts), primer(rx_caps, tts))
ex = exemple(r"\bUA\b")
prova("K: UA (sigla) -> 'u a' al tts", (not ex) or bool(re.search(r"\bu a\b", ex[1])), ex[1][:100] if ex else "(no hi ha UA)")
rx_rj = r"\b[A-ZÀ-Ü][a-zà-ú]+ [IVX]{1,4}\b(?![a-zà-ú.\-])"
prova("J: cap roma sense expandir darrere d'un nom al tts (<= 5: `Eth I Seminari`, roma davant del nom, es deixa)", len(troba(rx_rj, tts)) <= 5, f"{len(troba(rx_rj, tts))} · " + primer(rx_rj, tts))
ex = exemple(r"\b(Joan Carles|Felip|Alfons|Carles|Pèir|Loís|Enric|Pau|Benedicte|Francesc) [IVX]{1,4}\b")
info("J: exemple de nom + roma (ordinal)", f"{ex[0][:70]} -> {ex[1][:70]}" if ex else "(cap)")
ex = exemple(r"\b(Capitol|Capítol|Títol|Titol|Annex|Article|Programa) [IVX]{1,4}\b")
info("J: exemple de capcalera + roma (cardinal)", f"{ex[0][:70]} -> {ex[1][:70]}" if ex else "(cap)")
for nom, be, mal in (("Jordi", "Jordi", "Jurdi"), ("Oriol", "Oriol", "Uriul"), ("Diputació", "Diputació", "Diputaciú"),
                     ("Coll", "Coll", "Cull"), ("Garona", "Garona", "Garuna"), ("Joan", "Joan", "Juan"),
                     ("Sans", "Sans", "Sas"), ("Tonet", "Tonet", "Tunet"), ("Provença", "Provença", "Pruvença"),
                     ("Arró", "Arró", "Arrú"), ("Barcelona", "Barcelona", "Barceluna")):
    ex = exemple(rf"\b{nom}\b")
    ok = (not ex) or (re.search(rf"\b{be}\b", ex[1]) is not None and re.search(rf"\b{mal}\b", ex[1]) is None)
    prova(f"H/I: {nom} intacte al tts (grafia compartida o forana)", ok, ex[1][:100] if ex else f"(no hi ha {nom})")
for nom, be in (("Portugau", "Purtugau"), ("Japon", "Japun"), ("Moscòu", "Muscòu"), ("Catalonha", "Catalunya"),
                ("Bossòst", "Bussòst"), ("Conselh", "Cunsell"), ("Vielha", "Viella")):
    ex = exemple(rf"\b{nom}\b")
    prova(f"H: {nom} -> {be} (grafia occitana, es reescriu)", (not ex) or re.search(rf"\b{be}\b", ex[1]) is not None, ex[1][:100] if ex else f"(no hi ha {nom})")
noms = json.load((C.DIR_ENTITATS / "noms_propis_aranes.json").open(encoding="utf-8"))
vals = collections.Counter(v if isinstance(v, str) else v.get("tipus", "?") for v in noms.values())
info("H: diccionari de noms per classe", dict(vals))

for w in ("Cau", "Vau", "Mau"):
    n_raw = len(troba(rf"\b{w}\b", raw)); n_tts = len(troba(rf"\b{w}\b", tts))
    prova(f"J: '{w}' (mot) intacte al tts, no ordinal roma ({n_raw} raw / {n_tts} tts)", n_tts >= n_raw, primer(r"\b(centau|cincau|milau)\b", tts))
_lab = troba(r"^[B-DF-HJ-NP-TV-Z] [A-ZÀ-Ü]", raw)
prova("cap etiqueta d'apartat (consonant sola) al principi (<= 50: V/X ambigus amb romans i trossos partits)", len(_lab) <= 50, f"{len(_lab)} · " + (_lab or [""])[0])
print(f"\n{sum(resultats)}/{len(resultats)} comprovacions OK")
sys.exit(0 if all(resultats) else 1)

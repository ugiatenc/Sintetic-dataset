#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera README.md a partir de resultats_ablacio.json (i resultats_ng.json si hi es)."""
from __future__ import annotations

import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ORDRE = ["cru", "digrafs", "o_tot", "o_tonica", "ng", "ts", "totes",
         "proposta", "proposta_digrafs"]


def ic(d):
    """Formata un interval de confianca."""
    if not d.get("n"):
        return "—"
    return f"{d['mitjana']:+.3f} [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]"


def veredicte(t):
    """Una regla te efecte si l'IC95% de la seva delta, alla on actua, no toca el zero."""
    d = t["delta_vs_cru_tocades"]
    if not d.get("n"):
        return "—"
    baix, alt = d["ic95"]
    if alt < 0:
        return "**baixa P(ca)**"
    if baix > 0:
        return "**puja P(ca)**"
    return "indistingible de zero"


def main() -> int:
    """Punt d'entrada: arguments de la linia d'ordres i execucio."""
    r = json.loads((AQUI / "resultats_ablacio.json").read_text(encoding="utf-8"))
    V, ref, obj = r["variants"], r["referencia"], r["objectiu_p_catala"]
    noms = [n for n in ORDRE if n in V] + [n for n in V if n not in ORDRE]
    fng = AQUI / "resultats_ng.json"
    NG = json.loads(fng.read_text(encoding="utf-8")) if fng.exists() else None

    files_tot = "\n".join(
        f"| `{n}` | {V[n]['frases_tocades']}/115 | {V[n]['p_catala']:.3f} | "
        f"{V[n]['dif_amb_real_4_veus']:+.3f} | {V[n]['js_amb_real_4_veus']:.3f} | "
        f"{ic(V[n]['delta_vs_cru_115'])} | {V[n]['WER']:.3f} | {V[n]['CER']:.3f} |"
        for n in noms)
    files_toc = "\n".join(
        f"| `{n}` | {V[n]['descripcio']} | {V[n]['delta_vs_cru_tocades'].get('n', 0)} | "
        f"{ic(V[n]['delta_vs_cru_tocades'])} | {veredicte(V[n])} |"
        for n in noms if n != "cru")

    if NG:
        bloc_ng = f"""
## `ng` sobre les regles d'avui

L'ablacio mesura `ng` **sol**, sobre text aranes cru. Pero les regles d'avui es queden, o
sigui que la decisio real es "`ng` A MES de les regles" (`totes_ng`), i aquest joc no hi
era. Es va generar a part (`escolta_ng.py`):

| Joc | P(ca) | Δ vs `cru` | Δ vs `totes` | WER | CER | frases amb `g` de mes |
|---|---|---|---|---|---|---|
""" + "\n".join(
            f"| `{n}` | {NG['p_catala'][n]:.3f} | {ic(NG['deltes'].get(f'{n}_vs_cru', {}))} | "
            f"{ic(NG['deltes'].get(f'{n}_vs_totes', {}))} | {NG['metriques'][n]['WER']:.3f} | "
            f"{NG['metriques'][n]['CER']:.3f} | "
            f"**{NG['g_de_mes'][n]['frases']}**/{NG['g_de_mes'][n]['de']} |"
            for n in ("cru", "totes", "ng", "totes_ng")) + f"""

**`ng` no sobreviu a la combinacio.** Sol dona {ic(V['ng']['delta_vs_cru_115'])}; damunt de les
regles d'avui es queda en {ic(NG['deltes']['totes_ng_vs_totes'])}, i comparat amb el text cru,
{ic(NG['deltes']['totes_ng_vs_cru'])}. Els dos intervals creuen el zero. Les regles no son
additives: el que `ng` guanya sobre text cru, es perd quan la resta de regles ja hi son.

### I la [g] s'hi sent

L'ultima columna es el test directe de la por amb que les normes van descartar aquesta
regla. Compta les frases on **Whisper escriu una `g` que el text aranes no porta**: si
l'escriu, l'ha sentida. Passa de {NG['g_de_mes']['totes']['frases']}/115 sense la regla a
{NG['g_de_mes']['totes_ng']['frases']}/115 amb ella.

""" + "\n".join(f"- ARANES `{e['aranes']}`\n  WHISPER `{e['whisper']}`"
                 for e in NG["g_de_mes"]["totes_ng"]["exemples"][:4]) + f"""

El WER no ho detectava: ja es de 0.7 i una [g] de mes s'hi perd. **La baixada de P(catala)
de `ng` no es que soni mes aranes, es que soni espatllat** -- i espatllat tambe s'allunya
del catala. Es exactament el mode de fallada contra el qual serveix la columna JS, i el
motiu pel qual P(catala) tota sola no es un objectiu valid.
"""
    else:
        bloc_ng = ""

    d_totes, d_o_tot = ic(V["totes"]["delta_vs_cru_115"]), ic(V["o_tot"]["delta_vs_cru_115"])
    d_o_ton = ic(V["o_tonica"]["delta_vs_cru_115"])
    d_dig, d_dig_t = ic(V["digrafs"]["delta_vs_cru_115"]), ic(V["digrafs"]["delta_vs_cru_tocades"])
    d_ng, d_ng_t = ic(V["ng"]["delta_vs_cru_115"]), ic(V["ng"]["delta_vs_cru_tocades"])
    d_ts_t = ic(V["ts"]["delta_vs_cru_tocades"])
    g_ng = NG["g_de_mes"]["totes_ng"]["frases"] if NG else 0
    g_base = NG["g_de_mes"]["totes"]["frases"] if NG else 0
    pct = (V["cru"]["p_catala"] - V["ng"]["p_catala"]) / (V["cru"]["p_catala"] - obj) * 100

    (AQUI / "README.md").write_text(f"""# Ablacio de les regles de reescriptura fonetica

Quina de les regles de `src/respelling.py` fa que l'audio sintetic soni **menys catala**, i
quines de noves valdrien la pena. Es genera **un joc d'audios per regla** sobre les
mateixes 115 frases del split `test` i amb les mateixes 4 veus clonades: entre jocs nomes
canvia el text que llegeix el TTS.

> ### Que NO respon aquest experiment
>
> Nomes mesura una cosa: **si una regla allunya l'audio del catala**. No mesura si la
> pronuncia es correcta, que es una pregunta diferent i es la que ja es va validar
> escoltant els audios a ma.
>
> Les dues poden ser certes alhora, i de fet ho son: `nh`->`ny` fa que el TTS digui [ɲ]
> en comptes d'una `n` seguida de res -- una correccio real -- i **precisament per aixo**
> no pot baixar P(catala), perque [ɲ] es un so catala. Que una regla surti "indistingible
> de zero" aqui **no es un argument per treure-la**.

Tot el que hi ha aqui es autocontingut. Els jocs `cru` i `totes` es reaprofiten de
`lab/proves_inicials/aranes/audios/validacio_tts/` (nomes es llegeixen); la resta d'audios son a
`audios/`.

```
python3 ablacio.py --comprovar     # equivalencia amb respelling.py, sense GPU
python3 ablacio.py --nomes-textos  # els textos de cada variant
python3 ablacio.py                 # genera, mesura, escriu resultats_ablacio.json
python3 informe.py                 # refa aquest README des dels JSON
python3 escolta_u.py               # prova d'escolta: grafies per a la `u` [y]
python3 escolta_ng.py              # `ng` damunt les regles d'avui + carpeta d'escolta
```

## Com llegir les xifres

**P(catala)** es la probabilitat que Whisper doni al catala sense transcriure. L'objectiu
**no** es minimitzar-la, sino acostar-se a la de l'aranes real.

**L'objectiu bo es {obj:.3f}**, no {ref['real']['p_catala']:.3f}. El README de `escolta_hibrid` compara contra els
59 locutors del split test ({ref['real']['p_catala']:.3f}), pero els audios sintetics clonen **4 locutors
concrets**, i l'audio real d'aquests 4 ({ref['real_4_veus']['n']} clips) dona **{ref['real_4_veus']['p_catala']:.3f}**. Es la referencia
correcta i deixa la distancia a tancar en {V['cru']['dif_amb_real_4_veus']:+.3f}, no en +0.156.

**Δ vs `cru`** es la comparacio **aparellada**: per a cada frase, P(ca) de la variant menys
P(ca) del text cru, amb la mateixa veu i la mateixa llavor. L'interval es un bootstrap de
10.000 remostreigs sobre les frases. Es l'interval que faltava a la mesura original: sense
ell, un -0.012 no es pot distingir de zero.

**JS** es la distancia de Jensen-Shannon entre el perfil d'idioma sencer de la variant i
el dels 4 locutors reals. Hi es perque P(catala) tota sola es degenerada: `language="oc"`
te P(ca) = 0.06 i no s'assembla mes a l'aranes, nomes s'ha desviat cap al frances.

**WER/CER** son contra l'aranes original: mesuren si una regla descatalanitza a canvi de
trencar la intelligibilitat.

## Resultat per variant (les 115 frases)

| Variant | Frases tocades | P(ca) | dif. amb el real | JS | Δ vs `cru` (IC95%) | WER | CER |
|---|---|---|---|---|---|---|---|
{files_tot}

Referencia: **real, 4 locutors clonats** P(ca) = {ref['real_4_veus']['p_catala']:.3f} ({ref['real_4_veus']['n']} clips) ·
**real, 59 locutors del test** P(ca) = {ref['real']['p_catala']:.3f} (WER {ref['real']['WER']:.3f}, CER {ref['real']['CER']:.3f}).

## Efecte de cada regla alla on actua

Una regla que nomes toca 9 frases de 115 surt diluida a la taula de dalt encara que faci
molt alla on entra. Aqui la mateixa delta aparellada es calcula **nomes sobre les frases
que la variant canvia**.

| Variant | Que fa | Frases | Δ vs `cru` (IC95%) | Veredicte |
|---|---|---|---|---|
{files_toc}

"Indistingible de zero" amb poques frases vol dir que **no se sap**, no que la regla no
faci res. I, com diu l'avis de dalt, tampoc no vol dir que la regla sobri.
{bloc_ng}
## Que en surt

**1. Cap regla d'ortografia no descatalanitza l'audio de manera demostrable.** Ni les
d'avui juntes ({d_totes}), ni `o_tot` tota sola ({d_o_tot}), ni `o_tonica`
({d_o_ton}), ni `ts`. L'unica que en solitari surt de zero es `ng`, i es desfa en
combinar-la (veure la seccio anterior).

**2. `o_tot` es un zero ben mesurat, no un "no se sap".** Fa {V['o_tot']['edicions']['o_tot']} edicions -- el 78 % de
totes -- i dona {d_o_tot}: l'interval mes estret de l'experiment. Amb `language="ca"`,
OmniVoice ja tanca prou la `o`, atona i tonica. Aixo **no** diu que la regla sobri: diu
que el que aporta es pronuncia correcta, no distancia amb el catala. Si es vol la meitat
del cost en WER, `o_tonica` fa {V['o_tonica']['edicions']['o_tonica']} edicions en comptes de {V['o_tot']['edicions']['o_tot']} amb el mateix efecte
(WER {V['o_tonica']['WER']:.3f} contra {V['o_tot']['WER']:.3f}) -- pero caldria tornar a escoltar-ho abans de canviar-ho.

**3. Els digrafs empenyen, si de cas, cap al catala.** {d_dig} sobre les 115 i
{d_dig_t} alla on actuen. Cap interval exclou el zero, pero el signe es el que preveu
el mecanisme: [ɲ], [ʎ] i [ʃ] son sons de l'inventari catala. Es exactament el que s'espera
d'una regla que existeix per corregir la pronuncia, no per allunyar-la del catala.

**4. `ng` s'ha de descartar, i les normes ja tenien rao.** En solitari baixa P(catala)
{d_ng}, l'unic efecte significatiu de tot l'experiment -- pero per un motiu dolent:
**hi posa una [g] audible**, que Whisper escriu en {g_ng}/115 frases contra {g_base}/115
sense la regla (`Engguha`, `guha ira`, `Nung li`, `mengpair`). I damunt de les regles
d'avui l'efecte ni tan sols sobreviu. La baixada de P(catala) era un artefacte de
degradacio, no de fonetica aranesa. L'escolta manual que ja deia que la `-n` final no
sonava malament queda confirmada; a `escolta/ng/` hi ha els audios per verificar-ho.

**5. Descartat `ng`, l'ortografia s'ha quedat sense marge.** El millor joc que queda dret
son les regles d'avui, a {V['totes']['dif_amb_real_4_veus']:+.3f} del real. Cap regla de grafia catalana no tanca
aquesta distancia, i no es estrany: el que falta -- la `u` [y] i la `h` aspirada -- **no te
grafia catalana**. Per aixo la prova seguent no es una regla mes, sino provar grafies
d'ALTRES idiomes (`ü` alemany, `y` nordic, el simbol AFI): `escolta/u_simbols/`.

### Recomanacio

- **Deixar les regles com estan.** Aquest experiment no dona cap motiu per treure'n cap:
  no mesura allo per a que es van posar.
- **No afegir `ng`**: hi posa una [g] audible. Les normes tenien rao.
- **La `u` es on queda marge de debo**, i es on hi ha la prova d'escolta preparada.
- **No esperar gaire mes de l'ortografia.** Per acostar-se al perfil de l'aranes real cal
  una altra palanca (l'etiqueta d'idioma, o barrejar-la), no mes regles de grafia.

## Proves d'escolta

| Carpeta | Pregunta |
|---|---|
| `escolta/u_simbols/` | Quina grafia (`ü`, `y`, `iu`, `ui`, `ʏ`) dona la `u` aranesa [y]? |
| `escolta/ng/` | Confirmar per l'orella la [g] que Whisper ja delata |

Cada carpeta te el seu README i un `frases.csv` amb columnes per anotar-hi.

## Que hi ha a cada fitxer

| Fitxer | Que es |
|---|---|
| `regles.py` | Les regles, activables d'una en una. Amb `REGLES_ACTUALS` dona text identic a `src/respelling.py` |
| `ablacio.py` | Genera els audios, mesura i escriu `resultats_ablacio.json` |
| `escolta_u.py` | Grafies candidates per a la `u` [y] + carpeta d'escolta |
| `escolta_ng.py` | `totes+ng` sobre les 115 + carpeta d'escolta |
| `informe.py` | Aquest README, des dels JSON |
| `audios/<variant>/` | Els audios generats per aquest experiment |

## Limitacions

- **La deteccio d'idioma de Whisper es un proxy, no una orella.** Que Whisper deixi de dir
  "catala" no demostra que la pronuncia sigui mes aranesa: pot ser nomes que el TTS hi fa
  una cosa rara. `ng` n'es el cas practic, i per poc no passa per bo. En clips de 4-5 s, a mes, confon catala amb islandes i basc. Qui
  decideix es un parlant nadiu.
- **El disseny nomes respon "s'allunya del catala?".** La correccio de pronuncia, que es
  per a que existeixen les regles d'avui, no es mesura aqui.
- **Amb 115 frases i 4 veus, l'experiment detecta efectes d'uns 0.05 de P(ca).** Un
  interval que creua el zero vol dir "no demostrat", no "demostrat que no".
- **El sintetic parla un 28 % mes rapid que el real** i aixo afecta la deteccio d'idioma.
  Aqui no s'hi toca (`speed` per defecte) perque els jocs reaprofitats es van generar aixi.
""", encoding="utf-8")
    print(f"Escrit: {AQUI / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

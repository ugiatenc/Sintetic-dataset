# Decisions del pipeline aranès

Què s'ha decidit, per què i amb quina mesura. Substitueix la revisió i l'auditoria del
18-25/09/2026; les versions llargues són fora del repositori
(`../abans_neteja_commit_2509.tar.gz`).

---

## 1. Avaluació i fuites

**El test és sagrat.** 448 clips reals de Common Voice 22.0 `oc` (0,63 h, 80 locutors),
dels splits `test`, `dev` i `train`, que declaren accent aranès o passen el filtre
dialectal. Hi són els 16 locutors que declaren aranès.

- Cap frase del test entra al corpus (s'exclou als passos 1, 3 i 5).
- Cap locutor del test entra al banc de veus ni als clips reals del train, vingui del split
  que vingui. Fins al 21/09 només s'excloïa el split `test` de CV i 21 de les 51 veus del
  banc eren del test.
- Decisió Q: 2 locutors amb molt més àudio a `other` que al test (633 i 137 clips contra 14
  i 10) es van treure del test (472 → 448 clips) perquè el seu àudio pogués entrar al train.
  Els WER d'abans del 21/09 no són comparables.

**El dev, separat del test** (25/09). A `dataset_aranes_eval/`: 5 locutors reals de Common
Voice (`config.LOCUTORS_DEV_REAL`, 251 clips, 21 min) i 1 de cada 200 frases sintètiques
(hash estable de l'id). Un 90/10 aleatori sobre el sintètic no valia: mesuraria el domini
sintètic amb les mateixes veus i frases.

- Cap frase del dev és al train. Common Voice i el BSC surten dels mateixos llibres i 76 de
  les 251 frases del dev real també eren al corpus: es marquen `exclos` i no es generen.
- Matís inevitable: la veu clonada de 4 dels 5 locutors del dev és al train llegint altres
  frases. Sense veus del banc només quedaven 32 clips reals.

**Línia base** (Whisper large-v3-turbo sobre àudio real): **WER 0,91, CER 0,36**. El model
sent els sons però no sap escriure'ls: el dataset ha d'ensenyar ortografia. Forçar `oc` o
`ca` dona el mateix (0,826 contra 0,841); per entrenar es fa servir `<|oc|>`, que és el
token correcte. Per mesurar cal normalitzar: el test porta l'apòstrof `’` i el corpus `'`.

## 2. Corpus (passos 1-5)

**Fonts.** Fiables: Conselh Generau d'Aran, Arannoticies, Institut d'Estudis Aranesi i 30
frases de Claude. Reserva: corpus paral·lel català-aranès del BSC i Viquipèdia `oc`.
Fora (decisió E): AINA, traducció automàtica amb un 5,7 % de castellà residual, i
`visitvaldaran`.

**Filtre dialectal.** L'occità no és l'aranès: el llenguadocià escriu `lo`/`amb` on
l'aranès escriu `eth`/`damb`, i una frase contaminada ensenya a escriure malament. Una
frase passa amb **una marca aranesa forta i cap de forana**. Calibrat amb 472 frases
araneses, 2.000 catalanes i 1.183 llenguadocianes:

| Regla | Recall aranès | Falsos positius català | Falsos positius llenguadocià |
|---|---|---|---|
| fortes ≥ 1 | 84 % | 0,1 % | 2,4 % |
| **fortes ≥ 1 i foranes = 0** | **81 %** | **0,0 %** | **0,5 %** |
| fortes ≥ 2 i foranes = 0 | 42 % | 0,0 % | 0,0 % |

Les marques foranes inclouen el gascó de França i el francès (12.741 frases fora sense
tocar el recall). Cada marca nova es valida contra les fonts fiables abans d'afegir-la.

**Llargada.** Mínim 3 mots i 12 caràcters (decisió A: el test és parla curta). Es parteix a
partir de 290 caràcters i es descarta per sobre de 350, calculat a la velocitat lenta del
TTS (11,9 car/s) perquè cap àudio passi de 29 s. Primer es talla per `;` `:` i guió llarg;
si no n'hi ha prou, per talls febles (`, e `, `, mès `, `, pr'amor que `…) amb reagrupament
equilibrat (decisió B: 11.945 frases partides en lloc de perdre-les).

**Porta de qualitat.** Regla general: si l'error és de forma i la frase és bona, es repara;
si el contingut és brossa, es rebutja. Es reparen apòstrofs desplaçats, marques de llista,
números de clàusula, lletres d'apartat, inicials en minúscula de fonts ja segmentades i
sufixos amb accent castellà del BSC (`ocasión` → `ocasion`). `enumeracio` només rebutja
llistes de 5+ noms propis o numerals (decisió C). Al BSC, 2 senyals de castellà tomben la
frase (D). Cap plantilla no pot sortir més de 20 cops (G).

**Números en lletres.** `num2words` no té aranès i treure les frases amb xifres costava
~273 h. La taula de `config.py` està validada contra 484.292 frases reals: només `vint`
composa amb `-e-`, el 5è és `cincau`, els únics irregulars són `prumèr` i `setau`, juliol és
`juriòl`. Telèfons xifra a xifra, carreteres lletra a lletra (`N-230` → `ena dus cents
trenta`). Un romà d'una lletra amb `au` només pot ser `Iau` o `Xau` (`Cau` és un mot).

**Resultat (21/09):** 261.998 frases · 620,9 h estimades (fiables 108,5 h, BSC 509 h).
Verificació de 130 comprovacions, totes correctes (`lab/verifica_corpus_aranes.py`).

## 3. `tts_text`: què llegeix el TTS

OmniVoice no té aranès. Amb `oc` sona afrancesat (P(català) 0,06), amb `ca` exagera el
català (0,52), i l'aranès real queda entre mig (0,41). Es genera amb `ca` sobre un text
reescrit en grafia catalana, i el `raw_text` conserva l'aranès real com a ground truth.

**Regles** (`src/respelling.py`, `data/aranes/entitats/normes_ortografiques_aranes.json`):
`nh` → `ny`, `lh` → `ll`, `sh` → `x`/`ix`, `o`/`ó` → `u`/`ú` (no `ò`, `ou` ni `oo`),
plurals en `-ns` → `-s`, i fora el `·h` del punt volat. El `tts_text` és una funció pura
del `raw_text` i dels diccionaris: el pas 10 el torna a calcular i ha de coincidir.

- **Mesurat:** cap regla mou P(català) de manera mesurable (totes juntes, −0,012 [−0,065,
  +0,041]). Es mantenen perquè corregeixen la pronunciació, validada d'oïda.
- **La `o` → `u` és correcta:** els locutors reals diuen [u] en el 77 % de les `o`/`ó` (84 %
  en mots funcionals); la [o] és interferència del català o castellà i noms de fora. La `ò`
  és [ɔ] el 96 % de les vegades.
- **Innecessari, mesurat:** `tz` → `ts` (OmniVoice ja diu [ts]) i una veu nord-occidental
  per a la `-a` final (ja surt [a]).
- **Descartat:** `-n` → `-ng`. Baixava P(català), però Whisper hi sentia una `g` en 25 de
  115 frases: sonava trencat, no aranès.

**Entitats: el model classifica, el codi escriu** (pas 4, sempre amb la frase com a
context).

- **Sigles:** `lletrejada` (`DOGC` → `de o ge ce`), `acronim`, `mot`, `romana` o `normal`.
  Un mot corrent en majúscules passa a minúscula (decisió K: `CONSELH` → `cunsell`; `UA`
  partit → `u a`, `UA` en un titular → `ua`).
- **Noms: criteri ortogràfic** (decisió H). Grafia occitana → regles (`Conselh` →
  `Cunsell`, `Moscòu` → `Muscòu`). Grafia de fora o compartida amb el català o el castellà →
  intacte (`Barcelona`, `Joan`, `Jordi`, `Sans`). Grafia de fora amb dígrafs occitans → només
  els dígrafs (`Catalonha` → `Catalunya`). Els noms massa rars per classificar es protegeixen
  intactes (L).
- **Romans:** darrere d'un nom, ordinal (`Jaime II` → `dusau`, `Isabel II` → `dusaua`);
  darrere d'una capçalera, cardinal (`capítol V` → `cinc`).

## 4. Veus (pas 6)

Una referència és **un clip sencer** d'un locutor amb el seu text exacte, de 3 a 8 s. No es
retallen ni s'empalmen clips: el text deixaria de correspondre.

- **Neta:** SNR ≥ 30 dB (catalanes ≥ 45), sense transitoris, i que soni com la resta de
  clips del locutor (ECAPA). El TTS clona també el soroll de fons.
- **Prova de síntesi:** cada veu llegeix 3 cops una frase sense numerals, Whisper la
  transcriu i la veu cau si la mediana del CER contra el consens passa de 0,25. Les
  mètriques de senyal no ho predeien: dues veus amb nota 0,84-0,88 donaven galimaties. La
  primera versió de la prova, amb numerals i una sola lectura, era inestable: OmniVoice no
  és determinista ni amb llavor fixa.
- **L'accent declarat no és criteri:** CER contra el consens 0,109 declarats, 0,087 no
  declarats, 0,092 catalanes. La pronúncia la posen el `tts_text` i el TTS, no la
  referència. A més, tots els que declaren aranès són al test.
- **Resultat:** 67 veus = 47 occitanes + 20 catalanes (10 dones, 10 homes) per diversitat
  de timbre. La literatura diu que amb 50-100 veus n'hi ha prou. YouTube no serveix: música
  sota la veu, 0 referències netes de 149 trams.

## 5. Entorns (pas 7)

13 entorns actius, situacions d'Aran: estudi de ràdio, sala de plens, ajuntament,
església, aula, casa, cuina, muntanya amb riu o vent, carretera, autobús, parc, telèfon.
Soroll real de DEMAND; sales simulades amb pyroomacoustics i el RT60 mesurat (església
1,8 s, plens 0,87, estudi 0,18); canals amb còdec GSM real al telèfon. L'entorn es tria
coherent amb la font: un ple del Conselh no sona des d'un riu.

## 6. Generació (pas 8)

- **Pla determinista:** veus en round-robin (3.910 frases cadascuna), entorn per font i
  pes, velocitat 0,94-1,06 i llavor estable per frase. Cada frase es genera un sol cop.
- **Ordre:** fonts fiables al davant a ritme 2:1 per hores (a les 162 h ja hi són totes),
  per blocs de 2.000. Així, s'aturi quan s'aturi, el dataset és vàlid i té primer el text
  més fiable.
- **Velocitat:** dins de cada veu les frases s'ordenen per durada abans de fer lots de 4.
  Un lot dura el que dura la frase més llarga: de 4,2 a 6,1 s d'àudio per segon (×1,42).
  Lots de 8 anaven pitjor.
- **OmniVoice** amb els paràmetres per defecte (32 passos, guidance 2,0, denoise) i
  `language="ca"`. L'únic filtre és la durada: > 29 s es descarta (0,4 %).
- **Revisió de 804 clips:** CER mediana 0,18 (p90 0,30), 1 % de galimaties (sobretot
  frases de ≤ 5 mots), cap truncament ni saturació. Cap veu ni entorn falla.
- **Recursos:** ~115 MB per hora d'àudio (~74 GB en total). La GPU treballa a 84-85 °C; si
  cal, `nvidia-smi -pl 125` baixa la temperatura a canvi d'un 5-10 % de velocitat.

## 7. Dataset i àudio real

- **Estructura** (08, 08b, 09): `dataset_aranes/<font>_<bloc de 4.000>/<id>.wav + .json`;
  el dev a `dataset_aranes_eval/<font>/`; `train.jsonl` i `dev.jsonl`.
- **Àudio real de Common Voice:** de 11,8 h baixades, 7,2 h són de locutors del test i no
  es poden fer servir. En queden 1.352 clips (1,94 h): 1.097 al train (35 locutors) i 251
  al dev. Són l'1 % del train i l'únic aranès de debò: l'script d'entrenament els repeteix
  ×5.
- **Auditoria** (pas 10): `tts_text` reproduïble, 0 frases del test, 0 locutors del test,
  0 frases del dev al train, cap àudio de més de 30 s.

## 8. Descartat

| Què | Per què |
|---|---|
| Marca forta obligatòria també al pas 3 (F) | tombava el 85 % de frases bones |
| Dígrafs als noms rars protegits (L) | de moment no |
| Afinar el TTS amb aranès | fora d'abast |
| AINA | traducció automàtica, castellà residual |
| Veus de YouTube | música de fons |
| `-n` → `-ng`, `tz` → `ts`, veu per la `-a` | mesurats: dolent o innecessari |

## 9. Obert

- **P:** variació [o] en un 15-20 % dels clips, com fan els parlants reals.
- **Verificador per clip:** Whisper sobre tots els clips i fora els de CER > 0,5; trauria
  l'1 % de galimaties (~11 h de GPU).
- **Frases curtes i col·loquials** generades amb el pas 2 (hores, preus, neu, salutacions):
  el corpus és llarg, literari i administratiu.
- **Escolta d'un parlant nadiu:** cap mètrica no diu si sona a aranès autèntic.

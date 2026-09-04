# SINTETIC-DATASET

Simulador de dataset sintètic (àudio + ground truth) per a Whisper, en condicions
acústiques reals. Dos propòsits:

- **Reforçar entitats que Whisper falla avui**: noms propis, sigles i estrangerismes.
- **Ampliar dades d'un domini o idioma**, sense partir de cap entitat concreta —
  p.ex. per a un idioma amb pocs recursos on l'objectiu és simplement tenir més parla.

## Entorn

```bash
python3 -m venv /home/ugiat/.virtualenvs/sintetic
source /home/ugiat/.virtualenvs/sintetic/bin/activate
pip install jiwer rapidfuzz num2words openai spacy ipykernel pytest
python -m spacy download es_core_news_md
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers omnivoice soxr soundfile librosa pedalboard pyroomacoustics datasets
ollama pull qwen3:14b        # alternativa local i gratuïta per a la fonètica
```

## Ordre d'execució

**Camí A — reforçar entitats:**

1. `python3 src/evaluate.py` → `entidades_erroneas.json`
2. `create_entities_list.ipynb` sencer → `entidades_fuente_{a,b}_validadas.json`
3. `dictionary.ipynb` amb `ETAPA='treball'` → `diccionari_fonetic_treball_*.json`
4. `python3 src/verify_entities.py` *(GPU)* → `entidades_verificades_roundtrip.json`,
   i tot seguit `--input lab/entitats/entidades_fuente_a_omitidas.json` per mesurar
   les entitats que Whisper omet del tot (mateix fitxer de sortida: s'acumulen).
   Fes-lo servir **amb els defectes** (3 veus × 4 frases × 3 condicions = 36 mesures);
   reduir-los fa que `tasa_error` només pugui prendre uns pocs valors i el llindar
   de 0.3 passi a ser 2 fallades de 6.
5. `create_entities_list.ipynb`, **només l'última cel·la** → `entidades_candidatas.json`
6. `dictionary.ipynb` amb `ETAPA='final'` → `diccionari_fonetic_final_*.json`
7. `generate_sentences.ipynb` → frases dirigides per entitats. La **Fase 3** no és
   opcional: sense ella `tts_text` encara és igual que `raw_text`.

**Camí B — ampliar dades genèriques**, independent del Camí A:

1. `generate_scenarios.ipynb` → frases sense entitats, per domini/idioma

**Comú a tots dos:** `generate_voices_environments.ipynb` → àudios finals. Llegeix
qualsevol `.jsonl` amb `raw_text`/`tts_text`/`style`, el de `generate_sentences.ipynb`
o el de `generate_scenarios.ipynb` (o tots dos combinats).

Al Camí A, la cel·la del pas 5 es pot re-executar sola: llegeix de fitxers, no del kernel.

`src/` només té funcions; l'orquestració viu als notebooks de `lab/`.

---

## Entitats — `create_entities_list.ipynb`

Reuneix les entitats que val la pena reforçar, de tres fonts:

- **Font A**: entitats del ground truth de RNE que Whisper transcriu malament.
- **Font B**: entitats trobades amb NER sobre un corpus de text.
- **Font C**: com de fragmentada queda una entitat pel tokenizer de Whisper; s'usa
  per ordenar candidats de Font B, no per seleccionar-los.

Un LLM confirma la grafia correcta de cada entitat i en descarta les que no ho són.

## Round-trip — `src/verify_entities.py`

Mesura si Whisper **realment** falla amb una entitat: la sintetitza amb TTS, la passa
pels entorns acústics de `acoustic_sim.py` i la transcriu amb Whisper. Escriu la taxa
d'error mesurada; el llindar per decidir quines entren al dataset s'aplica després,
a la darrera cel·la de `create_entities_list.ipynb`.

## Diccionari — `dictionary.ipynb`

Genera el diccionari fonètic: per a cada entitat, decideix amb un LLM si cal
reescriure-la perquè el TTS la pronunciï bé i, si cal, en genera la reescriptura.
Només hi entren els overrides necessaris; la resta d'entitats es llegeixen tal com
s'escriuen.

`src/phonetics.py` defineix el format del diccionari i la seva cache. `src/llm.py` és
el client, els lots i els prompts compartits pels notebooks que criden un LLM.

## Frases dirigides per entitats — `generate_sentences.ipynb`

Genera les frases del dataset a partir de les entitats seleccionades, en diversos estils
periodístics. Cada frase té dues versions: `raw_text` (ortografia real, el ground truth)
i `tts_text` (el que llegeix el TTS).

**El model escriu només `raw_text`.** `tts_text` surt d'aplicar el diccionari final sobre
`raw_text` (`phonetics.aplicar_diccionari`) i d'expandir xifres i símbols
(`phonetics.expandir_xifres`). Sense cap crida a cap model.

> **Per què.** Quan el model també escrivia la versió fonètica, de 317 fragments
> reescrits al dataset n'hi havia **40 amb dues o tres grafies diferents** (`Valencia` →
> `Balénsia` *i* `Valensia`; `Radio Nacional` amb tres formes) i unes quantes que
> canviaven la identitat de l'entitat (`Fernandes` → `Fernández`, `Millán` → `Milán`).
> Com que `raw_text` és el ground truth, cadascuna ensenya a Whisper a escriure una cosa
> quan en sent una altra. No és un problema de prompt: és que ningú decidia la fonètica
> **una sola vegada**. Injectar el diccionari al prompt com a glossari obligatori tampoc
> ho va resoldre — `FC Barcelona` seguia sortint com `F C Barcelona` en 22 de 29
> aparicions en comptes del `Fútbol Club Barcelona` revisat.

`tts_text` és una **funció pura** de `raw_text` i del diccionari: es reprodueix sense
tornar a cridar cap model, i la mateixa entitat sona igual a totes les frases per
construcció. Tres comprovacions finals ho verifiquen: cap entitat del diccionari en cru,
`tts_text` idèntic al resultat determinista, i cap frase d'una entitat que ja no és a la
llista seleccionada.

Una entitat que no és al diccionari es queda tal com s'escriu — que és el comportament
correcte, perquè `dictionary.ipynb` només hi posa overrides quan la grafia crua es
llegiria malament. **Qui decideix fonètica és `dictionary.ipynb`, i només ell.** Si al
dataset hi apareixen entitats secundàries que necessitarien override (clubs, sigles,
marques que el model introdueix pel camí), la manera d'arreglar-ho és afegir-les a la
llista d'entitats i tornar a passar el diccionari, no reescriure-les a la frase.

## Frases genèriques — `generate_scenarios.ipynb`

Amplia dades **sense partir de cap entitat**: genera escenaris del domini (p.ex.
"un corresponsal informa des d'una manifestació") i, per a cada un, frases diverses i
versemblants. Útil quan l'objectiu no és reforçar una llista d'entitats concretes sinó
tenir més parla d'un domini o idioma — el cas típic és un idioma amb pocs recursos.

`tts_text` només difereix de `raw_text` en les xifres i els símbols, que s'expandeixen
amb `phonetics.expandir_xifres` — el mateix pas determinista que la Fase 3 del Camí A.
Sense un diccionari auditat darrere no hi ha res més concret a corregir.

## Veus i entorns — `generate_voices_environments.ipynb`

Assigna una veu i un entorn acústic a cada frase i genera els àudios finals del
dataset, junt amb el manifest amb les condicions de cada un.

`src/acoustic_sim.py` és el simulador acústic: afegeix reverberació de sala, soroll
ambiental i degradació de canal a la veu neta del TTS. `src/build_noise_bank.py`
descarrega el banc de soroll ambiental que fa servir.

Les veus són sempre clonatge (`ref_audio`); l'`instruct` d'OmniVoice es va descartar
perquè en castellà sempre sortia amb accent llatinoamericà. Hi ha dues menes de
referència: **netes** (sense sala ni soroll, passen pel simulador sencer) i **ja
ambientades** (p.ex. VoxPopuli; l'entorn `font_real` les deixa tal com són per no
duplicar-hi condicions acústiques). `src/build_voice_bank.py` extreu referències
netes de Common Voice (gated a HuggingFace: cal acceptar la llicència un cop i tenir
`HF_TOKEN`).

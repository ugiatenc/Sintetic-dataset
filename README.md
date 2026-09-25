# Sintetic-dataset

Datasets sintètics d'àudio amb el seu ground truth per entrenar Whisper, en condicions
acústiques reals. Dos usos:

- **Castellà (RNE): reforçar entitats** que Whisper falla (noms propis, sigles,
  estrangerismes) amb frases dirigides.
- **Aranès: ampliar dades d'un idioma amb pocs recursos.** Pipeline complet a
  [`pipelines/aranes/`](pipelines/aranes/README.md).

El codi és comú; les dades van per idioma. `data/` és material de treball regenerable i
`datasets/` és el producte que s'entrega. Cap dels dos va al git, excepte els diccionaris
(`data/aranes/entitats/`) i la definició del test.

```
src/                         mòduls comuns: simulador acústic, bancs de veus i soroll, corpus,
                             porta de qualitat, reescriptura, fonètica, clients de LLM
pipelines/aranes/            un script per pas + config de l'aranès + entrenament
lab/                         notebooks del castellà i estudis (vegeu lab/README.md)
data/<idioma>/               text, entitats i àudio de treball
data/comu/entorns/           RIR i soroll
datasets/<idioma>/           el dataset i el test
```

## Entorn

```bash
python3 -m venv /home/ugiat/.virtualenvs/sintetic && source /home/ugiat/.virtualenvs/sintetic/bin/activate
pip install jiwer rapidfuzz num2words openai spacy ipykernel pytest
python -m spacy download es_core_news_md
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers omnivoice soxr soundfile librosa pedalboard pyroomacoustics datasets
```

L'entrenament (`pipelines/aranes/main_aranes.py`) necessita a més `peft` i `evaluate`, i
corre en un altre servidor. Els passos que classifiquen amb Claude fan servir el CLI de
Claude Code (`src/claude_cli.py`) i en guarden les respostes a `.cache_claude/`.

## Castellà: camí d'entitats

1. `python3 src/evaluate.py`: entitats del ground truth de RNE que Whisper transcriu malament.
2. `create_entities_list.ipynb`: entitats candidates de tres fonts (errors de RNE, NER sobre
   un corpus, fragmentació al tokenizer); un LLM en confirma la grafia.
3. `dictionary.ipynb` amb `ETAPA='treball'`: diccionari fonètic, només amb els overrides
   necessaris perquè el TTS pronunciï bé.
4. `python3 src/verify_entities.py` (GPU): TTS → entorn acústic → Whisper, per mesurar si
   Whisper falla de debò cada entitat. Amb els valors per defecte (36 mesures per entitat).
5. `create_entities_list.ipynb`, última cel·la: selecció final amb el llindar d'error.
6. `dictionary.ipynb` amb `ETAPA='final'`.
7. `generate_sentences.ipynb`: frases per entitat. El model escriu només el `raw_text`; el
   `tts_text` surt del diccionari i de l'expansió de xifres, sense cap model. Així una
   entitat sona igual a totes les frases i el ground truth no pot divergir.
8. `generate_voices_environments.ipynb`: veu clonada i entorn per frase, i els àudios.

Veus sempre per clonatge (l'`instruct` d'OmniVoice donava accent llatinoamericà); les
referències de Common Voice són gated a HuggingFace i demanen `HF_TOKEN`.

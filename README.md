# SINTETIC-DATASET

### ENTORN VIRTUAL

- Crear entorn
python3 -m venv /home/ugiat/.virtualenvs/sintetic

- Accedir entorn:
source /home/ugiat/.virtualenvs/sintetic/bin/activate

## PASSOS PER GENERAR EL DATASET SINTÈTIC

### 1. ENTITATS --------------------------------------------------------------------------------

`lab/create_entities_list.ipynb` produeix la llista d'entitats candidates combinant tres fonts:

- **Font A — errors reals** — compara les transcripcions de `large-v3` amb el ground truth (4 àudios RNE, `lab/entitats/entitats_fallades/evaluate.py`). L'alineament és per **regió** (una racha sencera de substitucions/esborrats), no token a token: alinear posició a posició partia una paraula que Whisper fragmenta i deixava l'entitat emparellada només amb l'últim tros (`radiogaceta` -> `gaceta`, similitud 0 amb l'entitat completa). Un embut de descartes automàtics (similitud grafia+fonètica pròpia, robusta a com Whisper segmenti les paraules; detecció de xifres amb `num2words`; noms comuns capitalitzats per posició) treu soroll i reconstrueix noms compostos multiparaula.
- **Font B — NER sobre corpus** (`spaCy` `es_core_news_md` sobre RTVE/Wikipedia) — generador de candidats. Es filtra per POS i límits de frase (`motivo_descarte_b`), i les formes curtes que són part d'una altra entitat de la mateixa llista (`Ronaldo` dins de `Cristiano Ronaldo`) es marquen com `forma_larga` per no gastar verificació dues vegades pel mateix referent.
- **Font C — fragmentació del tokenizer de Whisper** (`transformers.WhisperTokenizer`) — prior barat per ordenar els candidats de B (subtokens per caràcter).

A i B es validen amb un LLM (OpenAI `gpt-4o-mini`, *structured outputs*, `temperature=0`) que confirma la grafia i tipifica l'entitat (PERSONA/LLOC/ORG.../NO_ENTIDAD) donant-li la frase original com a context. **Sense `util_dataset`**: es va provar que aquest camp ("val la pena reforçar aquesta entitat?") decidia per fama, no per evidència — va rebutjar en ferm `Lamine Yamal`, `OSCE`, `carcacén`... entitats reals que el verificador round-trip mesura després que fallen de veritat. Qui decideix si val la pena és el round-trip, amb àudio, no el LLM endevinant. Un segon guardarraíl (`similitud_del_cambio` a `evaluate.py`) detecta quan el LLM "corregeix" la grafia substituint l'entitat per una altra de diferent en comptes d'arreglar-li l'accentuació (`Japoel Tel Aviv` -> `Maccabi Tel Aviv`, dos clubs israelians DISTINTS): mira només els trossos que canvien, no la similitud global, perquè les paraules compartides ("Tel Aviv") l'inflaven i amagaven la substitució.

**Font A sencera + Font B verificada** s'exporten a `lab/entitats/entidades_candidatas.json`, ordenada per prioritat.

#### Verificador round-trip

`src/verify_entities.py` mesura si **Whisper realment falla** en una entitat en comptes que ho decideixi un LLM endevinant. Sintetitza l'entitat en frases portadores neutres, les passa pels entorns acústics reals (`src/acoustic_sim.py`) i les transcriu amb el mateix Whisper `large-v3` que `evaluate.py`:

```bash
python3 src/verify_entities.py --limit 5    # prova ràpida
python3 src/verify_entities.py              # Font B sencera (validades + rebutjades que són entitat real)
```

**Cura amb `--batch-entities`** (per defecte 8): OmniVoice + Whisper `large-v3` en fp16 ja ocupen ~15 GiB d'una GPU de 16 GiB abans de processar res — en aquesta màquina el valor per defecte fa OOM. Baixa'l a 2-4 si tens una GPU similar.

Resultat mesurat: `Lamine Yamal` falla el 100% de les frases (`"la mina Yamal"`, `"lámina y a mal"`), `OSCE` el 67%. Sobre la Font B sencera (98 entitats mesurades), **24% tenen `tasa_error >= 0.3`**, incloent 8 amb fallada del 100% (`OSCE`, `CIDOB`, `Pete Hegseth`...) que `util_dataset` havia rebutjat abans d'existir aquest verificador. No decideix el llindar de tall — escriu `tasa_error` per entitat a `lab/entitats/entidades_verificades_roundtrip.json` com a evidència; `create_entities_list.ipynb` és qui aplica `UMBRAL_TASA_ERROR` per injectar-les a la llista final.

### 2. DICCIONARI DE PRONUNCIACIÓ --------------------------------------------------------------------------------

`lab/dictionary.ipynb` genera, per cada entitat, la seva transcripció fonètica perquè el TTS la llegeixi bé.

- Crida l'API d'OpenAI (`gpt-4o`) per lots, aplicant regles fixes: sigles deletrejades lletra a lletra ("UE" → "U E"), acrònims lèxics amb accentuació ajustada ("PSOE" → "Psoe"), respelling fonètic d'estrangerismes amb vocal de suport si el grup consonàntic és impronunciable ("Mbappé" → "Embapé"), i expansió de símbols/unitats ("ºC" → "grados Celsius").
- Les regles s'adapten a l'idioma objectiu (castellà/català/euskera...).
- Valida que la resposta conserva exactament les mateixes entitats i el mateix ordre de l'entrada, amb reintents amb backoff davant errors transitoris de l'API.

Sortida: `diccionari_fonetic_<idioma>.json` — **requereix revisió manual** abans de `generate_sentences.ipynb`.

### 3. FRASES --------------------------------------------------------------------------------

`lab/generate_sentences.ipynb` genera les frases sintètiques a partir del diccionari fonètic, en tres fases:

- **Fase 1 — context** — per cada entitat, un LLM (`gpt-4o-mini`) dedueix el tipus i un context periodístic breu (`contextos_auditables.json`, revisió manual).
- **Fase 2 — generació massiva** — per cada combinació entitat × estil (titular, crònica, entradilla, roda de premsa...) genera frases en paral·lel (`concurrent.futures`) amb dues versions: `raw_text` (ortogràfica) i `tts_text` (entitat substituïda per la fonètica, flexionada igual). Descarta frases que no contenen l'arrel de l'entitat o que són duplicades; és idempotent (es pot relanç sense duplicar el que ja hi ha).
- **Fase 3 — entitats secundàries** — tracta la resta d'entitats que apareixen "de rebot" (p.ex. "FC Barcelona" en una frase sobre "Lamine Yamal") i que la Fase 2 deixa intactes. Post-procés en tres passos sense tocar el prompt de generació: **3a** detecció amb LLM dels fragments impronunciables al `raw_text`; **3b** transcripció fonètica d'aquests candidats un únic cop (mateixes regles que `dictionary.ipynb`), per garantir consistència entre frases; **3c** aplicació determinista per find-and-replace sobre `tts_text` (sense LLM), amb una única regex d'alternança ordenada de més llarga a més curta per evitar efecte cascada.

Sortida: `lab/outputs/frases/dataset_entitats_whisper_final.jsonl`.

### 4. SIMULACIÓ ACÚSTICA I VEUS --------------------------------------------------------------------------------

`src/acoustic_sim.py` afegeix ambient i soroll de fons als àudios TTS, seguint l'ordre físic en què passen a la realitat:

```
veu seca -> [SALA] -> [+ SOROLL] -> [CANAL] -> normalització
             RIR       SNR (dB)     filtre/còdec
```

- **Sala** — convolució amb respostes impulsionals generades amb `pyroomacoustics` (6 sales × 6 variants); no cal descarregar cap banc de reverberacions.
- **Soroll** — mescla additiva a SNR controlada en dB, mesurant el nivell de veu només sobre les trames actives (fer-ho sobre el senyal sencer infla el soroll perquè els silencis baixen l'RMS). El banc ambiental es descarrega un sol cop amb `python3 src/build_noise_bank.py` (6 ambients de **DEMAND** via mirror de HuggingFace, CC BY 4.0, ~86 MB a `datasets/audios/entorns/soroll/`); si no hi és, el pipeline recorre a soroll procedural o a murmuri dels propis àudios TTS, sense bloquejar-se mai.
- **Canal** — cadenes de `pedalboard` amb còdecs reals: GSM 06.10 per al telèfon (banda 300-3400 Hz de veritat) i MP3 per a la connexió mòbil.

`lab/generate_voices_environments.ipynb` assigna veu, entorn i velocitat a cada frase de manera equilibrada (round-robin, no atzar pur) i coherent amb l'`style`, i genera els `.wav` finals. Requereix `pip install pedalboard pyroomacoustics` a l'entorn virtual.

Sortida: màster a 24 kHz + còpia a 16 kHz llesta per a Whisper, i `lab/outputs/converses/manifest_converses.jsonl` amb la SNR exacta, la RIR i el canal aplicats a cada àudio.


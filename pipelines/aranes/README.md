# Pipeline aranès

Dataset sintètic (àudio + ground truth) per fer fine-tuning de Whisper en aranès. Whisper
avui transcriu l'aranès amb **WER 0,91** però CER 0,36: sent els sons i no sap escriure'ls.
El dataset li ha d'ensenyar l'ortografia.

Com funciona: text aranès real → filtre dialectal i neteja → el text es reescriu en grafia
catalana perquè OmniVoice (`language="ca"`) el pronunciï bé → clonatge de veus de Common
Voice → entorns acústics d'Aran. El text que s'entrena és sempre l'aranès original.

Per què cada decisió, amb les mesures: [DECISIONS.md](DECISIONS.md).

## Estat (25/09/2026)

| | |
|---|---|
| Corpus | 261.998 frases · ~621 h (fonts fiables 108 h), verificat |
| Veus | 67 (47 occitanes de Common Voice + 20 catalanes), cap del test |
| Generació | en marxa, fonts fiables primer; ~360 h fetes el 25/09, ~645 h en acabar |
| Test | 448 clips reals · 0,63 h · 80 locutors, mai al train |
| Dev | 5 locutors reals (251 clips) + 1 de cada 200 frases sintètiques |
| Àudio real al train | 1.097 clips · 1,6 h |

## Els passos

| Script | Què fa | Surt a |
|---|---|---|
| `config.py` | Tot el que és específic de l'aranès: rutes, fonts, filtre, llindars, numerals, split | — |
| `01_corpus.py` | Baixa cada font un cop (`brut.jsonl`) i aplica el filtre dialectal | `data/aranes/text/corpus[_reserva]/<font>/` |
| `02_frases_claude.py` | Frases generades amb Claude per cobrir buits | `.../corpus/claude/` |
| `03_neteja.py` | Parteix les llargues, porta de qualitat, dedup | `<font>/net.jsonl` |
| `04_entitats.py` | Classifica sigles i noms propis amb Claude (amb context) | `data/aranes/entitats/` |
| `05_respelling.py` | Fusiona, deduplica i escriu el `tts_text` | `data/aranes/text/corpus.jsonl` |
| `06_veus.py` | Banc de veus de Common Voice, amb prova de síntesi | `data/aranes/audio/veus/` |
| `06c_cv_real.py` | Clips reals de Common Voice per al train i el dev | `datasets/aranes/dataset_aranes*/commonvoice*/` |
| `07_entorns.py` | Soroll (DEMAND) i sales d'Aran (RIR) | `data/comu/entorns/` |
| `08_tts.py` | Pla determinista i generació amb OmniVoice + simulador | `datasets/aranes/dataset_aranes[_eval]/` |
| `08b_estructura_clips.py` | Recol·loca els clips si canvien les regles de carpeta o de split | idem |
| `09_dataset.py` | `train.jsonl`, `dev.jsonl` i sessions de ~28 s | `datasets/aranes/` |
| `10_auditoria.py` | Comprovacions dures (test, dev, `tts_text`, durades) | `datasets/aranes/AUDITORIA.md` |
| `main_aranes.py` | Fine-tuning LoRA de Whisper large-v3 (al servidor d'entrenament) | `output_whisper_aranes/` |

El test es construeix amb `lab/proves_inicials/aranes/00_baixar_test_set.py`. `entorns.py`
defineix les sales i sorolls d'Aran; el codi comú a tots els idiomes és a `src/`.

## Executar-ho

```bash
cd pipelines/aranes
python3 01_corpus.py --incloure-poc-fiables   # ~1,5 h el primer cop; després minuts (--rebaixar per tornar a baixar)
python3 03_neteja.py --incloure-reserva
python3 04_entitats.py                        # crides a Claude, amb cache a .cache_claude/
python3 05_respelling.py
python3 06_veus.py --baixar && python3 06_veus.py --afegir-catalanes 20
python3 06c_cv_real.py
python3 07_entorns.py
python3 08_tts.py --planificar                # fonts fiables primer; --ordre barrejat per a una mostra proporcional
python3 08_tts.py --generar                   # reprenible; --limit 200 per a un pilot
python3 09_dataset.py && python3 10_auditoria.py
```

La generació llarga es llança amb `datasets/aranes/genera.sh` (reintenta si el procés cau;
log a `datasets/aranes/generacio.log`). Un canvi a les regles de text no demana tornar a
baixar res: de `01` a `05` des de `brut.jsonl` són uns 14 minuts.

## Què surt i on

```
data/aranes/                      material de treball, regenerable (no va al git, excepte entitats/)
  text/corpus/<font>/             fonts fiables: brut, capturat, net, descartades (amb motiu i pas)
  text/corpus_reserva/<font>/     BSC i Viquipèdia
  text/corpus.jsonl               el corpus final: raw_text (ground truth) + tts_text
  entitats/                       normes ortogràfiques i diccionaris de sigles, noms i plurals
  audio/                          clips crus de Common Voice, reserva catalana i el banc de veus
data/comu/entorns/                RIR i soroll, comuns a tots els idiomes

datasets/aranes/                  el producte (vegeu el seu README)
  dataset_aranes/<font>_<bloc>/   train: <id>.wav + <id>.json, blocs de 4.000
  dataset_aranes_eval/<font>/     dev
  test/                           avaluació final, mai al train
  train.jsonl  dev.jsonl  manifest.jsonl  pla.jsonl
```

Correccions a mà: un `correccions.jsonl` a la carpeta de la font
(`{"id": ..., "raw_text": ...}` o `{"id": ..., "elimina": true}`), que el pas 5 aplica. No
s'editen `capturat.jsonl` ni `net.jsonl`: es regeneren.

## Entrenar

`main_aranes.py` és l'script d'entrenament del projecte RNE amb els canvis per a l'aranès
marcats `#canviat`: llegeix el camp `text`, idioma `Occitan`, LoRA r=32 també a les capes
`fc`, clips reals ×5, WER i CER sobre text normalitzat, avaluació cada 500 passos sobre el
dev. Al servidor es copien `dataset_aranes/`, `dataset_aranes_eval/` i `test/` amb rsync
(amb `--delete` a les dues primeres). El número final és el del `test/`, sobre el model
original i l'entrenat.

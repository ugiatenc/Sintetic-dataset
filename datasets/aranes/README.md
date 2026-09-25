# `datasets/aranes/`: el dataset per entrenar

Al servidor d'entrenament es copien `dataset_aranes/` (train), `dataset_aranes_eval/` (dev)
i `test/`. Les llistes `train.jsonl` i `dev.jsonl` són opcionals: l'script d'entrenament
carrega per carpetes.

## Train: `dataset_aranes/<font>_<bloc>/`

```
dataset_aranes/
  arannoticies_000000-003999/
      arannoticies-0000001.wav      16 kHz, mono, PCM 16 bits
      arannoticies-0000001.json     el registre del clip
  conselharan_…/  bsc_ca_arn_…/  institut_estudis_…/  wikipedia_oc_…/  claude_…/
  commonvoice_000000-003999/        els clips REALS de Common Voice, mateix format
```

La carpeta surt de l'id: `bsc_ca_arn-0122293` va a `bsc_ca_arn_120000-123999`, i els
trossos d'una frase partida (`…-0121269-2`) van amb el seu índex base. Hi ha forats a la
numeració: són les frases que la neteja va descartar.

Del `.json` només calen dos camps per entrenar:

| Camp | Què és |
|---|---|
| `text` (i `transcript`, idèntic) | **el ground truth**: l'aranès real, tal com s'ha d'escriure |
| `audio` | ruta del wav relativa a l'arrel del repositori |

La resta és per auditar: `tts_text` (el que va llegir el TTS, en grafia catalana; **no** és
el ground truth), `font`, `veu_id`, `entorn_id`, `snr_db`, `canal`, `speed`, `durada_s`,
`split`… Els clips reals porten `entorn_id: real` i `veu_id: cv_<locutor>`.

## Dev: `dataset_aranes_eval/<font>/`

Per triar el millor checkpoint, fora del train. `commonvoice/`: 251 clips reals de 5
locutors reservats, amb frases que no surten al train. La resta de carpetes: 1 de cada 200
frases sintètiques. La veu clonada de 4 d'aquells 5 locutors sí que és al train llegint
altres frases; per això el número honest és el del test.

## Test: `test/`

448 clips reals, 80 locutors, cap frase ni locutor al train. Vegeu el seu README.

## Manifests

- `manifest.jsonl`: una línia per frase del pla ja processada. `status: ok` té àudio;
  `massa_llarg` (> 29 s) i `exclos` (la frase és al dev) no en tenen.
- `cv_real/manifest.jsonl`: els clips reals, amb `split` train, dev o exclos.
- `pla.jsonl`: el pla de generació sencer, fet abans de generar.
- `train.jsonl`, `dev.jsonl`: `audio`, `text`, `duration`, `speaker`, `entorn`, `font`,
  `split`, generats pel pas 9. `AUDITORIA.md`: el resultat del pas 10.

Mentre la generació corre el conjunt creix; qualsevol moment és un dataset vàlid.

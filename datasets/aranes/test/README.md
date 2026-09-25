# Test real d'aranès (Common Voice `oc`)

448 clips · 0,63 h · 80 locutors. Àudio real amb transcripció validada per persones: és
l'únic WER de debò del projecte, i es reserva per comparar el model original amb
l'entrenat. Cap frase d'aquí entra al corpus i cap locutor d'aquí entra al train ni al
banc de veus; el pas 10 ho comprova.

Construït amb `lab/proves_inicials/aranes/00_baixar_test_set.py` a partir dels splits
`test`, `dev` i `train` de Common Voice 22.0 (`fsicoli/common_voice_22_0`): els clips que
declaren accent aranès o el text dels quals passa el filtre dialectal.

El 21/09 se'n van treure 2 locutors (24 clips) que tenien molt més àudio a `other` que al
test, perquè aquell àudio pogués entrar al train (`config.LOCUTORS_FORA_DEL_TEST`). Els WER
mesurats abans d'aquella data, sobre 472 clips, no són comparables.

`manifest.jsonl`: `audio` (relativa a aquesta carpeta), `text`, `client_id`, `split`,
`accents`, `durada_s`. Cada wav de `clips/` té al costat un `.json` amb el mateix registre,
per llegir-lo amb el mateix carregador que el train.

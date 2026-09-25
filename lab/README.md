# lab/

Proves i estudis que han decidit coses del pipeline. El resultat de cadascun, en poques
línies; el context complet és a [`pipelines/aranes/DECISIONS.md`](../pipelines/aranes/DECISIONS.md).
Els àudios que generen per escoltar no van al git.

## Castellà: notebooks del camí d'entitats

`create_entities_list.ipynb`, `dictionary.ipynb`, `generate_sentences.ipynb` i
`generate_voices_environments.ipynb`. L'ordre i què fa cadascun és al
[README general](../README.md).

## Aranès

**`proves_inicials/aranes/`**: línia base, 14-16/09. Whisper large-v3-turbo sobre àudio real
d'aranès: WER 0,91, CER 0,36; forçar `oc` o `ca` dona el mateix. OmniVoice amb `oc` sona
afrancesat i amb `ca` massa català, i per això el pipeline genera amb `ca` sobre text
reescrit. `00_baixar_test_set.py` construeix el test (`datasets/aranes/test/`) i
`03_provar_whisper.py` en mesura el WER/CER per idioma forçat.

**`ablacio_respelling/`**: quina regla de reescriptura acosta l'àudio a l'aranès. 115
frases, 4 veus, bootstrap aparellat: cap regla mou la probabilitat de català de manera
mesurable (totes juntes, −0,012 [−0,065, +0,041]). Es mantenen perquè corregeixen la
pronunciació. `-n` → `-ng` es va descartar: Whisper hi sentia una `g` en 25 de 115 frases.
Resultats a `resultats_ablacio.json` i `resultats_ng.json`.

**`estudi_o_u/`**: com diuen la `o` els locutors reals (alineament forçat MMS + reconeixedor
de fonemes wav2vec2 + formants). [u] en el 77 % de les `o`/`ó`, 84 % en mots funcionals;
la `ò` és [ɔ] el 96 %. La regla `o` → `u` és correcta. Informe a `informe_o_u.txt`.

**`banc_veus/`**: 145 locutors de Common Voice `oc` i 5 catalans clonats amb OmniVoice sobre
dues frases i passats per Whisper. L'accent declarat no canvia res (CER contra el consens
0,109 declarats, 0,087 no declarats, 0,092 catalans) i tres veus amb bona nota donaven
galimaties: d'aquí la prova de síntesi del pas 6. `construeix_banc.py` sintetitza,
`avalua_whisper.py` mesura i `fes_readme.py` refà les carpetes per escoltar a partir de
`banc.jsonl`.

**`revisio_generacio/`**: revisió objectiva d'una mostra de clips generats per veu i per
entorn (CER de Whisper, similitud ECAPA amb la referència, senyal). Sobre 804 clips: CER
mediana 0,18, 1 % de galimaties, cap veu ni entorn que falli. `revisa_generacio.py` es pot
tornar a passar quan vulguis.

**`verifica_corpus_aranes.py`**: les 130 comprovacions del corpus final (xifres, neteja,
respelling, entitats, cap frase del test). Totes correctes el 21/09.

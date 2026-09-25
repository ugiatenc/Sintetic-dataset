import random
import torch
import datasets
import evaluate

# from datasets import Dataset
from datasets import IterableDataset, Features, Audio, Value
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    # TaskType,
    # prepare_model_for_kbit_training,
)
import json
import re
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import warnings

warnings.filterwarnings("ignore")  # Global suppression

REAL_OVERSAMPLE = 5  #canviat: cada clip REAL de Common Voice (carpeta commonvoice*) entra 5 cops al train; son l'1 % i son l'unic aranes real


def get_all_files(dataset_dir):
    """Get all file pairs first for splitting"""
    file_pairs = []
    for first_tier_dir in sorted(Path(dataset_dir).iterdir()):
        if not first_tier_dir.is_dir():
            continue
        for json_file in sorted(first_tier_dir.glob("*.json")):
            audio_file = json_file.with_suffix(".wav")
            if audio_file.exists():
                file_pairs.append((json_file, audio_file))
            else:
                audio_file = Path(json_file.parent) / "output.wav"
                if audio_file.exists():
                    file_pairs.append((json_file, audio_file))
    return file_pairs


def clean_transcription(text_raw):
    """
    Limpia el texto de transcripción:
    - Elimina [!Música] y variantes dondequiera que aparezcan
    - Limpia espacios dobles resultantes
    - Devuelve string vacío "" si solo había música
    - Devuelve el texto limpio si había voz con música intercalada
    """
    text = text_raw.split('\t')[0].strip()

    # Eliminar [!Música] y variantes (case-insensitive, con o sin tilde)
    text = re.sub(r'\[!M[uú]sica\]', '', text, flags=re.IGNORECASE)

    # Por si hay otras etiquetas similares que quieras tratar igual
    # text = re.sub(r'\[!Sinton[ií]a\]', '', text, flags=re.IGNORECASE)
    # text = re.sub(r'\[!Efecto\]', '', text, flags=re.IGNORECASE)

    # Limpiar espacios dobles y puntuación suelta que pueda quedar
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_local_iterable_dataset_with_splits(
    data_dir, train_ratio=0.7, seed=42
):
    """
    Create train/validation IterableDatasets with splitting
    """

    def data_generator(file_pairs, start_idx=0, end_idx=None):
        """Generator for a specific subset of files"""
        if end_idx is None:
            end_idx = len(file_pairs)

        for i in range(start_idx, end_idx):
            json_file, audio_file = file_pairs[i]
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                text_raw = data.get("text", data.get("transcript", "")).strip()  #canviat: el ground truth aranes es al camp `text` (mai `tts_text`)

            transcription_text = clean_transcription(text_raw)

            # Casos después de limpiar:
            # - "" → segmento de música pura → lo pasamos con label vacío
            #         el modelo aprende: música = no generar nada
            # - "texto limpio" → voz (con o sin música de fondo) → label correcto
            # Solo saltamos si text_raw era vacío desde el origen (sample sin datos)
            if text_raw == "" or text_raw.split('\t')[0].strip() == "":
                continue

            yield {"audio": str(audio_file), "text": transcription_text, "path_id": str(audio_file)}

    # Define features schema
    features = Features(
        {"audio": Audio(sampling_rate=16000), "text": Value("string"), "path_id": Value("string")}
    )

    # Get all files and shuffle for splitting
    all_files = get_all_files(data_dir)

    # Simple deterministic shuffle based on filename hash
    import hashlib

    all_files.sort(key=lambda x: hashlib.md5(str(x[0]).encode()).hexdigest())

    # Split into train/val
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    #canviat: sobremostreig dels clips reals nomes al train (l'eval es queda intacte) i barreja amb llavor fixa
    reals = [p for p in train_files if p[0].parent.name.startswith("commonvoice")]  #canviat
    train_files = train_files + reals * (REAL_OVERSAMPLE - 1)  #canviat
    random.Random(seed).shuffle(train_files)  #canviat

    # Create datasets with explicit features
    train_dataset = IterableDataset.from_generator(
        lambda: data_generator(train_files), features=features
    )

    val_dataset = IterableDataset.from_generator(
        lambda: data_generator(val_files), features=features
    )

    return train_dataset, val_dataset


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = batch["text"]
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # Process features
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Process labels
        label_list = [feature["labels"] for feature in features]
        labels_batch = self.processor.tokenizer(
            label_list,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["decoder_attention_mask"] = labels_batch["attention_mask"]
        return batch


def save_samples_to_disk(dataset, num_samples, output_dir):
    """
    Save a subset of samples from the dataset to disk.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Save audio to file
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        audio_path = f"{output_dir}/sample_{i}_audio.wav"

        # Save wav and text data
        sf.write(audio_path, audio_array, sampling_rate)
        sample_info = {
            "audio_path": audio_path,
            "text": sample["text"],
            "id": sample.get("id", None),
        }
        samples.append(sample_info)

    with open(f"{output_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


# Afegim filtre de seguretat de tokens
def filter_long_samples(example):
    labels = tokenizer(example["text"]).input_ids
    # Permitimos vacío (música pura) pero filtramos muestras corruptas largas
    return len(labels) < 200


def normalize_for_metrics(text):  #canviat: WER/CER sense majuscules ni puntuacio, i amb l'apostrof unificat (el test de Common Voice porta ’ i el train ')
    text = text.lower().replace("’", "'").replace("‘", "'")  #canviat
    text = re.sub(r"[^\w\s']", " ", text)  #canviat
    return re.sub(r"\s+", " ", text).strip()  #canviat


if __name__ == "__main__":
    # Whisper model
    openai_model_id = "openai/whisper-large-v3"
    model_id = openai_model_id

    # Training directories: dataset principal para train, dataset separado para test/eval
    dataset_train_dir = "datasets/dataset_aranes/dataset_aranes_1"  #canviat: train aranes (carpetes <font>_<bloc>/ amb wav + json)
    dataset_eval_dir = "datasets/dataset_aranes/dataset_aranes_eval_1"  #canviat: dev (5 locutors reals + 1/200 sintetiques); el test/ es reserva per a la comparacio final
    out_dir = "output_whisper_aranes/training_aranes"  #canviat
    model_dir = "output_whisper_aranes/model_aranes"  #canviat

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Training arguments
    epochs = 2
    batch_size = 2
    gradient_accumulation_steps = 16  #canviat: batch efectiu 32 (abans 16), mes estable amb LoRA r=32 sobre 138k clips
    checkpoint_step = int(model_id.split("-")[-1]) if "checkpoint" in model_id else 0
    # Steps totales basados SOLO en el dataset de train (se usa al 100%)
    all_train_files = get_all_files(dataset_train_dir)  #canviat
    total_train_files = len(all_train_files) + (REAL_OVERSAMPLE - 1) * sum(1 for p in all_train_files if p[0].parent.name.startswith("commonvoice"))  #canviat: compta les copies del sobremostreig
    steps_per_run = epochs * (total_train_files // (batch_size * gradient_accumulation_steps))
    max_steps = checkpoint_step + steps_per_run
    print("Max steps:", max_steps)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        openai_model_id
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        openai_model_id, language="Occitan", task="transcribe"  #canviat: token <|oc|>; Whisper no te aranes i l'occita es la seva llengua
    )
    processor = WhisperProcessor.from_pretrained(
        openai_model_id, language="Occitan", task="transcribe"  #canviat
    )

    # Carga separada: 100% del dataset principal para train (ratio 1.0)
    train_dataset, _ = load_local_iterable_dataset_with_splits(
        dataset_train_dir, train_ratio=1.0
    )
    # 100% del dataset de test para eval (ratio 0.0 deja todo en validación)
    _, eval_dataset = load_local_iterable_dataset_with_splits(
        dataset_eval_dir, train_ratio=0.0
    )

    # Aplicar filtro de longitud (permite vacíos de música, filtra corruptos)
    train_dataset = train_dataset.filter(filter_long_samples)
    eval_dataset = eval_dataset.filter(filter_long_samples)

    # Apply preprocessing to datasets
    rne_dataset_train = train_dataset.map(prepare_dataset)
    # Se procesa TODO el dataset de test (sin .take() artificial)
    rne_dataset_valid = eval_dataset.map(prepare_dataset)

    # Load the Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(openai_model_id)
    model.generation_config.language = "occitan"  #canviat
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None


    lora_config = LoraConfig(
        r=32,  #canviat: mes capacitat: el decoder ha d'aprendre una ortografia que no ha vist mai (abans 16)
        lora_alpha=64,  #canviat: 2*r
        target_modules=[
            "q_proj",
            "k_proj",  #canviat
            "v_proj",
            "out_proj",
            "fc1",  #canviat: les capes fc son on viu el "vocabulari" del decoder
            "fc2",  #canviat
        ],  # Apply LoRA to attention projections. More efficient, small datasets.
        lora_dropout=0.05,  #canviat: 350 h de dades, menys regularitzacio (abans 0.1)
        bias="none",  # Don't adapt bias terms
    )

    """ LORA CONFIGURATION FOR STRONGER ADAPTATION """
    """
    lora_config = LoraConfig(
         r=32,
         lora_alpha=64,
         target_modules=[
             "q_proj",
             "v_proj",
             "k_proj",
             "out_proj",
             "fc1",
             "fc2",
         ],
         lora_dropout=0.05,
         bias="none",
     )
    """

    # Apply LoRA to model — load from checkpoint if resuming, otherwise fresh LoRA
    if "checkpoint" in model_id:
        model = PeftModel.from_pretrained(model, model_id, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Instantiate the data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Defining evaluation metrics
    metric = evaluate.load("wer")
    metric_cer = evaluate.load("cer")  #canviat: el problema de Whisper amb l'aranes es ortografic; el CER ho mesura millor que el WER

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_for_metrics(p) for p in pred_str]  #canviat
        label_str = [normalize_for_metrics(r) for r in label_str]  #canviat

        # Filter out pairs where the reference is empty to avoid ZeroDivisionError in WER
        filtered = [(p, r) for p, r in zip(pred_str, label_str) if r.strip()]
        if not filtered:
            return {"wer": float("inf")}
        pred_str_filtered, label_str_filtered = zip(*filtered)

        wer = 100 * metric.compute(predictions=list(pred_str_filtered), references=list(label_str_filtered))
        cer = 100 * metric_cer.compute(predictions=list(pred_str_filtered), references=list(label_str_filtered))  #canviat

        return {"wer": wer, "cer": cer}  #canviat

    # Create the Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,  #canviat: l'eval nomes genera, hi cap un lot mes gran i va 4x mes rapid (927 clips de dev)
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-4, #1e-5, #2e-5,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=500,  #canviat: ~8.600 passos en total; cada 500 dona prou punts per triar el millor checkpoint
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,  #canviat: ha d'anar amb eval_steps (load_best_model_at_end)
        eval_on_start=True,  #canviat: WER/CER del large-v3 sense entrenar, amb <|oc|>, com a punt de partida al mateix log
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        save_total_limit=20,
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        ignore_data_skip=True,
        ddp_find_unused_parameters=False,
        skip_memory_metrics=True,
        eval_accumulation_steps=10,
        label_names=["labels"]
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=rne_dataset_train,
        eval_dataset=rne_dataset_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=model_id if "checkpoint" in model_id else None)

    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

    print("Training complete. Saving model...")

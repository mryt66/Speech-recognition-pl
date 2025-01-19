import os
import time
import torch
import torchaudio
import pandas as pd
import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred, processor):
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [p.lower() for p in pred_str]
    label_str = [l.lower() for l in label_str]
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"eval_wer": wer}

def transcribe_samples(model, processor, samples):
    results = []
    times = []
    predictions = []
    references = []
    wer_list = []

    for sample in samples:
        try:
            waveform, sample_rate = torchaudio.load(sample["file_path"])
            if sample_rate != processor.feature_extractor.sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, processor.feature_extractor.sampling_rate)
                waveform = resampler(waveform)
            waveform = waveform.squeeze(0)
            input_features = processor(waveform, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt").input_features
            reference_str = sample["reference"]
            start_time = time.time()
            pred_ids = model.generate(input_features, attention_mask=torch.ones_like(input_features))
            pred_str = processor.decode(pred_ids[0], skip_special_tokens=True)
            end_time = time.time()
            predictions.append(pred_str)
            references.append(reference_str)
            times.append(end_time - start_time)
            pred = {
                "predictions": processor.tokenizer(pred_str, return_tensors="pt", padding=True, truncation=True).input_ids,
                "label_ids": processor.tokenizer(reference_str, return_tensors="pt", padding=True, truncation=True).input_ids,
            }
            metrics = compute_metrics(pred, processor)
            wer_list.append(metrics["eval_wer"])
            results.append({"file_name": os.path.basename(sample["file_path"]), "transcription": pred_str})
        except Exception as e:
            print(f"Error processing {sample['file_path']}: {e}")

    return results

def evaluate_model(model_name, samples):
    processor = WhisperProcessor.from_pretrained(model_name, language="pl", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    transcriptions = transcribe_samples(model, processor, samples)
    return transcriptions

if __name__ == "__main__":
    model_name = sys.argv[1]
    num_samples = int(sys.argv[2])
    audio_folder = "clips"
    data = pd.read_csv("validated.tsv", sep="\t")
    samples = []
    for index, row in data.head(num_samples).iterrows():
        file_path = os.path.join(audio_folder, row["path"])
        if os.path.exists(file_path):
            samples.append({"file_path": file_path, "reference": row["sentence"]})
    transcriptions = evaluate_model(model_name, samples)
    for transcription in transcriptions:
        print(f"Filename: {transcription['file_name']}, Transcription: {transcription['transcription']}")
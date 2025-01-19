import sys
import torch
import torchaudio
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_audio(file_path, model_name):
    processor = WhisperProcessor.from_pretrained(model_name, language="pl", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != processor.feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, processor.feature_extractor.sampling_rate)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0)
    input_features = processor(waveform, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt").input_features
    pred_ids = model.generate(input_features, attention_mask=torch.ones_like(input_features))
    transcription = processor.decode(pred_ids[0], skip_special_tokens=True)
    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <audio_file> <model_name>")
        sys.exit(1)
    audio_file = sys.argv[1]
    model_name = sys.argv[2]
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found.")
        sys.exit(1)
    transcription = transcribe_audio(audio_file, model_name)
    print(f"Transcription: {transcription}")

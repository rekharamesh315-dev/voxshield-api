import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

SAMPLE_RATE = 16000

# ---------- Model definition (must match training) ----------
class VoiceCloneDetector(nn.Module):
    def __init__(self, n_mels=80, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, mel):
        x = self.conv(mel.unsqueeze(1))
        return self.classifier(x.flatten(1)).squeeze(-1)

# ---------- Feature extraction ----------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160, n_mels=80
)
db_transform = torchaudio.transforms.AmplitudeToDB()

def logmel(wav):
    mel = mel_transform(wav.unsqueeze(0))
    return db_transform(mel).squeeze(0)

def pad_or_crop_logmel(mel, target_frames=300):
    n_mels, frames = mel.shape
    if frames >= target_frames:
        start = (frames - target_frames) // 2
        return mel[:, start:start + target_frames]
    return F.pad(mel, (0, target_frames - frames))

# ---------- Load model ----------
device = "cpu"
model = VoiceCloneDetector().to(device)
model.load_state_dict(torch.load("voice_clone_detector.pt", map_location=device))
model.eval()

# ---------- Streaming risk scorer ----------
def analyze_audio(wav, sr, window_sec=3.0, stride_sec=1.0,
                   medium_threshold=0.45, high_threshold=0.75):
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    window_samples = int(window_sec * SAMPLE_RATE)
    stride_samples = int(stride_sec * SAMPLE_RATE)

    duration = round(wav.shape[0] / SAMPLE_RATE, 1)
    if duration < window_sec:
        return {"error": f"Audio too short ({duration}s). Needs at least {window_sec}s."}

    scores = []
    pos = 0
    with torch.no_grad():
        while pos + window_samples <= wav.shape[0]:
            chunk = wav[pos: pos + window_samples]
            mel = pad_or_crop_logmel(logmel(chunk)).unsqueeze(0)
            prob = torch.sigmoid(model(mel)).item()
            scores.append(prob)
            pos += stride_samples

    if not scores:
        return {"error": "Could not process audio."}

    # smoothed = simple moving average over all windows
    smoothed = sum(scores) / len(scores)
    risk_score = round(smoothed * 100, 1)
    level = "HIGH" if smoothed >= high_threshold else "MEDIUM" if smoothed >= medium_threshold else "LOW"
    confidence = round(max(risk_score, 100 - risk_score), 1)

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "confidence": confidence,
        "duration_sec": duration,
        "num_windows": len(scores),
        "window_scores": [round(s * 100, 1) for s in scores],
    }

# ---------- FastAPI app ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "VoxShield AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    result = analyze_audio(wav, sr)
    return result

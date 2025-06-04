# voice_model.py

import numpy as np
import torch
import librosa
from scipy.spatial.distance import cosine
from mel_model import MelEncoder

def extract_mel(wav_path, sr=16000, n_mels=80):
    y, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # shape: (T, mel)

def load_model(model_path="contrastive_mel_encoder.pth"):
    model = MelEncoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def get_embedding(model, mel):
    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        embedding = model(mel_tensor)
    return embedding.squeeze(0).numpy()

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def normalize(vec):
    return vec / np.linalg.norm(vec)

def evaluate_wav(wav_path, model_path="contrastive_mel_encoder.pth", l1_embeddings_path="L1_embeddings.npy"):
    model = load_model(model_path)
    user_mel = extract_mel(wav_path)
    user_embedding = normalize(get_embedding(model, user_mel))

    L1_embeddings = np.load(l1_embeddings_path)
    L1_embeddings = np.array([normalize(e) for e in L1_embeddings])

    # 평균 기반 비교
    user_sim = cosine_similarity(user_embedding, L1_embeddings.mean(axis=0))
    similarities = np.array([cosine_similarity(user_embedding, emb) for emb in L1_embeddings])

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    z_score = (user_sim - mean_sim) / std_sim

    if z_score > 1.0:
        grade = "A"
    elif z_score > -0.5:
        grade = "B"
    else:
        grade = "C"
    z_score = round(z_score, 2)
    user_score = round(user_sim,4)
    return (generate_feedback(grade))
    #round(user_sim, 4),
        # "z_score": round(z_score, 2),
        # "grade": grade,
        # "feedback": generate_feedback(grade)

def generate_feedback(grade):
    if grade == "A":
        return "발화 억양이 매우 자연스럽고 원어민 스타일과 유사합니다."
    elif grade == "B":
        return "대체로 자연스럽지만 억양 흐름에서 약간의 부자연스러움이 감지됩니다."
    else:
        return "억양이 원어민 스타일과 상당히 다르며 리듬이나 강세의 조정이 필요합니다."

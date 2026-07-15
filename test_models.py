import torch
import librosa

def _load_audio(audio_path, sr = 22050):
    """Helper to load audio file using librosa."""
    return librosa.load(audio_path, sr=sr, mono=True)[0]

test_audio_path = "sax.mp3"

from fadtk import CLAPLaionModel
clap_type = "audio"
model = CLAPLaionModel(type=clap_type)
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
audio_embedding = torch.mean(embedding, dim=0)

print("------------------------------")
print("[INFO] Laion CLAP Audio Model Checked")
print("------------------------------")

clap_type = "music"
model = CLAPLaionModel(type=clap_type)
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
audio_embedding = torch.mean(embedding, dim=0)

print("------------------------------")
print("[INFO] Laion CLAP Music Model Checked")
print("------------------------------")

from fadtk import CLAPModel
model = CLAPModel(type='2023')
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
audio_embedding = torch.mean(embedding, dim=0)

print("------------------------------")
print("[INFO] Microsoft CLAP Model Checked")
print("------------------------------")

from fadtk import MERTModel
model = MERTModel(size='v1-95M')
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
print(embedding.shape)
audio_embedding = torch.mean(embedding, dim=0)
print(audio_embedding.shape)

print("------------------------------")
print("[INFO] MERT v1-95M Model Checked")
print("------------------------------")

from fadtk import MERTModel
model = MERTModel(size='v1-330M')
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
print(embedding.shape)
audio_embedding = torch.mean(embedding, dim=0)
print(audio_embedding.shape)

print("------------------------------")
print("[INFO] MERT v1-330M Model Checked")
print("------------------------------")

from fadtk import MERTModel
sr = 16000
model = MERTModel(size='v0-public')
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
audio_embedding = torch.mean(embedding, dim=0)

print("------------------------------")
print("[INFO] MERT v0-public Model Checked")
print("------------------------------")

from fadtk import VGGishModel
model = VGGishModel()
model.load_model()

audio = _load_audio(test_audio_path, sr = model.sr)
embedding = model._get_embedding(audio)
audio_embedding = torch.mean(embedding, dim=0)

print("------------------------------")
print("[INFO] VGGish Model Checked")
print("------------------------------")

from fadtk import CdpamModel
model = CdpamModel(mode='acoustic')
model.load_model()

audio = CdpamModel.load_wav(test_audio_path)
embedding = model._get_embedding(audio)
print(embedding.shape)
audio_embedding = torch.mean(embedding, dim=0)
print(audio_embedding.shape)

print("------------------------------")
print("[INFO] Cdpam Model Checked")
print("------------------------------")
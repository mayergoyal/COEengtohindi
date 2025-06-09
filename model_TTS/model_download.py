from TTS.api import TTS

print("Downloading model and caching locally...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
print("Model downloaded and cached!")

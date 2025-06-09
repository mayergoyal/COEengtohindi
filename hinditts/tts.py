import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS

# ✅ Allow PyTorch to unpickle XTTS configs safely
add_safe_globals([XttsConfig, XttsAudioConfig])
model_path="C:\\Users\\Mayer\\OneDrive\\Desktop\\coe\\hinditts\\models\\tts_models--multilingual--multi-dataset--xtts_v2"
configpath="C:\\Users\\Mayer\\OneDrive\\Desktop\\coe\\hinditts\\models\\tts_models--multilingual--multi-dataset--xtts_v2\\config.json"
# ✅ Load the model
#tts = TTS(model_path=model_path,config_path=configpath, progress_bar=False)
#tts.config.use_language_conditioning = True
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

with open("C:\\Users\\Mayer\\OneDrive\\Desktop\\coe\\hinditext.txt","r", encoding='utf-8') as file:
    text = file.read()
    print(text)
print("Model languages:", tts.languages)
print("Is multilingual?", tts.is_multi_lingual)

# ✅ Generate speech from text and speaker wav
tts.tts_to_file(
    text=text,
    speaker_wav="cv-corpus-7.0-2021-07-21/hi/clipswav/common_voice_hi_23795360.wav",
    language="hi",
    file_path="output.wav"
)

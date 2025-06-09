import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS


def generate_tts(text, speaker_wav_path, output_wav_path):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language="hi",
        file_path=output_wav_path
    )
    print(f"TTS audio saved at: {output_wav_path}")

if __name__ == "__main__":
    print("model_run.py loaded correctly")
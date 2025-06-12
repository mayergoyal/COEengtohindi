import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS
import os


def generate_tts(text, speaker_wav_path, output_wav_path):
    if not text or not text.strip():
        raise ValueError("TTS input text is empty. Please provide valid text.")
    if not speaker_wav_path or not os.path.isfile(speaker_wav_path):
        raise FileNotFoundError(f"Speaker WAV file '{speaker_wav_path}' does not exist or is not a file.")
    output_dir = os.path.dirname(output_wav_path)
    if output_dir and not os.path.isdir(output_dir):
        raise FileNotFoundError(f"The directory '{output_dir}' for output WAV does not exist.")


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
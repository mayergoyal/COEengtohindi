import re
import chardet
import os
import subprocess
from model_TTS.model_run import generate_tts
import subprocess
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import subprocess
import json
from transformers import MBart50Tokenizer, MBartForConditionalGeneration,pipeline
from textwrap import wrap
from pydub import AudioSegment
model_path = "/Users/karansood/Desktop/internship/model_translation"
import urllib.request

tokenizer = MBart50Tokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)

# Step 1: Extract audio using ffmpeg
def video_se_audio(video_path, audio_path):
    cmd = f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_path}"'
    subprocess.run(cmd, shell=True, check=True)
    print("Audio ka path:", audio_path)
    return audio_path

def audio_se_text(audio_path, text_path):
    command = [
    "./build/bin/whisper-cli",
    "-f",
    "../audio_text_files/eng_output.wav"
]

    with open("audio_text_files/eng_output.txt", "w") as outfile:
        subprocess.run(command, cwd="whisper-cpp-new", stdout=outfile)
    print("text ka path", text_path)
    return text_path

def clean_pehle(text:str)->str:
    text=text.strip()
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", text)
    if(text and not text.endswith(('.','!','?'))):
        text+='.'
    print(" cleaned text is ",text)
    return text

def translate_to_hindi(text:str)-> str:
    translator = pipeline("translation", model=model, tokenizer=tokenizer,
                          src_lang="en_XX", tgt_lang="hi_IN", device=-1)

    hindi_text = translator(text, max_length=512)
    return hindi_text[0]['translation_text']

def save_output(text: str, out_path="audio_text_files/hindi_output.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Hindi text is here:", out_path)


def hindi_sentence_split(text, max_length=200, min_length=50):
    import re

    sentences = re.split(r'(?<=[।!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if adding this sentence keeps us within max_length
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            # If the current_chunk is too small, force add the sentence
            if len(current_chunk) < min_length:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def ttss(text, speaker_wav_path):
    if not text.strip():
        raise ValueError("Input text is empty. Cannot synthesize speech.")

    # Split text into chunks
    chunks = hindi_sentence_split(text)

    # Prepare output directory
    chunk_dir = "audio_text_files/hindi_chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    # Clear old chunk files to prevent reuse
    for old_file in os.listdir(chunk_dir):
        os.remove(os.path.join(chunk_dir, old_file))

    combined = AudioSegment.empty()

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue  # Skip empty chunks

        print(f"[Chunk {i}]: {chunk}")
        chunk_wav_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        
        # Generate audio only for valid chunk
        generate_tts(chunk, speaker_wav_path, chunk_wav_path)

        # Append generated audio
        combined += AudioSegment.from_wav(chunk_wav_path)

    # Export final combined output
    combined.export("audio_text_files/hindi_output.wav", format="wav")

def get_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]

    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout
        
    data = json.loads(output)
    
    duration = float(data['format']['duration'])
    return duration

def build_atempo_filter(speed: float) -> str:
    #Build an 'atempo' filter chain to achieve the desired speed for audio video sync.
    if speed <= 0:
        raise ValueError("Speed must be positive")
    
    filters = []
    
    # Break down the speed into multiple factors within [0.5, 2.0]
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    
    # Append the remainder if not 1.0
    if abs(speed - 1.0) > 1e-3:
        filters.append(f"atempo={speed:.3f}")
    
    return ",".join(filters) if filters else "atempo=1.0"


def run_pipeline(video_path):
    audio_path = "audio_text_files/eng_output.wav"
    text_path = "audio_text_files/eng_output.txt"
    hindi_out_path = "audio_text_files/hindi_output.txt"

    video_se_audio(video_path, audio_path)
    audio_se_text(audio_path, text_path)

    # Read and clean transcript
    with open(text_path, "rb") as f:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        text = raw.decode(encoding)
    print("Reading Whisper transcript...")
    print("Input text is:\n", text)

    cleaned_text = clean_pehle(text)

    # Translate
    print("\nTranslating to Hindi...")
    hindi_text = translate_to_hindi(cleaned_text)
    
    # Save output
    hindi_text = hindi_text.replace('.', '।')

    save_output(hindi_text, hindi_out_path)
    print(hindi_text)

    #text to audio 
    speaker_wav_path = "HIN_M_AvdheshT.wav"
    ttss(hindi_text, speaker_wav_path)

    #sync to video 
    video_path = input_video
    audio_path = "audio_text_files/hindi_output.wav"

    
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    stretch_factor = audio_duration/video_duration
    atempo_filter = build_atempo_filter(stretch_factor)
    adjusted_audio_path = "audio_text_files/hindi_output_adjusted.wav"

    subprocess.run([
    "ffmpeg", "-y",
    "-i", audio_path,
    "-filter:a", atempo_filter,
    adjusted_audio_path
 ], check=True)

    final_output_path = "final_output.mp4"
    subprocess.run([
    "ffmpeg", "-y",
    "-i", video_path,
    "-i", adjusted_audio_path,
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "192k",
    "-ar", "44100",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",  
    final_output_path
    ], check=True)

    print("\nPipeline complete")


if __name__ == "__main__":
    input_video =  '/Users/karansood/Desktop/Movie on 12-06-25 at 9.46 PM.mov' 
    run_pipeline(input_video)

    
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
import librosa
model_path = "/Users/karansood/Desktop/internship/model_translation"
import urllib.request

tokenizer = MBart50Tokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)
translator = pipeline("translation", model=model, tokenizer=tokenizer,
                      src_lang="en_XX", tgt_lang="hi_IN", device=-1)

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

#getting hindi txt with timestamps
def translate_timestamped_file(eng_txt_path: str, hindi_txt_path: str):
    with open(eng_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    hindi_lines = []
    print("Total lines read:", len(lines))
    print("First 5 lines:", lines[:5])

    for line in lines:
        # Match timestamp block
        match = re.match(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)", line)
        if not match:
            continue  # skip invalid lines

        start_time = match.group(1)
        end_time = match.group(2)
        english_text = match.group(3)

       # Translate only the text part
        hindi_translation = translator(english_text, max_length=512)[0]['translation_text']
        # Format: timestamp_line \t translation
        hindi_lines.append(f"[{start_time} --> {end_time}] {hindi_translation.strip()}")


    # Save to hindi.txt with timestamps
    with open(hindi_txt_path, 'w', encoding='utf-8') as f:
        for line in hindi_lines:
            f.write(line + "\n")

    print(f"Translated with timestamps: {hindi_txt_path}")


def merge_timestamped(input_path, output_path, max_gap=1.0, min_chars=200):
    def timestamp_to_seconds(timestamp: str) -> float:
        h, m, s = timestamp.split(':')
        s, ms = s.split('.')
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


    # Parse input file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # ⛔ Skip blank lines
        match = re.match(r"^\[(.*?)\s*-->\s*(.*?)\]\s*(.+)", line)


        if not match:
            print(f"❌ Could not parse line: {repr(line.strip())}")
            continue

        start = timestamp_to_seconds(match.group(1))
        end = timestamp_to_seconds(match.group(2))

        text = match.group(3).strip()
        chunks.append((start, end, text))

    # Merging logic
    merged = []
    current_text = ""
    current_start = None
    current_end = None

    for i, (start, end, text) in enumerate(chunks):
        if current_text == "":
            current_text = text
            current_start = start
            current_end = end
            continue

        gap = start - current_end
        temp_text = current_text + " " + text

        # Only merge if:
        # 1. Gap is within allowed range
        # 2. Final char of current text is `।` (full sentence)
        if gap <= max_gap and (current_text.strip().endswith("।") or current_text.strip().endswith(".")):
            current_text = temp_text
            current_end = end

            # If enough chars, save
            if len(current_text.strip()) >= min_chars:
                merged.append((current_start, current_end, current_text.strip()))
                current_text = ""
                current_start = None
                current_end = None
        else:
            # Save current if not empty
            if current_text.strip():
                merged.append((current_start, current_end, current_text.strip()))
            current_text = text
            current_start = start
            current_end = end

    # Save last one if needed
    if current_text.strip():
        merged.append((current_start, current_end, current_text.strip()))

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for start, end, text in merged:
            f.write(f"[{start:.2f} - {end:.2f}] {text}\n")

    print(f"✅ Merged output saved to {output_path}")


#getting hindi.srt from hindi.txt
def convert_timestamped_txt_to_srt(txt_file, srt_file):
    def format_time(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(srt_file, 'w', encoding='utf-8') as f:
        index = 1
        for line in lines:
            match = re.match(r'\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s+(.*)', line)
            if match:
                start_sec = float(match.group(1))
                end_sec = float(match.group(2))
                text = match.group(3).strip()

                start = format_time(start_sec)
                end = format_time(end_sec)

                f.write(f"{index}\n{start} --> {end}\n{text}\n\n")
                index += 1
#removing timestamps for TTS
def extract_only_hindi_text(hindi_txt_path: str) -> str:
    with open(hindi_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    hindi_lines = []
    for line in lines:
        match = re.match(r'\[.*?\]\s+(.*)', line)
        if match:
            hindi_lines.append(match.group(1).strip())

    hindi_text = ' '.join(hindi_lines)
    return hindi_text

#breaking the timestamped hindi text into chunks 
def parse_srt_to_chunks(srt_path):
    def to_seconds(time_str):
        """Convert 'HH:MM:SS,mmm' to seconds (float)"""
        h, m, s_ms = time_str.strip().split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    chunks = []
    entries = re.split(r'\n\s*\n', content)

    for entry in entries:
        lines = entry.strip().splitlines()
        if len(lines) >= 2:
            timestamp_line = lines[1].strip()
            text = " ".join(lines[2:]).strip()

            match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
            if match:
                start = to_seconds(match.group(1))
                end = to_seconds(match.group(2))
                chunks.append((start, end, text))

    return chunks


#generating audio with the broken timestamped chunks 
def generate_aligned_audio(srtPath, sample_rate=16000):
    chunks = parse_srt_to_chunks(srtPath)
    temp_dir = "audio_text_files/hindi_chunks"
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(temp_dir)

    final_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)

    for i, (start, end, text) in enumerate(chunks):
        print(f"🎙️ [Chunk {i+1}] Text: {text}")
        chunk_path = os.path.join(temp_dir, f"chunk{i+1}.wav")
        generate_tts(text, "HIN_M_AvdheshT.wav", chunk_path)

        target_duration = (end - start) * 1000  # in milliseconds
        original_audio = AudioSegment.from_file(chunk_path)
        original_duration = len(original_audio)

        if abs(original_duration - target_duration) > 50:
            # Stretch or shrink using ffmpeg atempo
            atempo = original_duration / target_duration
            atempo = max(0.5, min(2.0, atempo))  # FFmpeg atempo supports 0.5–2.0 per filter
            temp_path = os.path.join(temp_dir, "temp.wav")

            subprocess.run([
                "ffmpeg", "-y", "-i", chunk_path,
                "-filter:a", f"atempo={atempo}",
                temp_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            tts_audio = AudioSegment.from_file(temp_path)
        else:
            tts_audio = original_audio

        # Pad silence if needed
        current_time = len(final_audio)
        start_ms = int(start * 1000)
        if start_ms > current_time:
            silence = AudioSegment.silent(duration=start_ms - current_time, frame_rate=sample_rate)
            final_audio += silence

        final_audio += tts_audio

    final_audio.export("audio_text_files/hindi_output.wav", format="wav")
    print("✅ Hindi Final Audio saved.")



def build_ffmpeg_command(audio_choice, subtitle_choice,input_video):
    # Map user choices to file names
    audio_file = "audio_text_files/hindi_output.wav" if audio_choice == "hindi" else "audio_text_files/eng_output.wav"

    subtitle_map = {
        "none": None,
        "english": "audio_text_files/eng._output.srt",
        "hindi": "audio_text_files/hindi_output.srt"
    }
    subtitle_file = subtitle_map.get(subtitle_choice.lower())

    # Construct ffmpeg command
    cmd = f'ffmpeg -y -i "{input_video}" -i "{audio_file}" '

    if subtitle_file:
        cmd += f'-vf subtitles="{subtitle_file}" '

    cmd += '-map 0:v -map 1:a -c:v libx264 -c:a aac -shortest '
    cmd += "final_output.mp4"
    return cmd


def run_pipeline(video_path):
    eng_audio_path = "audio_text_files/eng_output.wav"
    eng_text_path = "audio_text_files/eng_output.txt"
    hindi_text_path = "audio_text_files/hindi_output.txt"

    #eng video------>eng audio
    video_se_audio(video_path, eng_audio_path)

    #eng audio----->timestamped eng text
    audio_se_text(eng_audio_path, eng_text_path)

    # Read and clean transcript
    with open(eng_text_path, "rb") as f:
        raw = f.read()
    encoding = chardet.detect(raw)['encoding'] or 'utf-8'
    text = raw.decode(encoding)
    print("Reading Whisper transcript...")
    print("Input text is:\n", text)

    cleaned_text = clean_pehle(text)

    # eng timestamped text-------->hindi timestamped text
    print("\nTranslating to Hindi...")
    translate_timestamped_file(eng_text_path,hindi_text_path)

    #merging timestamps 
    merge_timestamped(hindi_text_path, "audio_text_files/hindi_updated_timestamps.txt")
    merge_timestamped(eng_text_path, "audio_text_files/eng_updated_timestamps.txt")
    
    
    hindi_text=extract_only_hindi_text("audio_text_files/hindi_updated_timestamps.txt")
    convert_timestamped_txt_to_srt("audio_text_files/hindi_updated_timestamps.txt", "audio_text_files/hindi_output.srt")
    convert_timestamped_txt_to_srt("audio_text_files/eng_updated_timestamps.txt","audio_text_files/eng._output.srt")


    #text to audio 
    speaker_wav_path = "HIN_M_AvdheshT.wav"
    generate_aligned_audio("audio_text_files/hindi_output.srt")
    print(hindi_text)

    

    audio_choice = input("Choose audio language (english/hindi): ").strip().lower()
    subtitle_choice = input("Choose subtitles (none/english/hindi): ").strip().lower()

    if audio_choice not in ["english", "hindi"]:
        print("Invalid audio choice")
        return
    if subtitle_choice not in ["none", "english", "hindi"]:
        print("Invalid subtitle choice")
        return

    # Build and run command
    cmd = build_ffmpeg_command(audio_choice, subtitle_choice,video_path)
    print(f"\nRunning:\n{cmd}\n")
    
    os.system(cmd)
    print("Video generated")


if __name__ == "__main__":
    input_video =  '/Users/karansood/Desktop/Movie on 14-06-25 at 11.34 AM.mov' 
    run_pipeline(input_video)

    
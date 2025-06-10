from transformers import MarianTokenizer as token, MarianMTModel as mtmodel
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

def get_duration(path):
    # Properly quote the filename if it contains spaces
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]

    try:
        # Run ffprobe and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse JSON output
        data = json.loads(output)
        
        # Extract duration
        duration = float(data['format']['duration'])
        return duration

    except subprocess.CalledProcessError as e:
        print(f"ffprobe error: {e}")
    except KeyError:
        print(f"Could not find duration info in ffprobe output: {output}")
    except json.JSONDecodeError:
        print(f"Failed to parse ffprobe output as JSON: {output}")

    return None  # Return None if something went wrong

def build_atempo_filter(speed: float) -> str:
    """
    Build an 'atempo' filter chain to achieve the desired speed.
    atempo accepts values between 0.5 and 2.0, so chain filters if outside this range.
    """
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
    "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/eng_output.wav"
]

    with open("/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/eng_output.txt", "w") as outfile:
        subprocess.run(command, cwd="/Users/karansood/Desktop/internship/COEengtohindi/whisper-cpp-new", stdout=outfile)
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
    
    tokenizer = token.from_pretrained("/Users/karansood/Desktop/internship/model_engTohindiText")
    model = mtmodel.from_pretrained("/Users/karansood/Desktop/internship/model_engTohindiText")
    
    #time to tokenize the input text
    inputs=tokenizer(text,return_tensors='pt',padding=True)
    
    # now ab transltae karte hain
    translated=model.generate(**inputs)
    #decoding the translation
    hindi_text=tokenizer.decode(translated[0],skip_special_tokens=True)
    return hindi_text

def save_output(text: str, out_path="hindi_output.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Hindi text is here:", out_path)



def run_pipeline(video_path):
    audio_path = "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/eng_output.wav"
    text_path = "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/eng_output.txt"
    hindi_out_path = "hindi_output.txt"

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
    save_output(hindi_text, hindi_out_path)

    
    print(hindi_text)


    #sync to video back 
    speaker_wav_path = "/Users/karansood/Desktop/internship/HIN_M_AvdheshT.wav"
    tts_output_wav = "/Users/karansood/Desktop/internship/COEengtohindi/hindi_output.wav"
    generate_tts(hindi_text, speaker_wav_path, tts_output_wav)

    video_path = input_video
    audio_path = "/Users/karansood/Desktop/internship/COEengtohindi/hindi_output.wav"

    
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    stretch_factor = audio_duration/video_duration
    atempo_filter = build_atempo_filter(stretch_factor)
    adjusted_audio_path = "/Users/karansood/Desktop/internship/COEengtohindi/hindi_output_adjusted.wav"

    subprocess.run([
    "ffmpeg", "-y",
    "-i", audio_path,
    "-filter:a", atempo_filter,
    adjusted_audio_path
 ], check=True)

    final_output_path = "/Users/karansood/Desktop/internship/COEengtohindi/final_output.mp4"
    subprocess.run([
    "ffmpeg", "-y",
    "-i", video_path,
    "-i", adjusted_audio_path,
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "192k",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",  
    final_output_path
    ], check=True)

    print("\nPipeline complete")

if __name__ == "__main__":
    input_video = '/Users/karansood/Desktop/Movie on 03-06-25 at 8.43 PM.mov'

    if os.path.exists(input_video):
        print(f"Selected video: {input_video}")
        run_pipeline(input_video)
    else:
        print("Video not found at path. Exiting...")
    
    


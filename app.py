from transformers import MarianTokenizer as token, MarianMTModel as mtmodel
import re
import chardet
import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = warnings, 2 = errors, 3 = fatal

# Step 1: Extract audio using ffmpeg
def video_se_audio(video_path, audio_path):
    cmd = f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_path}"'
    subprocess.run(cmd, shell=True, check=True)
    print("Audio ka path:", audio_path)
    return audio_path

def audio_se_text(audio_path, text_path):
    #cmb=f"cd whisper-cpp-new"
    #whisper_cmd = f"./build/bin/whisper-cli -f {audio_path}  >{text_path}"
    #subprocess.run(cmb, shell=True, check=True)
    #subprocess.run(whisper_cmd, shell=True, check=True)
    command = [
    "./build/bin/whisper-cli",
    "-f",
    "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/output.wav"
]

    with open("/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/output.txt", "w") as outfile:
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
    audio_path = "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/output.wav"
    text_path = "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/output.txt"
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

    print("\nPipeline complete. Hindi output:\n")
    print(hindi_text)

if __name__ == "__main__":
    input_video = "/Users/karansood/Desktop/internship/COEengtohindi/whisper.cpp/Movie on 04-06-25 at 12.17 PM.mov"  # Update your video path here
    run_pipeline(input_video)

    
import os
from pydub import AudioSegment
imputdir="C:\\Users\\Mayer\\OneDrive\\Desktop\\coe\\hinditts\\cv-corpus-7.0-2021-07-21\\hi\\clips"
outputdir="C:\\Users\\Mayer\\OneDrive\\Desktop\\coe\\hinditts\\cv-corpus-7.0-2021-07-21\\hi\\clipswav"
os.makedirs(outputdir, exist_ok=True)
for filename in os.listdir(imputdir):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(imputdir, filename)
        wav_path = os.path.join(outputdir, filename.replace(".mp3", ".wav"))
        
        # Load the MP3 file
        audio = AudioSegment.from_mp3(mp3_path)
        audio=audio.set_channels(1).set_frame_rate(16000)
        # Export as WAV
        audio.export(wav_path, format="wav")
        print(f"Converted {filename} to WAV format.")
    else:
        print(f"Skipping {filename}, not an MP3 file.")
print("All MP3 files have been converted to WAV format and saved in the output directory.")
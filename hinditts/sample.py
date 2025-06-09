from TTS.api import TTS
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")



# ✅ Check if the model is multilingual
print("Model languages:", tts.languages)
print("Is multilingual?", tts.is_multi_lingual)

# ✅ Read the input Hindi text
with open("C:\\Users\\Mayer\\OneDrive\\Desktop\\coe\\hinditext.txt", "r", encoding="utf-8") as file:
    text = file.read()
print("Input text:\n", text)

# ✅ Run TTS with Hindi language and reference voice
try:
    tts.tts_to_file(
        text=text,
        speaker_wav="cv-corpus-7.0-2021-07-21/hi/clipswav/common_voice_hi_23795360.wav",
        language="hi",
        file_path="output.wav"
    )
    print("✅ Audio generated successfully: output.wav")

except Exception as e:
    print("❌ Error during TTS generation:", e)
    print("⚠️ Make sure that 'hi' is supported and speaker_wav is a clean Hindi clip.")

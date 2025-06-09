from transformers import MarianTokenizer as token, MarianMTModel as mtmodel
import re
import chardet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = warnings, 2 = errors, 3 = fatal

def clean_pehle(text:str)->str:
    text=text.strip()
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", text)
    if(text and not text.endswith(('.','!','?'))):
        text+='.'
    print(" cleaned text is ",text)
    return text
def translate_to_hindi(text:str)-> str:
    
    tokenizer = token.from_pretrained("C:\\Users\\Mayer\\OneDrive\\Documents\\model")
    model = mtmodel.from_pretrained("C:\\Users\\Mayer\\OneDrive\\Documents\\model")
    
    #time to tokenize the input text
    inputs=tokenizer(text,return_tensors='pt',padding=True)
    
    # now ab transltae karte hain
    translated=model.generate(**inputs)
    #decoding the translation
    hindi_text=tokenizer.decode(translated[0],skip_special_tokens=True)
    return hindi_text

if __name__=="__main__":
    with open(".\\whisper.cpp\\output.txt", "rb") as file:
        inp = file.read()
        encoding=chardet.detect(inp)['encoding']
    inp=inp.decode(encoding)
    print("Translating to Hindi ...")
    print("Input text is ", inp)
    inp = clean_pehle(inp)
    print(" cleaned Input text is ", inp)
    output=translate_to_hindi(inp)
    print("here u go ")
    print(output)
    with open("hinditext.txt","w",encoding='utf-8') as file:
        file.write(output)
        print("output successfully written")
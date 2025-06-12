
from huggingface_hub import snapshot_download
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

def download_model():
    snapshot_download(
        repo_id="ai4bharat/indictrans2-en-indic-dist-200M",
        local_dir="/Users/karansood/Desktop/internship/model_translation",
        local_dir_use_symlinks=False
    )

def download_tokenizer():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    tokenizer.save_pretrained("/Users/karansood/Desktop/internship/model_translation")
    model.save_pretrained("/Users/karansood/Desktop/internship/model_translation")

download_model()
download_tokenizer()


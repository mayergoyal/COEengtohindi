from transformers import MarianTokenizer, MarianMTModel

model_name = "Helsinki-NLP/opus-mt-en-hi"  # English → Hindi model

# Download and save both tokenizer + model to ./model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

tokenizer.save_pretrained("C:\\Users\\Mayer\\OneDrive\\Documents\\model")
model.save_pretrained("C:\\Users\\Mayer\\OneDrive\\Documents\\model")

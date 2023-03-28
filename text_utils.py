from transformers import AutoTokenizer
#class GetTextFromAudio: 
#@TODO #@Joon define your class that gets speech transcription from audio

class TokenizeText:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokeizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(self, text):
        return tokenizer(text, return_tensors="pt")

#@TODO #@Shaunak tokenize your text obtained from @Joon's GetTextFromAudio
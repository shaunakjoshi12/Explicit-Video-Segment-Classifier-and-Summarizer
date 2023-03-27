import torch, pdb
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

class LanguageModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", output_attentions=True):
        """
            Description: Language model which takes input as text proecessed from whisper and gives the final attention layers as output
            @param model_name: Pretrained model name
            @param output_attentions: Boolean specifies whether to give attention layer output or not
        """
        super(LanguageModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, output_attentions=output_attentions
        )
        self.att_layer_dims = 1024
        self.linear_out = 50
        self.linear = nn.Linear(self.att_layer_dims, self.linear_out)
        self.flatten = nn.Flatten()

    def forward(self, tokenized_text):
        """
            Description: Forward function takes the text tokenized by the bert encoder and passes through the model

            @param tokenized_text: Text tokenized using BERT
        """
        x = self.model(**tokenized_text).attentions[-1]
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.flatten(self.linear(x))
        return x

class UnifiedModel(nn.Module):
    def __init__(self, in_dims=None, intermediate_dim=None):
        """
            Description: A unified model that takes language model output , video_classifier output and audio_classifier output. Here audio_classifier output is spectrogram

            @param in_dims: The dimensions obtained from concatenating language model output , video_classifier output and audio_classifier output
            @param intermediate_dim: The dimension obtained by using an intermediate linear layer over the input obtained from the 'in_dims' layer
        """
        super(UnifiedModel, self).__init__()
        self.in_dims = None #dim_lang_model + dim_video_classifier + dim_audio_classifier
        self.intermediate_dim = None #obtained after linear layer on in_dims
        self.num_classes = 1
        self.linear1 = nn.Linear(self.in_dims, self.intermediate_dim)
        self.linear2 = nn.Linear(self.intermediate_dim, self.num_classes)

    def forward(self, language_model_out, video_classifier_out, audio_classifier_out):
        """
            Description: Forward function takes language model output , video_classifier output and audio_classifier output

            @param language_model_out: the final layer flattened output of language_model
            @param video_classifier_out: the final layer flattened output of video classifier model
            @param audio_classifier_out: the final layer flattened output of audio classifier model (spectrogram classifier)
        """
        x = torch.cat((language_model_out, video_classifier_out, audio_classifier_out), axis=-1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    llm = LanguageModel()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    inputs = tokenizer(text, return_tensors="pt")    

    with torch.no_grad():
        out_ = llm(inputs)



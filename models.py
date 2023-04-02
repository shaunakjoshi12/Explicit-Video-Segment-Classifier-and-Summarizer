import torch, pdb
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

#class VideoModel(nn.Module):
#@TODO #@Raghav define your class which does 1. Takes input as transformed_video tensor from dataset I have defined and returns the pre-final layer from videoclassifier
class VideoModel(nn.Module):
    def __init__(self, model_name='slowfast_r50', pretrained=True) -> None:
        super().__init__()

        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

    def forward(self, x):
        x = x.squeeze(0)
        pred = self.model(x)
        return pred

#class AudioModel(nn.Module):
#@TODO #@Arpita define your class which does 1. Takes input as processed_spectrogram tensor from dataset I have defined and returns the pre-final layer from audioclassifier

class LanguageModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", output_attentions=True):
        """
            Description: Language model which takes input as processed_speech from dataset I have defined and gives the final attention layers as output
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
        ## WORKAROUND! NOT AT ALL RECOMMENDED, STILL TRYIN TO FIGURE OUT WHY THE BELOW TWO LINES ARE NEEDED FOR LANGUAGE MODEL
        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
        x = self.model(**tokenized_text).attentions[-1]
        x = x.reshape(x.size(0), -1)
        x = self.flatten(self.linear(x))
        return x

class UnifiedModel(nn.Module):
    def __init__(self, in_dims=None, intermediate_dims=None, LanguageModel_obj=None, VideModel_obj=None, AudioModel_obj=None):
        """
            Description: A unified model that takes language model output , video_classifier output and audio_classifier output. Here audio_classifier output is spectrogram

            @param in_dims: The dimensions obtained from concatenating language model output , video_classifier output and audio_classifier output
            @param intermediate_dim: The dimension obtained by using an intermediate linear layer over the input obtained from the 'in_dims' layer
            @param LanguageModel_obj: The pytorch model of LanguageModel defined above
            @param VideModel_obj: The pytorch model of VideoModel defined above
            @param AudioModel_obj: The pytorch model of AudioModel defined above
            
        """
        super(UnifiedModel, self).__init__()
        self.in_dims = in_dims #dim_lang_model + dim_video_classifier + dim_audio_classifier
        self.intermediate_dims = intermediate_dims #obtained after linear layer on in_dims
        self.num_classes = 2
        self.LanguageModel_obj = LanguageModel_obj
        self.VideModel_obj = VideModel_obj
        self.AudioModel_obj = AudioModel_obj
        self.linear1 = nn.Linear(self.in_dims, self.intermediate_dims)
        self.linear2 = nn.Linear(self.intermediate_dims, self.num_classes)
        #self. = nn.Sigmoid()

    def forward(self, language_model_in, video_classifier_in):#, audio_classifier_in):
        """
            Description: Forward function takes language model output , video_classifier output and audio_classifier output

            @param language_model_in: the processed tokenized input from dataset class
            @param video_classifier_in: the processed video input from dataset class
            @param audio_classifier_in: the processed audio input from dataset class
        """
        language_model_out = self.LanguageModel_obj(language_model_in)
        video_classifier_out = self.VideModel_obj(video_classifier_in)
        #audio_classifier_out = self.AudioModel_obj(audio_classifier_in)
        
        x = torch.cat((language_model_out, video_classifier_out), axis=-1)
        x = self.linear1(x)
        x = self.linear2(x)
        #x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    llm = LanguageModel()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    inputs = tokenizer(text, return_tensors="pt")    

    with torch.no_grad():
        out_ = llm(inputs)
        print(out_.size())



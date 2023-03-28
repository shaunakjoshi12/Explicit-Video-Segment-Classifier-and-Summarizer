import os
import glob
from torch.utils.data import Dataset

#@Raghav refer to torchhub notebook

class VideoClipDataset(Dataset):
    def __init__(self, **dataset_dict):#):video_paths=None, root_dir_path=None, EncodeAndTransformedVideo_obj=None, GetSpectrogramFromAudio_obj=None, GetTextFromAudio_obj=None, TokenizeText_obj=None): 
        """
            Description: A unified dataset for all the modalities i.e. video, text and audio

            @param video_paths: A list containing paths to all those videos corresponding to train or val based upon the task being performed
            @param root_dir_path: Path where videos are stored in form of root_dir/explicit/* AND root_dir/non_explicit/* 
            @param EncodeAndTransformedVideo_obj:  Object of Raghav's class that does 1. EncodedVideo(video) and 2. transform using ApplyTransformToKey 
            @param GetSpectrogramFromAudio_obj: Object of Arpita's class that gets a spectrogram image from audio
            @param GetTextFromAudio_obj: Object of Joon's class that gets speech transcription from audio
            @param TokenizeText_obj: Object of Shaunak's class that gets tokens from speech transcription

        """
        self.videos, self.root_dir_path, self.EncodeAndTransformedVideo_obj, self.GetSpectrogramFromAudio_obj, self.GetTextFromAudio_obj, self.TokenizeText_obj = dataset_dict.values()
        self.root_dir_path = root_dir_path
        self.classes = {elem:i for i, elem.split('/')[-1] in enumerate(glob.glob(os.path.join(self.root_dir_path,'*')))} #Map class name to id
        

    def __getitem__(self, index):
        video = self.videos[i]
        transformed_video = self.EncodeAndTransformedVideo_obj.get_video(video) #get_video (or any func name): function defined by Raghav in his class to get the encoded and transformed video
        processed_spectrogram = self.GetSpectrogramFromAudio_obj.get_spectrogram(video) #get_spectrogram (or any func name): function defined by Arpita in her class to get the spectrogram
        processed_speech = self.TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(video)) #get_speech (or any func name): function defined by Joon in his class to get the transcribed speech
        #get_token_tensor: function defined by Shaunak in his class to get the token tensor from the transcribed speech

        return transformed_video, processed_spectrogram, processed_speech, self.classes[video.split('/')[-2]]

    def __len__(self):
        return len(self.videos)
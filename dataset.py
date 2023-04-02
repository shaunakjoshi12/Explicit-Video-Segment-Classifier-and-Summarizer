import os
import pdb
import glob
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class VideoClipDataset(Dataset):
    def __init__(self, **dataset_dict):
        """
            Description: A unified dataset for all the modalities i.e. video, text and audio

            Params extracted from unpacking dataset_dict
            @param root_dir_path: The directory one level above processed_data which contains raw data as well which hasn't been split
            @param encoded_videos: The folders which have audio and video encodings
                                    Structure: processed_data/encoded_videos
                                                - explicit
                                                    - God Bless America (Video name)
                                                        - video_encs (I have made 1 minute chunks of any that exceeds 95 secs because the RAM gets killed/encodings don't get saved)
                                                            god_bless_america_video_enc_0
                                                            god_bless_america_video_enc_1
                                                                        .
                                                                        .
                                                                        .
                                                            god_bless_america_video_enc_n

                                                        - audio_encs
                                                            god_bless_america_audio_enc_0
                                                            god_bless_america_audio_enc_1
                                                                        .
                                                                        .
                                                                        .
                                                            god_bless_america_audio_enc_n
            @param device: "cuda" or "cpu"
        """
    
        self.root_dir_path, self.encoded_videos, self.device = dataset_dict.values()
        video_encodings = list()
        audio_encodings = list()
        for video in self.encoded_videos:
            video_encodings+=glob.glob(os.path.join(video, 'video_encs/*'))
            audio_encodings+=glob.glob(os.path.join(video, 'audio_encs/*'))

        assert len(video_encodings)==len(audio_encodings), 'check your data'
        del video_encodings, audio_encodings
        #print(self.encoded_videos)
            
        self.classes = {elem.split('/')[-1]:i for i, elem in enumerate(glob.glob(os.path.join(self.root_dir_path,'processed_data/encoded_videos/*')))} #Map class name to id

    def __getitem__(self, index):
        video_enc_path, audio_enc_path = glob.glob(os.path.join(self.encoded_videos[index],'video_encs/*')), glob.glob(os.path.join(self.encoded_videos[index], 'audio_encs/*'))
        assert len(video_enc_path)==len(audio_enc_path)
        assert len(video_enc_path)==1
        video_enc_path = video_enc_path[0]
        audio_enc_path = audio_enc_path[0]
        class_video_enc = video_enc_path.split('/')[-4]
        class_audio_enc = audio_enc_path.split('/')[-4]

        assert class_video_enc == class_audio_enc,'class mismatch! check your data'
        video_enc = pickle.load(open(video_enc_path,'rb'))['transformed_video']
        audio_enc = pickle.load(open(audio_enc_path,'rb'))['processed_speech']
        video_enc = [elem.to(self.device) for elem in video_enc]
        audio_enc = {key:audio_enc[key].to(self.device) for key in audio_enc.keys()}
        class_video_enc = self.classes[class_video_enc]
        class_video_enc = torch.LongTensor(class_video_enc).to(self.device)

        return video_enc, audio_enc, class_video_enc

    def __len__(self):
        return len(self.encoded_videos)


if __name__=='__main__':
    root_dir_path = os.path.join(os.path.expanduser('~'), 'cls_data')
    dataset_dict = {
        'root_dir_path':root_dir_path,
        'encoded_videos':os.path.join(os.path.join(root_dir_path, 'processed_data/encoded_videos'))
    }
    videoclipdataset = VideoClipDataset(**dataset_dict)
    videoclipdataloader = DataLoader(videoclipdataset, batch_size=2)
    for data in videoclipdataloader:
        video_enc, audio_enc, class_ = data
        


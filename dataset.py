import os
import glob
from torch.utils.data import Dataset


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
        """
        
        self.root_dir_path, self.encoded_videos = dataset_dict.values()
        self.video_encodings = list()
        self.audio_encodings = list()
        for encoded_video_dir in self.encoded_videos:
            enc_videos = glob.glob(os.path.join(encoded_video_dir,'video_encs/*'))
            enc_audios = glob.glob(os.path.join(encoded_video_dir,'audio_encs/*'))
            self.video_encodings += enc_videos
            self.audio_encodings += enc_audios

        self.classes = {elem:i for i, elem.split('/')[-1] in enumerate(glob.glob(os.path.join(self.root_dir_path,'processed_data/non_encoded_videos/*')))} #Map class name to id
        

    def __getitem__(self, index):
        video_enc, audio_enc = self.video_encodings[i], self.audio_encodings[i]
        return video_enc, audio_enc, self.classes[video.split('/')[-2]]

    def __len__(self):
        return len(self.videos)

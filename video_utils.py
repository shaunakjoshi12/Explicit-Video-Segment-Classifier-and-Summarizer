#class EncodeAndTransformedVideo:

import os
import glob
import torch
# Choose the `slowfast_r50` model 

from tqdm import tqdm
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    RandomShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 

class PackPathway(torch.nn.Module):
        """
        Transform for converting video frames as a list of tensors. 
        """
        def __init__(self, slowfast_alpha):
            super().__init__()
            self.slowfast_alpha = slowfast_alpha

        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
            return frame_list
        
class EncodeVideo:
    def __init__(self) -> None:
        
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 32
        self.sampling_rate = 2
        self.frames_per_second = 30
        self.slowfast_alpha = 4
        self.num_clips = 10
        self.num_crops = 3

        self.transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                Lambda(lambda x: x/255.0),
                NormalizeVideo(self.mean, self.std),
                ShortSideScale(
                    size=self.side_size
                ),
                PackPathway(self.slowfast_alpha)
            ]),)
        
        self.clip_duration = (self.num_frames * self.sampling_rate)/self.frames_per_second
    
        # Select the duration of the clip to load by specifying the start and end duration
        # The start_sec should correspond to where the action occurs in the video
        self.start_sec = 0
        self.end_sec = self.start_sec + self.clip_duration

    def get_video(self, video_path, device=torch.device('cpu')):
            video = EncodedVideo.from_path(video_path)
            end_sec = video.duration            
            video_data = video.get_clip(start_sec=self.start_sec, end_sec=end_sec)
            video_data = self.transform(video_data)

            inputs = video_data["video"]
            with torch.no_grad():
                inputs = [i.to(device)[None, ...] for i in inputs]
                torch.cuda.empty_cache()
            return inputs

if __name__=='__main__':
    videos = glob.glob('/home/shaunaks/cls_data/processed_data/*/*')
    enc_vid = EncodeVideo()
    for vid in tqdm(videos):
        enc_v  = enc_vid.get_output(vid)
        #del enc_v
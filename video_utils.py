#class EncodeAndTransformedVideo:

import torch
# Choose the `slowfast_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
# Set to GPU or CPU
device = "cpu"
model = model.eval()
model = model.to(device)

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
    UniformTemporalSubsample,
    UniformCropVideo
) 

class PackPathway(torch.nn.Module):
        """
        Transform for converting video frames as a list of tensors. 
        """
        def __init__(self):
            super().__init__()

        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
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
                UniformTemporalSubsample(self.num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(self.mean, self.std),
                ShortSideScale(
                    size=self.side_size
                ),
                CenterCropVideo(self.crop_size),
                PackPathway()
            ]),)
        
        self.clip_duration = (self.num_frames * self.sampling_rate)/self.frames_per_second
    
        # Select the duration of the clip to load by specifying the start and end duration
        # The start_sec should correspond to where the action occurs in the video
        self.start_sec = 0
        self.end_sec = self.start_sec + self.clip_duration


        def get_output(self, video_path):

            video = EncodedVideo.from_path(video_path)

            # Load the desired clip
            video_data = video.get_clip(start_sec=self.start_sec, end_sec=self.end_sec)

            # Apply a transform to normalize the video input
            video_data = self.transform(video_data)

            # Move the inputs to the desired device
            inputs = video_data["video"]
            inputs = [i.to(device)[None, ...] for i in inputs]

            return inputs
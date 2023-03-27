import os
import torch
import glob
import argparse
import numpy as np
import torch.nn as nn
from models import *
from torch.optim import SGD, Adam
from dataset import VideoClipDataset
from torch.utils.data import DataLoader
from text_utils import GetTextFromAudio
from video_utils import EncodeAndTransformedVideo
from audio_utils import GetSpectrogramFromAudio
from torch.utils.data import SubsetRandomSampler

def get_train_val_split_samplers(root_dir, split_pct=0.2):
    all_videos = glob.glob(os.path.join(root_dir,'*/*'))
    indices = range(len(all_videos))
    np.random.shuffle(indices)
    val_split_index = int(len(all_videos)*split_pct)
    val_indices, train_indices = indices[:val_split_index], indices[val_split_index:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, val_sampler

#def train_val():

#def val():
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--optimizer_name',type=str)
    parser.add_argument('--root_dir', type=str,description='path where videos will be stored in the form of root_folder/<class>/video_file')
    parser.add_argument('--language_model_name', type=str,description='path to the fine-tuned model OR huggingface pretrained model name')
    parser.add_argument('--video_model_name', type=str,description='path to the fine-tuned model OR pretrained model name') #Optional
    parser.add_argument('--audio_model_name', type=str,description='path to the fine-tuned model OR pretrained model name') #Optional
    args = parser.parse_args()

    n_epochs = args.epochs
    learning_rate = args.learning_rate
    root_dir = args.root_dir
    language_model_name = args.language_model_name
    video_model_name = audio_model_name = language_model_name = None
    optimizer_name = args.optimizer_name

    if args.video_model_name:
        video_model_name = args.video_model_name

    if args.language_model_name:
        language_model_name = args.language_model_name

    if args.audio_model_name:
        audio_model_name = args.audio_model_name

    train_sampler, val_sampler = get_train_val_split_samplers(root_dir)

    ##Functions to transform modalities
    #EncodeAndTransformedVideo_obj = EncodeAndTransformedVideo() @Raghav
    #GetSpectrogramFromAudio_obj = GetSpectrogramFromAudio() @Arpita
    #GetTextFromAudio_obj = GetTextFromAudio() @Joon
    #TokenizeText_obj = TokenizeText() @Shaunak

    ##Model init
    #LanguageModel_obj = LanguageModel() @Shaunak
    #VideModel_obj = VideModel() @Raghav
    #AudioModel_obj = AudioModel() @Joon
    #in_dims = TBD
    #intermediate_dims = TBD
    #UnifiedModel_obj = UnifiedModel(in_dims, intermediate_dim, LanguageModel_obj, VideModel_obj, AudioModel_obj)

    if optimizer_name in ['SGD','sgd']:
        optimizer = SGD(UnifiedModel_obj.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name in ['Adam','adam']:
        optimizer = Adam(UnifiedModel_obj.parameters(), lr=learning_rate)

    all_videos = glob.glob(os.path.join(root_dir,'*/*'))
    dataset_dict = {
        'all_videos':all_videos,
        'root_dir_path':root_dir_path,
        'EncodeAndTransformedVideo_obj':EncodeAndTransformedVideo_obj,
        'GetSpectrogramFromAudio_obj':GetSpectrogramFromAudio_obj,
        'GetTextFromAudio_obj':GetTextFromAudio_obj,
        'TokenizeText_obj':TokenizeText_obj
    }


    train_dataloader, val_dataloader = DataLoader(VideoClipDataset(train_dataset_dict), shuffle=False, batch_size=batch_size, sampler=train_sampler),\
    DataLoader(VideoClipDataset(dataset_dict), shuffle=False, batch_size=batch_size, sampler=val_sampler)

    print('Training on \n train:{} batches \n val:{} batches'.format(len(train_dataloader), len(val_dataloader)))

    train_val()





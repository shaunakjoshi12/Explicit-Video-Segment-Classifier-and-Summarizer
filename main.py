import os
import pdb
import torch
import pickle
import glob
import argparse
import numpy as np
from data_utils import *
import torch.nn as nn
from models import *
from torch.optim import SGD, Adam
from dataset import VideoClipDataset
from models import LanguageModel, UnifiedModel
from torch.utils.data import DataLoader
from text_utils import GetTextFromAudio, TokenizeText
from video_utils import EncodeVideo
from models import VideoModel
from audio_utils import GetSpectrogramFromAudio
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


def get_train_val_split_videos(root_dir, split_pct=0.2):
    #Split explicit_train_val videos
    explicit_videos = glob.glob(os.path.join(root_dir,'explicit/*'))
    explicit_indices = range(len(explicit_videos))
    np.random.shuffle(explicit_indices)
    explicit_val_split_index = int(len(explicit_videos)*split_pct)
    explicit_videos_val,  explicit_videos_train = explicit_indices[:explicit_val_split_index], explicit_indices[explicit_val_split_index:]

    #Split non_explicit_train_val videos
    non_explicit_videos = glob.glob(os.path.join(root_dir,'non_explicit/*'))    
    non_explicit_indices = range(len(non_explicit_videos))
    np.random.shuffle(non_explicit_indices)
    non_explicit_val_split_index = int(len(non_explicit_videos)*split_pct)
    non_explicit_videos_val,  non_explicit_videos_train = non_explicit_indices[:non_explicit_val_split_index], non_explicit_indices[non_explicit_val_split_index:]

    #Get the total train_val videos
    train_videos, val_videos = explicit_videos_train+non_explicit_videos_train, explicit_videos_val+non_explicit_videos_val
    return train_videos, val_videos


def train_val(**train_val_arg_dict):
    unifiedmodel_obj, optimizer, train_dataloader, val_dataloader, n_epochs, batch_size, print_every, experiment_path = train_val_arg_dict.values()
    runs_dir = os.path.join(experiment_path,'runs')
    writer = SummaryWriter(runs_dir)
    train_losses = list()
    val_losses = list()
    loss_ = nn.BCELoss()
    best_loss = float('inf')
    sigmoid_threshold = 0.5

    for epoch in range(n_epochs):
        #train
        print('\n\n Epoch: {}'.format(epoch+1))
        print('\n Train')
        epoch_loss_train=0
        correct_train_preds = 0
        unifiedmodel_obj.train()
        for i, modality_inputs in enumerate(train_dataloader):
            transformed_video, processed_spectrogram, processed_speech, target = modality_inputs
            optimizer.zero_grad()
            predictions = unifiedmodel_obj(modality_inputs)
            batch_loss = loss_(predictions, target)
            batch_loss.backward()
            optimizer.step()
            predictions = predictions.cpu().detach()
            target = target.cpu().detach()
            predictions[predictions > 0.5] = 1
            predictions[predictions <= 0.5] = 0
            num_correct_preds = (predictions==target).sum()
            correct_train_preds+=num_correct_preds
            epoch_loss_train+=batch_loss.cpu().detach().item()
            writer.add_scalar("Loss/train", epoch_loss_train/(i+1), epoch)

            if i % print_every == 0:
                print('Batch:{}, Train epoch loss average:{} and accuracy till now:{}'.format(i+1, epoch_loss_train/(i+1), correct_train_preds/((i+1)*batch_size)))

        average_train_loss_per_epoch = epoch_loss_train/len(train_dataloader)
        print('For epoch:{} the average train loss: {} and the accuracy: {}'.format(epoch+1, average_train_loss_per_epoch, correct_train_preds/(len(train_dataloader)*batch_size)))
        train_losses.append(average_train_loss_per_epoch)

        #Val
        print('\n Val')
        unifiedmodel_obj.eval()
        epoch_loss_val=0
        correct_val_preds = 0
        for i, modality_inputs in enumerate(val_dataloader):
            with torch.no_grad():
                transformed_video, processed_spectrogram, processed_speech, target = modality_inputs
                predictions = unifiedmodel_obj(modality_inputs)
                batch_loss = loss_(predictions, target)
                predictions[predictions > 0.5] = 1
                predictions[predictions <= 0.5] = 0
                num_correct_preds = (predictions==target).sum()
                correct_val_preds+=num_correct_preds
                epoch_loss_val+=batch_loss
                writer.add_scalar("Loss/val", epoch_loss_val/(i+1), epoch)

            if i % print_every == 0:
                print('Batch:{}, Val epoch loss average:{}'.format(i+1, epoch_loss_val/(i+1)))

        average_val_loss_per_epoch = epoch_loss_val/len(val_dataloader)
        print('For epoch:{} the average val loss: {} and the accuracy:{}'.format(epoch+1, average_val_loss_per_epoch, correct_preds/(len(val_dataloader)*batch_size)))
        val_losses.append(average_val_loss_per_epoch)

        if average_val_loss_per_epoch < best_loss:
            best_loss = average_val_loss_per_epoch
            torch.save(unifiedmodel_obj.state_dict(), os.path.join(experiment_path, 'best_checkpoint.pth'))
    writer.flush()



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--optimizer_name',type=str)
    parser.add_argument('--root_dir', type=str,description='path where videos will be stored in the form of root_folder/<class>/video_file')
    parser.add_argument('--language_model_name', type=str,description='path to the fine-tuned model OR huggingface pretrained model name')
    parser.add_argument('--video_model_name', type=str,description='path to the fine-tuned model OR pretrained model name') #Optional
    parser.add_argument('--audio_model_name', type=str,description='path to the fine-tuned model OR pretrained model name') #Optional
    parser.add_argument('--experiment_path',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--print_every',type=int)
    args = parser.parse_args()

    n_epochs = args.epochs
    learning_rate = args.learning_rate
    root_dir = args.root_dir
    language_model_name = args.language_model_name
    video_model_name = audio_model_name = language_model_name = None
    optimizer_name = args.optimizer_name
    print_every = args.print_every
    experiment_path = args.experiment_path
    batch_size = args.batch_size

    makedir(experiment_path)

    if args.video_model_name:
        video_model_name = args.video_model_name

    if args.language_model_name:
        language_model_name = args.language_model_name

    if args.audio_model_name:
        audio_model_name = args.audio_model_name

    train_videos, val_videos = get_train_val_split_videos(root_dir)

    ##Functions to transform modalities
    EncodeVideo_obj = EncodeVideo() 
    #GetSpectrogramFromAudio_obj = GetSpectrogramFromAudio() @Arpita
    GetTextFromAudio_obj = GetTextFromAudio()
    TokenizeText_obj = TokenizeText()

    ##Model init
    LanguageModel_obj = LanguageModel(model_name=language_model_name)
    VideoModel_obj = VideoModel()
    #AudioModel_obj = AudioModel() @Joon
    in_dims = 2000
    intermediate_dims = 100
    UnifiedModel_obj = UnifiedModel(in_dims, intermediate_dim, LanguageModel_obj, VideModel_obj, AudioModel_obj)

    if optimizer_name in ['SGD','sgd']:
        optimizer = SGD(UnifiedModel_obj.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name in ['Adam','adam']:
        optimizer = Adam(UnifiedModel_obj.parameters(), lr=learning_rate)

    all_videos = glob.glob(os.path.join(root_dir,'processed_data/non_encoded_videos/*/*'))
    encoded_videos_path = os.path.join(root_dir,'processed_data/encoded_videos')
    if not os.path.exists(encoded_videos_path):
        encode_videos(all_videos, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj)    
    
    train_encoded_videos, val_encoded_videos = get_train_val_split_videos(encoded_videos_path)
    train_dataset_dict = {
        'root_dir':root_dir,
        'all_videos':train_encoded_videos,
    }

    val_dataset_dict = {
        'root_dir':root_dir,
        'all_videos':val_encoded_videos,
    }



    train_dataloader, val_dataloader = DataLoader(VideoClipDataset(train_dataset_dict), shuffle=True, batch_size=batch_size),\
    DataLoader(VideoClipDataset(val_dataset_dict), shuffle=True, batch_size=batch_size)

    print('Training on \n train:{} batches \n val:{} batches'.format(len(train_dataloader), len(val_dataloader)))

    train_val_arg_dict = {
        'unifiedmodel_obj':unifiedmodel_obj, 
        'optimizer':optimizer,
        'train_dataloader':train_dataloader,
        'val_dataloader':val_dataloader,
        'n_epochs':n_epochs,
        'batch_size':batch_size,
        'print_every':print_every,
        'experiment_path':experiment_path
    }
    train_val(train_val_arg_dict)





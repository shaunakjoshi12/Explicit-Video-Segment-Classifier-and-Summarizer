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
from torch.utils.tensorboard import SummaryWriter

def get_train_val_split_samplers(root_dir, split_pct=0.2):
    all_videos = glob.glob(os.path.join(root_dir,'*/*'))
    indices = range(len(all_videos))
    np.random.shuffle(indices)
    val_split_index = int(len(all_videos)*split_pct)
    val_indices, train_indices = indices[:val_split_index], indices[val_split_index:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, val_sampler

def train_val(unifiedmodel_obj, optimizer, train_dataloader, val_dataloader, n_epochs, batch_size, print_every, experiment_path):
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

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

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
    TokenizeText_obj = TokenizeText()

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





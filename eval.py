import os
import glob
import pdb
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
from models import VideoModel
from video_utils import EncodeVideo
from dataset import VideoClipDataset
from torch.utils.data import DataLoader
from models import LanguageModel, UnifiedModel
from torcheval.metrics.functional import multiclass_f1_score

def inference_on_val(non_encoded_videos_path, val_encoded_videos_pkl, classes, checkpoint_path, root_dir_path, EncodeVideo_obj, language_model_name='distilbert-base-uncased', video_model_name='slowfast_r50', device='cuda:0'):
    LanguageModel_obj = LanguageModel(model_name = language_model_name)
    VideoModel_obj = VideoModel(model_name = video_model_name)
    batch_size = 1
    softmax = nn.Softmax(dim=1)
    in_dims = 500
    intermediate_dims = 50
    UnifiedModel_obj = UnifiedModel(in_dims, intermediate_dims, LanguageModel_obj, VideoModel_obj).to(device)
    UnifiedModel_obj.load_state_dict(torch.load(checkpoint_path), strict=True)
    UnifiedModel_obj.eval()

    val_videos = pickle.load(open(val_encoded_videos_pkl,'rb'))

    
    val_dataset_dict = {
    'root_dir':root_dir_path,
    'all_encoded_videos':val_videos,
    'encoded_video_obj':EncodeVideo_obj,
    'device':device
    }

    val_dataloader = DataLoader(VideoClipDataset(**val_dataset_dict), shuffle=False, batch_size=batch_size)
        

    preds_val = list()
    targets_val = list()
    
    for i, modality_inputs in tqdm(enumerate(val_dataloader)):
        with torch.no_grad():
            transformed_video, processed_speech, target = modality_inputs
            target = target.to(device)
            predictions = UnifiedModel_obj(processed_speech, transformed_video)
            pred_softmax = softmax(predictions)
            pred_softmax = torch.argmax(pred_softmax, dim=1)
            preds_val.append(pred_softmax.cpu().item())
            targets_val.append(target.cpu().item())
    targets_val = torch.tensor(targets_val)
    preds_val = torch.tensor(preds_val)
    print('f1-score:{} accuracy:{}'.format(multiclass_f1_score(preds_val, targets_val, num_classes=2, average="macro").item(), (preds_val==targets_val).sum()/len(targets_val)))

    
    #pdb.set_trace()

if __name__=='__main__':
    root_dir_path = os.path.join(os.path.expanduser('~'), 'cls_data')
    experiment_name = 'third_run_sgd_lr_1e-3_macro_f1'
    non_encoded_videos_path = os.path.join(root_dir_path,'processed_data/non_encoded_videos')
    val_encoded_videos_pkl = 'runs/{}/val_encoded_video.pkl'.format(experiment_name)
    classes = {elem.split('/')[-1]:i for i, elem in enumerate(glob.glob(os.path.join(root_dir_path,'processed_data/encoded_videos/*')))}
    checkpoint_path = os.path.join(os.getcwd(),'runs',experiment_name, 'best_checkpoint.pth')
    EncodeVideo_obj = EncodeVideo()
    inference_on_val(non_encoded_videos_path, val_encoded_videos_pkl, classes, checkpoint_path, root_dir_path, EncodeVideo_obj)

import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import math
import glob
import pickle
from tqdm import tqdm
from video_utils import EncodeVideo
from text_utils import GetTextFromAudio, TokenizeText
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        return True
    return False

def break_larger_video_and_save(input_file, duration_sec,EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj, explicit_encoded, non_explicit_encoded):    
    ext_ = os.path.splitext(input_file)[1]
    video_chunks = math.ceil(duration_sec/60)
    class_ = input_file.split('/')[-2]

    
        
    for chunk_idx in range(video_chunks):
            output_file = os.path.join(os.path.expanduser('~'),'temp_{}'.format(chunk_idx)+ext_)
            if class_=='explicit':
                use_var = explicit_encoded
            else:
                use_var = non_explicit_encoded

            save_path_enc_dir = os.path.join(use_var, input_file.split('/')[-1])
            save_dir_video = os.path.join(save_path_enc_dir,'video_encs')
            save_dir_audio = os.path.join(save_path_enc_dir,'audio_encs')
            makedir(save_path_enc_dir)
            makedir(save_dir_video)
            makedir(save_dir_audio)
            save_path_video_enc = os.path.join(save_dir_video, input_file.split('/')[-1].replace(ext_,'_video_enc_{}'.format(chunk_idx)))
            save_path_audio_enc = os.path.join(save_dir_audio, input_file.split('/')[-1].replace(ext_,'_audio_enc_{}'.format(chunk_idx)))
            if os.path.exists(save_path_video_enc) and os.path.exists(save_path_audio_enc):
                continue

            
            if chunk_idx==video_chunks-1:
                ffmpeg_extract_subclip(input_file,chunk_idx*60, duration_sec, targetname=output_file)
            else:
                ffmpeg_extract_subclip(input_file,chunk_idx*60, chunk_idx*60 + 60, targetname=output_file)
            

            
            transformed_video = EncodeVideo_obj.get_video(output_file) #get_video (or any func name): function defined by Raghav in his class to get the encoded and transformed video
            #processed_spectrogram = GetSpectrogramFromAudio_obj.get_spectrogram(video) #get_spectrogram (or any func name): function defined by Arpita in her class to get the spectrogram
            processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(output_file)) #get_speech (or any func name): function defined by Joon in his class to get the transcribed speech
            
            
                
            vid_enc = True
            aud_enc = True
            
            try:
                pickle.dump({'transformed_video':transformed_video}, open(save_path_video_enc, 'wb'))
    
            except Exception as e:
                vid_enc = False
                raise ValueError('Couldnt encode video after splitting as well :( :( !')
            

            
            try:
                pickle.dump({'processed_speech':processed_speech}, open(save_path_audio_enc, 'wb'))
            except Exception as e:
                aud_enc = False
                raise ValueError('Couldnt encode audio after splitting as well :( :( !')
            

            if not vid_enc or not aud_enc:
                if os.path.exists(save_path_video_enc):
                    os.remove(save_path_video_enc)
                if os.path.exists(save_path_audio_enc):
                    os.remove(save_path_audio_enc)

            os.remove(output_file)
            del transformed_video, processed_speech

    



def encode_videos(videos_path, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj):
    explicit_encoded = os.path.join(encoded_videos_path,'explicit')
    non_explicit_encoded = os.path.join(encoded_videos_path,'non_explicit')
    makedir(explicit_encoded)
    makedir(non_explicit_encoded)
    print('Encoding all videos and text...')
    skipped_videos = 0
    for video_path in tqdm(videos_path):
        class_ = video_path.split('/')[-2]
        if class_=='explicit':
            use_var = explicit_encoded
        else:
            use_var = non_explicit_encoded
        save_path_enc_dir = os.path.join(use_var, video_path.split('/')[-1])
        save_dir_video = os.path.join(save_path_enc_dir,'video_encs')
        save_dir_audio = os.path.join(save_path_enc_dir,'audio_encs')
        makedir(save_path_enc_dir)
        makedir(save_dir_video)
        makedir(save_dir_audio)
        if '.mp4' in video_path.split('/')[-1]:
            save_path_video_enc = os.path.join(save_dir_video, video_path.split('/')[-1].replace('.mp4','_video_enc'))
            save_path_audio_enc = os.path.join(save_dir_audio, video_path.split('/')[-1].replace('.mp4','_audio_enc'))
        elif '.avi' in video_path.split('/')[-1]:
            save_path_video_enc = os.path.join(save_dir_video, video_path.split('/')[-1].replace('.avi','_video_enc'))
            save_path_audio_enc = os.path.join(save_dir_audio, video_path.split('/')[-1].replace('.avi','_audio_enc'))

        if os.path.exists(save_path_video_enc) and os.path.exists(save_path_audio_enc):
            continue

        
        try:
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps
            video.release()
        except:
            print('Corrupt video')
            skipped_videos+=1
            continue

        if duration_sec > 60:
            skipped_videos+=1
            continue
        
        try:
            transformed_video = EncodeVideo_obj.get_video(video_path) #get_video (or any func name): function defined by Raghav in his class to get the encoded and transformed video
        except:
            print('Problem in encoding video')
            skipped_videos+=1
            continue
        #processed_spectrogram = GetSpectrogramFromAudio_obj.get_spectrogram(video) #get_spectrogram (or any func name): function defined by Arpita in her class to get the spectrogram
        try:
            processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(video_path)) #get_speech (or any func name): function defined by Joon in his class to get the transcribed speech
        except:
            print('Problem in encoding audio')
            skipped_videos+=1
            continue

        vid_enc = True
        aud_enc = True
        try:
            pickle.dump({'transformed_video':transformed_video}, open(save_path_video_enc, 'wb'))
        except Exception as e:
            vid_enc = False
            raise ValueError('Couldnt encode video')
        
        try:
            pickle.dump({'processed_speech':processed_speech}, open(save_path_audio_enc, 'wb'))
        except Exception as e:
            aud_enc = False
            raise ValueError('Couldnt encode audio')
            
            
    print('Out of {} videos, skipped {} videos'.format(len(videos_path), skipped_videos))


if __name__=='__main__':
    root_dir = r"C:\Users\ragha\Documents\CSCI566_project"
    all_videos = glob.glob(os.path.join(root_dir,'processed_data/non_encoded_videos/*/*'))
    encoded_videos_path = os.path.join(root_dir,'processed_data/encoded_videos/')
    EncodeVideo_obj = EncodeVideo() 
    GetTextFromAudio_obj = GetTextFromAudio()
    #GetTextFromAudio_obj = None
    TokenizeText_obj = TokenizeText()    
    #if not os.path.exists(encoded_videos_path):
    encode_videos(all_videos, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj)    

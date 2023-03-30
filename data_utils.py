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

def break_larger_video_and_save(input_file, duration_sec,EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj, explicit_encoded, non_explicit_encoded):    
    ext_ = os.path.splitext(input_file)[1]
    output_files = list()
    video_chunks = math.ceil(duration_sec/60)
    class_ = input_file.split('/')[-2]

    print('Video chunks ',video_chunks)

    for i in range(video_chunks):
        output_files.append(os.path.join(os.path.expanduser('~'),'temp_{}'.format(i)+ext_))

    # for chunk_idx in range(video_chunks):
    #     if chunk_idx==video_chunks-1:
    #         ffmpeg_extract_subclip(input_file,chunk_idx*60, duration_sec, targetname=output_files[chunk_idx])
    #     else:
    #         ffmpeg_extract_subclip(input_file,chunk_idx*60, chunk_idx*60 + 60, targetname=output_files[chunk_idx])

    for chunk_idx in range(video_chunks):
            if chunk_idx==video_chunks-1:
                ffmpeg_extract_subclip(input_file,chunk_idx*60, duration_sec, targetname=output_files[chunk_idx])
            else:
                ffmpeg_extract_subclip(input_file,chunk_idx*60, chunk_idx*60 + 60, targetname=output_files[chunk_idx])

            transformed_video = EncodeVideo_obj.get_video(output_files[chunk_idx]) #get_video (or any func name): function defined by Raghav in his class to get the encoded and transformed video
            #processed_spectrogram = GetSpectrogramFromAudio_obj.get_spectrogram(video) #get_spectrogram (or any func name): function defined by Arpita in her class to get the spectrogram
            processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(output_files[chunk_idx])) #get_speech (or any func name): function defined by Joon in his class to get the transcribed speech
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
            vid_enc = True
            aud_enc = True
            try:
                pickle.dump({'transformed_video':transformed_video}, open(save_path_video_enc, 'wb'))
    #        except:
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

            os.remove(output_files[chunk_idx])
            del transformed_video, processed_speech
                #print('Couldnt encode')

    



def encode_videos(videos_path, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj):
    explicit_encoded = os.path.join(encoded_videos_path,'explicit')
    non_explicit_encoded = os.path.join(encoded_videos_path,'non_explicit')
    makedir(explicit_encoded)
    makedir(non_explicit_encoded)
    print('Encoding all videos and text...')
    not_encoded = 0
    for video_path in tqdm(videos_path):
        video = cv2.VideoCapture(video_path)

        # Get the frame rate and total number of frames
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the duration of the video in seconds
        duration_sec = frame_count / fps

        # Print the duration of the video in seconds
        print("Duration of video: {} seconds".format(duration_sec))

        # Release the video object
        video.release()
        if duration_sec > 95:
            break_larger_video_and_save(video_path, duration_sec, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj, explicit_encoded, non_explicit_encoded)            
        else:
            transformed_video = EncodeVideo_obj.get_video(video_path) #get_video (or any func name): function defined by Raghav in his class to get the encoded and transformed video
            #processed_spectrogram = GetSpectrogramFromAudio_obj.get_spectrogram(video) #get_spectrogram (or any func name): function defined by Arpita in her class to get the spectrogram
            processed_speech = TokenizeText_obj.tokenize(GetTextFromAudio_obj.get_speech(video_path)) #get_speech (or any func name): function defined by Joon in his class to get the transcribed speech
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
            #for ex in ext:
                save_path_video_enc = os.path.join(save_dir_video, video_path.split('/')[-1].replace('.mp4','_video_enc'))
                save_path_audio_enc = os.path.join(save_dir_audio, video_path.split('/')[-1].replace('.mp4','_audio_enc'))
            elif '.avi' in video_path.split('/')[-1]:
                save_path_video_enc = os.path.join(save_dir_video, video_path.split('/')[-1].replace('.avi','_video_enc'))
                save_path_audio_enc = os.path.join(save_dir_audio, video_path.split('/')[-1].replace('.avi','_audio_enc'))

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
                
            if not vid_enc or not aud_enc:
                if os.path.exists(save_path_video_enc):
                    os.remove(save_path_video_enc)
                if os.path.exists(save_path_audio_enc):
                    os.remove(save_path_audio_enc)
                print('Couldnt encode one of the above , now breaking the video into smaller ones..')
                del transformed_video, processed_speech
                not_encoded+=1
            
            

    print('Out of {} videos couldnt encode {} videos'.format(len(videos_path), not_encoded))


if __name__=='__main__':
    root_dir = '/home/shaunaks/cls_data'
    all_videos = glob.glob(os.path.join(root_dir,'processed_data/non_encoded_videos/*/*'))
    encoded_videos_path = os.path.join(root_dir,'processed_data/encoded_videos')
    EncodeVideo_obj = EncodeVideo() 
    GetTextFromAudio_obj = GetTextFromAudio()
    #GetTextFromAudio_obj = None
    TokenizeText_obj = TokenizeText()    
    #if not os.path.exists(encoded_videos_path):
    encode_videos(all_videos, encoded_videos_path, EncodeVideo_obj, GetTextFromAudio_obj, TokenizeText_obj)    

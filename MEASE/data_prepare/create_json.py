import codecs, json
import os, glob, cv2
from tqdm import tqdm
import numpy as np
import argparse
import torch

def video2numpy(videopath, frames_np=None, fps=25, is_color=True):
    """
    read numpy array from video (mp4 or avi) or write numpy array to video (only avi)
    :param videopath: filepath of video
    :param frames_np: numpy array of image frames or None, None means read
    :param fps: frame per second
    :param is_color: colorful image or gray image
    :return: numpy array for read while None for write
    """
    if frames_np is None:
        frames = []
        video_capture = cv2.VideoCapture(videopath)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                if not is_color:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # color of video which read by cv2 is BGR
                else:
                    frames.append(frame)
                
            else:
                break
        video_capture.release()
        frames_np = np.array(frames)
        return frames_np
    else:
        frames_np = frames_np.astype('uint8')
        frame_size = (frames_np.shape[2], frames_np.shape[1])
        frame_count = frames_np.shape[0]
        if len(frames_np.shape) == 3:
            is_color = False
        elif len(frames_np.shape) == 4:
            is_color = True
        else:
            raise ValueError('unknown shape for frames_np')
        out_writer = cv2.VideoWriter(
            videopath, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, frame_size, is_color)
        for frame_idx in range(frame_count):
            frame = frames_np[frame_idx]
            out_writer.write(frame)
        out_writer.release()
        return None

def traverse_file(input_folder):
    waves = []
    gDirPath = os.walk(input_folder)
    for root, dirs, files in gDirPath:
        for flac_file in files:
            if flac_file.endswith(".flac") or flac_file.endswith(".wav"):
                waves.append(os.path.join(root, flac_file))
    return waves

def read_txt(txt_file):
    txt = open(txt_file, 'r')
    keys = []
    for line in txt.readlines():
        keys.append(line[5: 26])
    return keys

def read_scp(txt_file):
    txt = open(txt_file, 'r')
    keys = []
    for line in txt.readlines():
        keys.append(line[:-1])
    return keys

def generate_json(mixture_wavpath, clean_wavpath, videos_path, output_json_path, deleted_file, split_test):
    waves = sorted(traverse_file(mixture_wavpath))
    print(waves)
    deleted_key = read_txt(deleted_file)
    dev_keys = read_scp(split_test)
    dic = {'keys': [], 'duration': [], 'key2path': {}}
    unexist_video_num = 0
    for wav in tqdm(waves):
        key = os.path.splitext(os.path.basename(wav))[0]
        print(key)
        if key not in dev_keys:
            video_path = os.path.join(videos_path, key.replace('_-10db','').replace('_-5db','').replace('_20db','').replace('_10db','').replace('_15db','').replace('_5db','').replace('_0db','')+'.pt')
            if key.replace('_-10db','').replace('_-5db','').replace('_20db','').replace('_10db','').replace('_5db','').replace('_0db','') not in deleted_key:
                if os.path.exists(video_path):
                    video = torch.load(video_path)
                    video_frames = video.shape[0]
                    if video_frames==0:
                        print("video_path=",video_path)
                    audio_duration = video_frames / 25
                    clean_wav = os.path.join(clean_wavpath, key.replace('_-10db','').replace('_-5db','').replace('_20db','').replace('_10db','').replace('_15db','').replace('_5db','').replace('_0db','')+'.wav')
                    dic['keys'].append(key)
                    dic['duration'].append(audio_duration)
                    dic['key2path'][key] = {'mixture_wave': wav, 
                                            'lip_frames': video_path,
                                            'clean_wave': clean_wav,
                                            }
                else:
                    print("unexist_video:", video_path)
                    unexist_video_num +=1
            else:
                print(key.replace('-'+snr,''))
    with codecs.open(os.path.join(output_json_path, 'misp_sim_trainset.json'), 'w') as handle:
        json.dump(dic, handle)
    print("unexist_video_num=",unexist_video_num)

def generate_test_json(mixture_wavpath, clean_wavpath, videos_path, output_json_path, deleted_file, split_test):
    waves = sorted(traverse_file(mixture_wavpath))
    print(waves)
    deleted_key = read_txt(deleted_file)
    dev_keys = read_scp(split_test)
    dic = {'keys': [], 'duration': [], 'key2path': {}}
    unexist_video_num = 0
    for wav in tqdm(waves):
        key = os.path.splitext(os.path.basename(wav))[0]
        print(key)
        if key in dev_keys:
            video_path = os.path.join(videos_path, key.replace('_-10db','').replace('_-5db','').replace('_20db','').replace('_10db','').replace('_15db','').replace('_5db','').replace('_0db','')+'.pt')
            if key.replace('_-10db','').replace('_-5db','').replace('_20db','').replace('_10db','').replace('_5db','').replace('_0db','') not in deleted_key:
                if os.path.exists(video_path):
                    video = torch.load(video_path)
                    video_frames = video.shape[0]
                    if video_frames==0:
                        print("video_path=",video_path)
                    audio_duration = video_frames / 25
                    clean_wav = os.path.join(clean_wavpath, key.replace('_-10db','').replace('_-5db','').replace('_20db','').replace('_10db','').replace('_15db','').replace('_5db','').replace('_0db','')+'.wav')
                    dic['keys'].append(key)
                    dic['duration'].append(audio_duration)
                    dic['key2path'][key] = {'mixture_wave': wav, 
                                            'lip_frames': video_path,
                                            'clean_wave': clean_wav,
                                            }
                else:
                    print("unexist_video:", video_path)
                    unexist_video_num +=1
            else:
                print(key.replace('-'+snr,''))
    with codecs.open(os.path.join(output_json_path, 'misp_sim_trainset_test.json'), 'w') as handle:
        json.dump(dic, handle)
    print("unexist_video_num=",unexist_video_num)

def generate_real_dev_json(mixture_wavpath, videos_path, output_json_path):
    waves = sorted(traverse_file(mixture_wavpath))
    #print(waves)
    dic = {'keys': [], 'duration': [], 'key2path': {}}
    unexist_video_num = 0
    for wav in tqdm(waves):
        print(wav)
        print(wav.split("/")[-2])
        key = os.path.splitext(os.path.basename(wav))[0]
        #print(key)
        #old_name = key.replace("-", "_").split("_")
       #['R80', 'S455456457', 'C03', 'I0', 'Far', 'S457', '022232', '022552']
        #video_name = f"{old_name[5]}-{old_name[0]}_{old_name[1]}_{old_name[2]}_{old_name[3]}_Middle-{old_name[6]}_{old_name[7]}"
        #video_folder = os.path.join(videos_path, f"{old_name[0]}_{old_name[1]}_{old_name[2]}_{old_name[3]}_Middle")
        subfolder = wav.split("/")[-2]
        video_path = os.path.join(videos_path, subfolder, key+'.pt')
        video_path = video_path.replace("Far","Middle")
        if os.path.exists(video_path):
            video = torch.load(video_path)
            video_frames = video.shape[0]
            if video_frames==0:
                print("video_path=",video_path)
            audio_duration = video_frames / 25
            dic['keys'].append(key)
            dic['duration'].append(audio_duration)
            dic['key2path'][key] = {'mixture_wave': wav, 
                                    'lip_frames': video_path,
                                    }
        else:
            print("unexist_video:", video_path)
            unexist_video_num +=1
    with codecs.open(os.path.join(output_json_path, 'misp_real_devset.json'), 'w') as handle:
        json.dump(dic, handle)
    print("unexist_video_num=",unexist_video_num)


def generate_real_train_json(mixture_wavpath, videos_path, output_json_path, split_test, text):
    waves = sorted(traverse_file(mixture_wavpath))
    dev_keys = read_scp(split_test)
    dic = {'keys': [], 'duration': [], 'key2path': {}}
    unexist_video_num = 0
    data_dict = {}
    text_path = text
    with open(text_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                key1 = parts[0] 
                key1 = key1.replace('_', '-', 1)
                key1 = key1.rsplit('_', 1)
                key1 = key1[0] + '-' +key1[1].replace('-', '_')
                value1 = parts[1] 
                data_dict[key1] = value1

    for wav in tqdm(waves):
        key = os.path.splitext(os.path.basename(wav))[0]
        if key not in dev_keys:
            #sub = os.path.split(os.path.dirname(wav))[-1]
            subfolder = wav.split("/")[-2]
            video_path = os.path.join(videos_path, subfolder, key+'.pt')
            video_path = video_path.replace("Far","Middle")
            #print(video_path)
            if os.path.exists(video_path):
                video = torch.load(video_path)
                    #print("video=",video)
                video_frames = video.shape[0]
                if video_frames==0:
                    print("video_path=",video_path)
                audio_duration = video_frames / 25
                text = data_dict[key.replace('_Far', '')]
                dic['keys'].append(key)
                dic['duration'].append(audio_duration)
                dic['key2path'][key] = {'mixture_wave': wav, 
                                            'lip_frames': video_path,
                                            'text': text,
                                            }
            else:
                print("unexist_video:", video_path)
                unexist_video_num +=1
    #with codecs.open(os.path.join(path, 'avse2_devset.json'), 'w') as handle:
    with codecs.open(os.path.join(output_json_path, 'misp_real_trainset.json'), 'w', encoding="utf-8") as handle:
        json.dump(dic, handle)
    print("unexist_video_num=",unexist_video_num)

def generate_real_train_test_json(mixture_wavpath, videos_path, output_json_path, split_test, text):
    waves = sorted(traverse_file(mixture_wavpath))
    dev_keys = read_scp(split_test)
    dic = {'keys': [], 'duration': [], 'key2path': {}}
    unexist_video_num = 0
    data_dict = {}
    text_path = text
    with open(text_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                key1 = parts[0] 
                key1 = key1.replace('_', '-', 1)
                key1 = key1.rsplit('_', 1)
                key1 = key1[0] + '-' +key1[1].replace('-', '_')
                value1 = parts[1] 
                data_dict[key1] = value1

    for wav in tqdm(waves):
        key = os.path.splitext(os.path.basename(wav))[0]
        if key in dev_keys:
            subfolder = wav.split("/")[-2]
            video_path = os.path.join(videos_path, subfolder, key+'.pt')
            video_path = video_path.replace("Far","Middle")
            if os.path.exists(video_path):
                video = torch.load(video_path)
                    #print("video=",video)
                video_frames = video.shape[0]
                if video_frames==0:
                    print("video_path=",video_path)
                audio_duration = video_frames / 25
                text = data_dict[key.replace('_Far', '')]
                dic['keys'].append(key)
                dic['duration'].append(audio_duration)
                dic['key2path'][key] = {'mixture_wave': wav, 
                                            'lip_frames': video_path,
                                            'text': text,
                                            }
            else:
                print(key)
    #with codecs.open(os.path.join(path, 'avse2_devset.json'), 'w') as handle:
    with codecs.open(os.path.join(output_json_path, 'misp_real_trainset_test.json'), 'w', encoding="utf-8") as handle:
        json.dump(dic, handle)
    print("unexist_video_num=",unexist_video_num)
    # mixture_wave = glob.glob(os.path.join(mixture_wavpath, '*.wav'))
    # dic = {'keys': [], 'duration': [], 'key2path': {}}
    # for wav in tqdm(mixture_wave):
    #     key = os.path.splitext(os.path.basename(wav))[0]
    #     video_path = os.path.join(videos_path, key+'.pt')
    #     clean_path = os.path.join(clean_wavpath, key+'.wav')
    #     video = video2numpy(video_path)
    #     video_frames = video.shape[0]
    #     audio_duration = video_frames / 25
    #     dic['keys'].append(key)
    #     dic['duration'].append(audio_duration)
    #     dic['key2path'][key] = {'mixture_wave': wav, 
    #                             'lip_frames': video_path,
    #                             'clean_wave': clean_path,}
    # with codecs.open(output_json_path, 'w') as handle:
    #     json.dump(dic, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create the jsonfiles of trainset and devset")
    parser.add_argument('--data', type=str, default=None,
                        help='data_type')
    parser.add_argument('--trainset_mixture_wavpath', type=str, default=None,
                        help='Directory path of trainset mixture wave ')
    parser.add_argument('--trainset_clean_wavpath', type=str, default=None,
                        help='Directory path of trainset clean wave ')
    parser.add_argument('--trainset_videos_path', type=str, default=None,
                        help='Directory path of trainset videos')
    parser.add_argument('--trainset_output_json_path', type=str, default=None,
                        help='Directory path to put output jsonfile of trainset')
    parser.add_argument('--deleted_file', type=str, default=None,
                        help='fild to del')
    parser.add_argument('--split_test', type=str, default=None,
                        help='test_set')
    parser.add_argument('--text', type=str, default=None,
                        help='text file for real_trainingset')
    args = parser.parse_args()
    if args.data == 'sim':
        generate_json(args.trainset_mixture_wavpath, args.trainset_clean_wavpath, args.trainset_videos_path, args.trainset_output_json_path, args.deleted_file, args.split_test)
        generate_test_json(args.trainset_mixture_wavpath, args.trainset_clean_wavpath, args.trainset_videos_path, args.trainset_output_json_path, args.deleted_file, args.split_test)
    if args.data == 'real_dev':
        generate_real_dev_json(args.trainset_mixture_wavpath, args.trainset_videos_path, args.trainset_output_json_path)
    if args.data == 'real_train':
        generate_real_train_json(args.trainset_mixture_wavpath, args.trainset_videos_path, args.trainset_output_json_path, args.split_test, args.text)
        generate_real_train_test_json(args.trainset_mixture_wavpath, args.trainset_videos_path, args.trainset_output_json_path, args.split_test, args.text)
    

import os

import cv2
import librosa
import librosa.display
import numpy as np
import scipy.stats


def var_name(var, all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]


def save_audio(name, source_dir, result_dir):
    input_video = source_dir + name + '.mp4'
    output_audio = result_dir + name + '/' + name + '.wav'
    if os.path.exists(output_audio):
        os.remove(output_audio)
    command = "ffmpeg -i {} -ac 1 -ar {} {} && y".format(input_video, 16000, output_audio)
    print('CMD:', command)
    os.system(command)


def feature_extraction(wav_file):
    y, sr = librosa.load(wav_file)#,duration=60*10)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    onset_envelope = librosa.onset.onset_strength(y, sr=sr)
    dtempo = librosa.beat.tempo(onset_envelope=onset_envelope, sr=sr, aggregate=None)
    pulse = librosa.beat.plp(onset_envelope=onset_envelope, sr=sr)
    pulse_lognorm = librosa.beat.plp(onset_envelope=onset_envelope, sr=sr,
                                     prior=scipy.stats.lognorm(loc=np.log(120), scale=120, s=1))

    feature_dim = 26
    feature = np.zeros([feature_dim, int(len(y) / 512) + 1])
    row = 0
    time = len(y) / sr
    fps = 30
    total_fram = int(time * fps)

    onset_envelope = onset_envelope.reshape(1, -1)
    dtempo = dtempo.reshape(1, -1)
    pulse = pulse.reshape(1, -1)
    pulse_lognorm = pulse_lognorm.reshape(1, -1)

    for single_feature in [
        spectral_centroid,
        spectral_bandwidth,
        onset_envelope,
        pulse,
        pulse_lognorm,
        dtempo,
        mfcc,
    ]:
        feature_size = np.shape(single_feature)[0]
        single_feature = single_feature / np.max(single_feature)  # Normalization
        feature[row:row + feature_size] = single_feature
        row += feature_size

    resized_feature = cv2.resize(feature, (total_fram, feature_dim))
    return resized_feature

def get_video_properties(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, size, total_frames

if __name__ == '__main__':
    result_dir = 'D:\\conductor-dataset\\dataset\\'
    source_dir = 'D:\\conductor-dataset\\dataset\\'
    video_dir = 'D:\conductor-dataset\source-video\\'
    name_list = os.listdir(source_dir)

    for name in name_list:
        print('\n--- Processing audio: {} ---'.format(name))
        fps, size, total_frames = get_video_properties(video_dir + name + '.mp4')
        print('video info: fps {}, size {}, total_frame {}, duration {}s'.format(fps, size, total_frames, int(total_frames/fps)))
        y, sr = librosa.load(source_dir + name + '/' + name + '.wav')
        print('\ty: {}, sample rate: {}'.format(len(y), sr))

        # --- Spectral features --- #
        power_spectrogram = np.abs(librosa.stft(y))
        mel_spectrogram = librosa.feature.melspectrogram(S=power_spectrogram, sr=sr)
        cqt = np.abs(librosa.cqt(y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # --- temporal features --- #
        onset_envelope = librosa.onset.onset_strength(y, sr=sr)
        dtempo = librosa.beat.tempo(onset_envelope=onset_envelope, sr=sr, aggregate=None)
        pulse = librosa.beat.plp(onset_envelope=onset_envelope, sr=sr)
        pulse_lognorm = librosa.beat.plp(onset_envelope=onset_envelope, sr=sr,
                                         prior=scipy.stats.lognorm(loc=np.log(120), scale=120, s=1))
        tempogram = librosa.feature.tempogram(onset_envelope=onset_envelope, sr=sr)

        onset_envelope = onset_envelope.reshape(1, -1)
        dtempo = dtempo.reshape(1, -1)
        pulse = pulse.reshape(1, -1)
        pulse_lognorm = pulse_lognorm.reshape(1, -1)

        # --- stack features --- #
        feature_dim = 622
        feature = np.zeros([feature_dim, int(len(y) / 512) + 1])
        row = 0
        for single_feature in [
            spectral_centroid,
            spectral_bandwidth,
            onset_envelope,
            pulse,
            pulse_lognorm,
            dtempo,
            tempogram,
            mfcc,
            mel_spectrogram,
            cqt
        ]:
            feature_size = np.shape(single_feature)[0]
            #single_feature = single_feature / np.max(single_feature)  # Normalization
            feature[row:row + feature_size] = single_feature
            print('\tfrom {} to {}:'.format(row, row + feature_size), var_name(single_feature),
                  'with shape', np.shape(single_feature),
                  'max', np.max(single_feature))
            row += feature_size

        np.save(result_dir + name + '/' + 'audiofeature.npy', feature, allow_pickle=True)
        print('audio feature save to:', result_dir + name + '/' + 'audiofeature.npy')

        resized_feature = cv2.resize(feature, (total_frames, feature_dim))
        np.save(result_dir + name + '/' + 'audiofeature-30fps.npy', resized_feature, allow_pickle=True)
        print('30fps audio feature save to:', result_dir + name + '/' + 'audiofeature-30fps.npy')
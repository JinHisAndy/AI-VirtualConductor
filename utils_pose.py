import numpy as np
import tqdm
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
import datetime

line_pairs = [
    # (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (0, 17), (0, 18),  # simple head
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # arm
    (18, 19),
    # (19, 11), (19, 12),
    # (11, 13), (12, 14), (13, 15), (14, 16),
    # (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
]
lower_body = [
    # 0, # nose
    1, 2, 3, 4,  # face
    # 5,6,7,8,9,10, # arm
    11, 12,  # hips
    13, 14, 15, 16,  # leg
    # 17,18,19, # head-neck-hip
    20, 21, 22, 23, 24, 25  # foot
]
upper_body = [
    0,  # nose
    # 1,2,3,4, # face
    5, 6, 7, 8, 9, 10,  # arm
    # 11, 12,  # hips
    # 13, 14, 15, 16,  # leg
    17, 18, 19,  # head-neck-hip
    # 20, 21, 22, 23, 24, 25  # foot
]



def padding_results(keypoints):
    # expected input: (time, 20)
    global_result = np.zeros([len(keypoints), 17, 2])
    for i in range(7):  # 10 keypoints
        for j in range(2):  # 2 axis
            global_result[:, upper_body[i], j] = keypoints[:, i * 2 + j]

    # 二分之上本身 unit
    unit = (global_result[:,8,1] - global_result[:,0,1])/2

    # 脸
    global_result[:, 1, 0] = global_result[:, 0, 0]+0.25*unit
    global_result[:, 1, 1] = global_result[:, 0, 1]-0.15*unit
    global_result[:, 2, 0] = global_result[:, 0, 0]-0.25*unit
    global_result[:, 2, 1] = global_result[:, 0, 1]-0.15*unit
    global_result[:, 3, 0] = global_result[:, 0, 0]+0.5*unit
    global_result[:, 3, 1] = global_result[:, 0, 1]-0.15*unit
    global_result[:, 4, 0] = global_result[:, 0, 0]-0.5*unit
    global_result[:, 4, 1] = global_result[:, 0, 1]-0.15*unit


    # 两腰：11,12
    global_result[:,11,0] = keypoints[:, 9*2+0]+0.5*unit
    global_result[:,11,1] = keypoints[:, 9*2+1]
    global_result[:,12,0] = keypoints[:, 9*2+0]-0.5*unit
    global_result[:,12,1] = keypoints[:, 9*2+1]

    # 两膝：13,14
    global_result[:, 13, 0] = keypoints[:, 9 * 2 + 0] + 0.5 * unit
    global_result[:, 13, 1] = keypoints[:, 9 * 2 + 1] + 1.5*unit
    global_result[:, 14, 0] = keypoints[:, 9 * 2 + 0] - 0.5 * unit
    global_result[:, 14, 1] = keypoints[:, 9 * 2 + 1] + 1.5*unit

    # 两脚： 15, 16
    global_result[:, 15, 0] = keypoints[:, 9 * 2 + 0] + 0.5 * unit
    global_result[:, 15, 1] = keypoints[:, 9 * 2 + 1] + 2 * 1.5*unit
    global_result[:, 16, 0] = keypoints[:, 9 * 2 + 0] - 0.5 * unit
    global_result[:, 16, 1] = keypoints[:, 9 * 2 + 1] + 2 * 1.5*unit


    # to (-1,1)
    global_result[:, :, 0] = global_result[:, :, 0] -np.mean(global_result[:, 0, 0])
    global_result[:, :, 1] = global_result[:, :, 1] -np.mean(global_result[:, 0, 1])-0.8




    return global_result


def reshape_keypoints(keypoints):
    # drop lower body keypoints and reshape: from [num_frames, 26, 2] to [num_frames, 20]
    upper_result = np.zeros([len(keypoints), 20])
    for i in range(10):  # 10 keypoints
        for j in range(2):  # 2 axis
            upper_result[:, i * 2 + j] = keypoints[:, upper_body[i], j]
    return upper_result


def _reshape_result(upper_result):
    # pad lower body keypoints and reshape: from [num_frames, 20] to [num_frames, 26, 2]
    global_result = np.zeros([len(upper_result), 26, 2])
    for i in range(10):  # 10 keypoints
        for j in range(2):  # 2 axis
            global_result[:, upper_body[i], j] = upper_result[:, i * 2 + j]
    return global_result


def modify_26_keypoints(keypoints, conv_range=5, conv_range_hips=50):
    def np_move_avg(a, n, mode="same"):
        if n == 0:
            return a
        else:
            return np.convolve(a, np.ones((n,)) / n, mode=mode)

    smmothed = np.zeros_like(keypoints)
    for keypoint in range(len(smmothed[0])):
        x = keypoints[:, keypoint, 0]
        y = keypoints[:, keypoint, 1]
        if keypoint in [11, 12, 19]:
            x_conv = np_move_avg(x, conv_range_hips)
            y_conv = np_move_avg(y, conv_range_hips)
        else:
            x_conv = np_move_avg(x, conv_range)
            y_conv = np_move_avg(y, conv_range)
        smmothed[:, keypoint, 0] = x_conv
        smmothed[:, keypoint, 1] = y_conv

    return smmothed


def modify_20_keypoints(keypoints, conv_range=5):
    def np_move_avg(a, n, mode="same"):
        if n == 0:
            return a
        else:
            return np.convolve(a, np.ones((n,)) / n, mode=mode)

    smmothed = np.zeros_like(keypoints)
    for i in range(len(smmothed[0])):
        x = keypoints[:, i]
        x_conv = np_move_avg(x, conv_range)
        smmothed[:, i] = x_conv

    return smmothed


def normalization(smoothed_result, w, h):
    # Normalize so that [0, w] is mapped to [0, 1], while preserving the aspect ratio
    norm_result = smoothed_result / w * 2 - [1, h / w]
    return (norm_result + 1) / 2


def read_result(name, result_dir='C:/Users/wahaha/Desktop/ccdelworkspace/AlphaPose-master/runs/result/'):
    # read alphapose 2D pose estimation results
    # INPUT: name - video name
    # OUTPUT: keypoints - keypoints sequence with shape [num_frames, 26, 2]

    outputpath = result_dir + name + '/'

    pose_result = np.load(outputpath + 'alphapose-results.npy', allow_pickle=True)
    keypoints = np.zeros([len(pose_result), 26, 2])
    for i in range(len(pose_result)):
        frame_dict = pose_result[i]
        frame_idx = int(frame_dict['imgname'][:-4])
        frame_result = frame_dict['result']
        if frame_result is None or len(frame_result) == 0:
            continue
        else:
            fram_kpts = frame_result[0]['keypoints'].numpy()
            keypoints[frame_idx] = fram_kpts

    keypoints = modify_26_keypoints(keypoints, conv_range=5, conv_range_hips=50)  # convolution smoothing
    keypoints[:, 19, :] = np.average(keypoints[:, [11, 12, 19], :], axis=1)  # averaging hips
    keypoints = normalization(keypoints, w=852, h=480)  # Normalize to [0,1]
    keypoints = reshape_keypoints(keypoints)  # drop lower body keypoints and reshape to [num_frames, 20]

    return keypoints


def show_pose(keypoints, descriptions, name, video_save_dir=None):
    # INPUTS: keypoints = [keypoint, keypoint, ...]
    # keypoint: [num_frame, 20]

    time_stamp = datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')

    for numfig in range(len(keypoints)):
        keypoints[numfig] = _reshape_result(keypoints[numfig])

    figsize = 500
    num_fig = len(keypoints)
    num_frame = len(keypoints[0])

    if video_save_dir is not None:
        wirter = cv2.VideoWriter(video_save_dir + name + '.avi', 0, cv2.VideoWriter_fourcc(*'XVID'), 30,
                                 (int(num_fig * figsize), figsize))

    hand_trace = []
    for numfig in range(num_fig):
        hand_trace.append([])
    hand_trace_len = 25

    for frame in tqdm.tqdm(range(num_frame)):
        img = np.ones([figsize, num_fig * figsize, 3], np.uint8) * 255
        for numfig in range(num_fig):
            keypoint = keypoints[numfig] * figsize
            lines = []

            # --- draw hand trace --- #
            for i in range(min(frame, hand_trace_len)):
                x, y = hand_trace[numfig][i]
                cv2.circle(img, (x, y), 1, (int(255 - (hand_trace_len - i) / hand_trace_len * 200),
                                            int(255 - (hand_trace_len - i) / hand_trace_len * 200),
                                            int(255 - (hand_trace_len - i) / hand_trace_len * 200)), 2)
            # --- draw skeleton --- #
            for i, (start, end) in enumerate(line_pairs):
                line = np.array([[keypoint[frame, start, 0] + numfig * figsize, keypoint[frame, start, 1]],
                                 [keypoint[frame, end, 0] + numfig * figsize, keypoint[frame, end, 1]]],
                                np.int32).reshape((-1, 1, 2))
                lines.append(line)
            cv2.polylines(img, lines, True, (150, 200, 50), thickness=3)

            # --- draw points and save hand trace --- #
            for point in range(20):
                x, y = int(keypoint[frame, point, 0] + numfig * figsize), int(keypoint[frame, point, 1])
                cv2.circle(img, (x, y), 2, (0, 0, 0), 2)
                if point in [9, 10]:
                    hand_trace[numfig].insert(0, (x, y))

            # --- put text --- #
            cv2.putText(img, descriptions[numfig], (numfig * figsize + 5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(50, 50, 50), 1)
            cv2.putText(img, 'test time: ' + time_stamp, (numfig * figsize + 5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(50, 50, 50), 1)
            cv2.putText(img, 'frame ' + str(frame), (numfig * figsize + 5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(50, 50, 50), 1)
            cv2.putText(img, name, (numfig * figsize+5, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (50, 50, 50), 1)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if video_save_dir is not None:
            wirter.write(img)
    if video_save_dir is not None:
        wirter.release()
    cv2.destroyAllWindows()


def filter(keypoints, freq_low=0.4, freq_high=5, sample_rate=25, mode='high pass'):
    # Calculate pose data from given npy file, seperate the signals with 2 filters into 3 parts: high_pass pose,
    # low_pass pose, and noise.

    # the raw_pose should have a shape of [time, 17, 3]
    # raw_pose = np.load(npyfile)[start * sample_rate: end * sample_rate]

    highpass_pose = np.zeros_like(keypoints)
    lowpass_pose = np.zeros_like(keypoints)
    noise_pose = np.zeros_like(keypoints)

    # define the frequncy thresholds, and get filters accordingly
    wnl = 2 * freq_low / sample_rate
    wnh = 2 * freq_high / sample_rate

    high_b, high_a = signal.butter(8, [wnl, wnh], 'bandpass', output='ba')
    noise_b, noise_a = signal.butter(8, wnh, 'highpass', output='ba')
    low_b, low_a = signal.butter(8, wnl, 'lowpass', output='ba')

    # seperate pose signals
    for kept in range(20):
        highpass_pose[:, kept] = signal.filtfilt(high_b, high_a, keypoints[:, kept])
        noise_pose[:, kept] = signal.filtfilt(noise_b, noise_a, keypoints[:, kept])
        lowpass_pose[:, kept] = signal.filtfilt(low_b, low_a, keypoints[:, kept])

    if mode == 'high pass':
        return highpass_pose
    elif mode == 'low pass':
        return lowpass_pose
    else:
        Y = np.column_stack((highpass_pose, noise_pose))
        return Y


if __name__ == '__main__':
    my_keypoints = np.load(r'test\results\test_result_2021_03_29__10_54_20\Aiva_Sinfonietta_Orchestra_Olivier_Hecho_Aiva_Symphonic_Fantasy_for_Orchestra_in_G_Sharp_MinorOp7The_Awakening.mp3.npy')
    my_keypoints = padding_results(my_keypoints)
    print(np.shape(my_keypoints))

    keypoints = np.load('keypoints.npy')
    print(np.shape(keypoints))

    for i in range(len(keypoints)):
        plt.figure(figsize=(16,8))
        for j in range(17):
            plt.subplot(1,2,1)
            plt.scatter(my_keypoints[i, j, 0], my_keypoints[i, j, 1])
            plt.text(my_keypoints[i, j, 0], my_keypoints[i, j, 1],str(j))
            plt.xlim([-0.5,0.5])
            plt.ylim([0,-1])

            plt.subplot(1,2,2)
            plt.scatter(keypoints[i, j, 0], keypoints[i, j, 1])
            plt.text(keypoints[i, j, 0], keypoints[i, j, 1],str(j))
            plt.xlim([-0.5,0.5])
            plt.ylim([0,-1])
        plt.savefig('temp.png')
        plt.close()

        img = cv2.imread('temp.png')
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





















    input()

    result_dir = 'DS_dataset/'
    name_list = os.listdir(result_dir)
    for name in name_list:
        print('\n--- Processing pose: {} ---'.format(name))
        my_keypoints = read_result(name, result_dir=result_dir)

        high_pass = filter(my_keypoints, mode='high pass')
        low_pass = filter(my_keypoints, mode='low pass')

        keypoints_mean = np.mean(low_pass, axis=0)
        # np.save('keypoints_mean.npy', keypoints_mean)
        high_pass += keypoints_mean

        # show_pose([keypoints, high_pass, low_pass], ['keypoints', 'high_pass', 'low_pass'])

        # np.save(result_dir + name + '/' + 'high-pass-results.npy', high_pass, allow_pickle=True)
        # np.save(result_dir + name + '/' + 'low-pass-results.npy', low_pass, allow_pickle=True)
        show_pose([my_keypoints, high_pass, low_pass], ['alphapose', 'highpass', 'lowpass'])
        print('smoothed normalized upper body result save to:', result_dir + name + '/' + 'norm-results.npy')

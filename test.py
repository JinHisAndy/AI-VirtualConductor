import datetime
import subprocess
import torch
from moviepy.editor import *

from dataset import *
from utils_pose import show_pose, modify_20_keypoints


def test(model_dir, testset, descriptions, video_save_dir):
    dataset = TestDataset(test_samles_dir=testset)
    testloader = DataLoader(dataset=dataset, batch_size=1)
    print('loading model...')
    G = torch.load(model_dir).cuda()
    G.eval()

    for step, (music_feature, name) in enumerate(testloader):
        name = name[0]
        print('\n### ------ evaluating {}/{} ------ ###'.format(step, len(dataset)))
        print('name:', name)

        music_feature = music_feature.transpose(1, 2)
        var_x = music_feature.float().cuda()
        predicted_pose, hx = G(var_x, None)

        predicted_pose = predicted_pose[0].detach().cpu().numpy() / 4
        predicted_pose_mean = np.mean(predicted_pose, axis=0)
        predicted_pose -= predicted_pose_mean
        predicted_pose = modify_20_keypoints(predicted_pose)
        keypoints_mean = np.load('keypoints_mean.npy', allow_pickle=True)
        predicted_pose += keypoints_mean
        # pad17pose = padding_results(predicted_pose)

        np.save(video_save_dir + name, predicted_pose)
        # np.save(video_save_dir+'【17】'+name,pad17pose)

        print('rendering video...')
        show_pose([predicted_pose], descriptions=descriptions, name=name, video_save_dir=video_save_dir)

        print('mix audio and video...')
        video = VideoFileClip(video_save_dir + name + '.avi')
        video = video.set_audio((AudioFileClip(testset + name)))
        video.write_videofile(video_save_dir + name + '.mp4')
        os.remove(video_save_dir + name + '.avi')
    print('test finished')


def video2mp3(file_name):
    """
    将视频转为音频
    :param file_name: 传入视频文件的路径
    :return:
    """
    outfile_name = file_name.split('.')[0] + '.mp3'
    cmd = 'ffmpeg -i ' + file_name + ' -f mp3 ' + outfile_name
    print(cmd)
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    model_dir = 'checkpoints/G_globalstep34000.pt'
    time_stamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    video_save_dir = 'test\\results\\' + 'test_result_' + time_stamp + '/'
    os.mkdir(video_save_dir)
    test(model_dir=model_dir,
         testset='test\\testset\\',
         descriptions=['perceptual+adversrial'],
         video_save_dir=video_save_dir)

import time

import matplotlib
import torchvision
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd as autograd


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True


def plot_xy(predicted_pose, var_y, Green=None, Blue=None):
    if predicted_pose is not None:
        predicted_pose = predicted_pose.cpu().detach().numpy()
    if Green is not None:
        Green = Green.cpu().detach().numpy()
    if Blue is not None:
        Blue = Blue.cpu().detach().numpy()
    var_y = var_y.cpu().detach().numpy()

    fig = plt.figure(figsize=(15, 8))
    for c in range(20):
        plt.subplot(5, 4, c + 1)
        plt.plot(var_y[0, :, c], linewidth=0.5, color='gray')

        if predicted_pose is not None:
            plt.plot(predicted_pose[0, :, c], linewidth=0.5, color='r')

        if Green is not None:
            plt.plot(Green[0, :], linewidth=0.5, color='g')
        if Blue is not None:
            plt.plot(Blue[0, :], linewidth=0.5, color='b')
        plt.ylim(-1.2, 1.2)
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.05)
    time.sleep(0.5)
    return fig


def plot_hidden_feature(hidden_feature):
    hidden_feature = hidden_feature.transpose(1,2).cpu().detach().numpy()
    hidden_feature.astype(np.float32)
    # fig = plt.matshow(np.transpose(hidden_feature[0]))
    fig = plt.matshow(hidden_feature[0])
    plt.colorbar(fig)
    image_path = 'temp_.png'
    plt.savefig(image_path)
    image_PIL = Image.open(image_path)
    img = np.array(image_PIL)
    plt.close()
    time.sleep(0.5)
    return img


def plot_Dc_kernals(Dc, writer, global_step):
    for i, (name, param) in enumerate(Dc.named_parameters()):
        if 'conv_1.0' in name and 'weight' in name and 'BN' not in name:
            if len(param.size()) == 1:
                continue
            in_channels = param.size()[1]
            out_channels = param.size()[0]  # 输出通道，表示卷积核的个数

            k_w, k_h = param.size()[1], param.size()[2]  # 卷积核的尺寸
            kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
            kernel_grid = torchvision.utils.make_grid(kernel_all, normalize=True, nrow=in_channels)
            writer.add_image(f'{name}_all', kernel_grid, global_step=global_step)


def calc_gradient_penalty_Dr(Dr, var_x,real_data, fake_data):
    LAMBDA = 10
    '''alpha = torch.ones(real_data.size()[0], real_data.size()[1], real_data.size()[2]).cuda()
    rand = np.random.rand()
    alpha = alpha * rand

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)'''

    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size()).cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = Dr(var_x,interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def calc_gradient_penalty_Dc(Dc, real_music, real_data, fake_data):
    LAMBDA = 10
    '''alpha = torch.ones(real_data.size()[0], real_data.size()[1], real_data.size()[2]).cuda()
    rand = np.random.rand()
    alpha = alpha * rand

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)'''

    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size()).cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ ,_, _= Dc(real_music,interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
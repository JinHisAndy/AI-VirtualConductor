import torch.backends.cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from models.AMCNet import AMCNet_shallow
from models.discriminator import RealFakeDiscriminator
from models.generator import Generator_Sampling
from utils_train import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(319)
torch.cuda.manual_seed(319)
np.random.seed(319)

if __name__ == '__main__':

    global_step = 0

    # dataset config
    sample_limit = None
    batch_size = 16
    sample_length = 1001
    part_length = 901
    remain_length = 100

    # training config
    epoch_num = 50000
    plot_step = 50
    save_step = 1000
    mode = 'high'

    CRITIC_ITERS = 5
    TrainG = 1
    L3_prtrain = False

    training_set = ConductorDataset(sample_length=sample_length, dataset_dir='dataset\\', sample_limit=sample_limit,
                                    mode=mode)
    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    print('dataset initialized, length {}, batch size {}, and {} step per epoch'
          .format(training_set.__len__(), batch_size, training_set.__len__() / batch_size))

    AMCNet = AMCNet_shallow().cuda()
    # AMCNet.music_encoder = torch.load('checkpoints/AMCNet_globalstep137000.pt').cuda().music_encoder
    AMCNet_perceptual = torch.load('checkpoints/AMCNet_globalstep87000.pt').cuda().pose_encoder
    optimizer_AMC = torch.optim.Adam(AMCNet.parameters(), lr=0.001)

    Dr = RealFakeDiscriminator().cuda()
    # Dr = DoubleAMCDiscriminator().cuda()
    # Dr.ACM_Freeze=torch.load('checkpoints/Dc_globalstep87000.pt').cuda()
    # freeze(Dr.ACM_Freeze)
    # Dr.ACM_unFreeze=torch.load('checkpoints/Dc_globalstep87000.pt').cuda()
    optimizer_Dr = torch.optim.RMSprop(Dr.parameters(), lr=0.0005)

    G = Generator_Sampling().cuda()
    G.music_encoder = torch.load('checkpoints/AMCNet_globalstep87000.pt').cuda().music_encoder
    # freeze(G.music_encoder)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0005)

    MSE_criterion = nn.MSELoss()
    L1_criterion = nn.L1Loss()
    BCE_criterion = nn.BCELoss()

    writer = SummaryWriter()

    for e in range(epoch_num):
        pbar = tqdm.tqdm(enumerate(train_loader), total=train_loader.__len__())

        label_match = torch.ones([batch_size, part_length, 1]).cuda() - 0.0001
        label_mismatch = torch.zeros([batch_size, part_length, 1]).cuda() + 0.0001

        for step, (batch_x, batch_y) in pbar:
            if batch_x.size()[0] != batch_size:
                continue

            var_x = batch_x.cuda()
            var_y = batch_y.cuda()

            # ----------------------- #
            #     train Generator     #
            # ----------------------- #
            optimizer_G.zero_grad()
            fake_pose, hx = G(var_x, var_y)
            fake_output_Dr, _ = Dr(var_x, fake_pose)
            real_feature_all, real_feature = AMCNet_perceptual(var_y)
            fake_feature_all, fake_feature = AMCNet_perceptual(fake_pose)

            loss_Gadv_Dr = -torch.mean(fake_output_Dr)
            loss_perceptual = MSE_criterion(fake_feature_all, real_feature_all)
            loss_pred = MSE_criterion(fake_pose, var_y)

            loss_pred *= 0.1  # 0.005
            loss_Gadv_Dr *= 0.005 / 2  # 2
            loss_perceptual *= 0.5 * 0.005 / 0.03  # 0.03
            loss_G = loss_pred + loss_Gadv_Dr + loss_perceptual

            loss_pred.backward(retain_graph=True)
            pred_norm_G = 0
            for p in filter(lambda p: p.grad is not None, G.parameters()):
                param_norm = p.grad.data.norm(2)
                pred_norm_G += param_norm.item() ** 2
            pred_norm_G = pred_norm_G ** (1. / 2)

            loss_Gadv_Dr.backward(retain_graph=True)
            adv_norm_G = 0
            for p in filter(lambda p: p.grad is not None, G.parameters()):
                param_norm = p.grad.data.norm(2)
                adv_norm_G += param_norm.item() ** 2
            adv_norm_G = adv_norm_G ** (1. / 2)

            loss_perceptual.backward(retain_graph=True)
            percep_norm_G = 0
            for p in filter(lambda p: p.grad is not None, G.parameters()):
                param_norm = p.grad.data.norm(2)
                percep_norm_G += param_norm.item() ** 2
            percep_norm_G = percep_norm_G ** (1. / 2)

            total_norm_G = 0
            loss_G.backward(retain_graph=True)
            for p in filter(lambda p: p.grad is not None, G.parameters()):
                param_norm = p.grad.data.norm(2)
                total_norm_G += param_norm.item() ** 2
            total_norm_G = total_norm_G ** (1. / 2)

            optimizer_G.step()

            for critic_i in range(CRITIC_ITERS):
                # ----------------------- #
                #        train Dr         #
                # ----------------------- #
                optimizer_Dr.zero_grad()

                real_output_Dr, fmap_Dr_real = Dr(var_x, var_y)
                fake_output_Dr, fmap_Dr_fake = Dr(var_x, fake_pose.detach())

                loss_Dr_real = -torch.mean(real_output_Dr)
                loss_Dr_fake = torch.mean(fake_output_Dr)

                gradient_penalty_Dr = calc_gradient_penalty_Dr(Dr, var_x.data, var_y.data, fake_pose.data)

                loss_Dr = loss_Dr_real + loss_Dr_fake + gradient_penalty_Dr
                loss_Dr.backward()
                optimizer_Dr.step()

            '''if L3_prtrain:

                # ----------------------- #
                #     train AMC (L3)      #
                # ----------------------- #
                optimizer_AMC.zero_grad()

                random_start = int(np.random.rand() * remain_length)
                random_end = random_start + part_length
                random_scale_pose = np.random.random()
                var_x1 = var_x[:, :part_length, :]
                var_x2 = var_x[:, random_start:random_end, :]
                var_y1 = var_y[:, :part_length, :] * random_scale_pose
                var_y2 = var_y[:, random_start:random_end, :] * random_scale_pose

                x1y1, _, _, _ = AMCNet(var_x1, var_y1)
                x1y2, _, _, _ = AMCNet(var_x1, var_y2)
                x2y2,real_output_Dc , feature_cat_real, AMC_h1_real = AMCNet(var_x2, var_y2)
                x2y1,fake_output_Dc , feature_cat_fake, AMC_h1_fake= AMCNet(var_x2, var_y1)

                loss_match1 = MSE_criterion(x1y1, label_match)
                loss_match2 = MSE_criterion(x2y2, label_match)
                loss_mismatch1 = MSE_criterion(x1y2, label_mismatch)
                loss_mismatch2 = MSE_criterion(x2y1, label_mismatch)

                loss_L3 = loss_match1 + loss_match2 + loss_mismatch1 + loss_mismatch2

                loss_match_Dc = -torch.mean(real_output_Dc)
                loss_mismatch_Dc= torch.mean(real_output_Dc)
                loss_Dc = loss_match_Dc+loss_mismatch_Dc

                loss_AMC = loss_L3 + loss_Dc
                loss_AMC.backward()
                optimizer_AMC.step()'''

            ###############################################
            #                    Logging                  #
            ###############################################
            writer.add_scalars('1. Generator loss', {'MSE': loss_pred.item(),
                                                     'percep': loss_perceptual.item()}, global_step)
            writer.add_scalars('1. gradient', {'total': total_norm_G,
                                               'pred': pred_norm_G,
                                               'adv': adv_norm_G,
                                               'percep': percep_norm_G}, global_step)
            writer.add_scalars('2. Dr output', {'real_output_Dr': torch.mean(real_output_Dr).item(),
                                                'fake_output_Dr': torch.mean(fake_output_Dr).item(),
                                                'W_distance': torch.mean(real_output_Dr).item() -
                                                              torch.mean(fake_output_Dr).item()}, global_step)
            writer.add_scalars('3.variences', {'GT': torch.var(var_y),
                                               'pred': torch.var(fake_pose),
                                               'diff': torch.var(var_y) - torch.var(fake_pose)}, global_step)
            '''
            writer.add_scalars('3. AMC-Dc output', {'real_output_Dc': torch.mean(real_output_Dc).item(),
                                                    'fake_output_Dc': torch.mean(fake_output_Dc).item(),
                                                    'W_distance': torch.mean(real_output_Dc).item() -
                                                                  torch.mean(fake_output_Dc).item()}, global_step)
            writer.add_scalars('3. AMC-L3 output',
                               {'match_output_L3': 0.5 * torch.mean(x1y1).item() + 0.5 * torch.mean(x2y2).item(),
                                'mismatch_output_L3': 0.5 * torch.mean(x1y2).item() + 0.5 * torch.mean(x2y1).item(),
                                'loss_L3': loss_L3.item()}, global_step)
            '''

            pbar.set_description(
                'Epoch: %d Step: %d Global_step: %d | loss_pred: %.10f' % (e, step, global_step, loss_pred.item()))
            global_step += 1

            if global_step % plot_step == 0:
                # --- G
                img_hx = plot_hidden_feature(hx)
                writer.add_image("1.1 Generator_hx", img_hx, global_step, dataformats='HWC')
                fig_xy = plot_xy(fake_pose, var_y, Green=real_output_Dr, Blue=fake_output_Dr)
                writer.add_figure("1.2 training sample", fig_xy, global_step)
                # --- Dr
                feature_maps_Dr = plot_hidden_feature(torch.cat([fmap_Dr_real, fmap_Dr_fake], dim=1))
                writer.add_image('2 feature_maps_Dr', feature_maps_Dr, global_step, dataformats='HWC')
                # --- Perceptual
                feature_maps_perceptual = plot_hidden_feature(torch.cat([real_feature_all, fake_feature_all], dim=2))
                writer.add_image('3 feature_maps_perceptual', feature_maps_perceptual, global_step, dataformats='HWC')

                # --- AMCNet
                '''Dc_L3features = plot_hidden_feature(torch.cat([feature_cat_real, feature_cat_fake], dim=1))
                writer.add_image('3.1 Dc_L3features', Dc_L3features, global_step, dataformats='HWC')
                Dc_h1 = plot_hidden_feature(torch.cat([AMC_h1_real, AMC_h1_fake], dim=1))
                writer.add_image('3.2 Dc_h1', Dc_h1, global_step, dataformats='HWC')
                Dc_outputs = plot_hidden_feature(torch.cat([real_output_Dc, fake_output_Dc], dim=1))
                writer.add_image('3.3 Dc_outputs', Dc_outputs, global_step, dataformats='HWC')'''

            if global_step % save_step == 0:
                torch.save(G, 'checkpoints\\G_globalstep{}.pt'.format(global_step))
                torch.save(Dr, 'checkpoints\\Dr_globalstep{}.pt'.format(global_step))
                # torch.save(AMCNet, 'checkpoints\\Dc_globalstep{}.pt'.format(global_step))

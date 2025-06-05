from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from loss.InfoNCE import ContrastiveLoss
from loss.losses import DiceLoss, DiceBCELoss, PredictionMapDistillation, CLoss
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, args, loader, my_model_T, my_model_S,
                my_loss=None, my_lossv=None, ckp=None):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp  # ckp: checkpoint
        self.loader_train = loader.loader_train # 
        self.loader_test = loader.loader_test
        self.modelT = my_model_T
        self.modelS = my_model_S
        self.loss = my_loss
        self.lossv = my_lossv
        self.optimizer = utility.make_optimizer(args, self.modelS)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        # enoch = self.scheduler.last_epoch + 1
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()


        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.modelS.train()
        self.modelT.eval()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        noise_level = [0.05, 0.1, 0.15, 0.2, 0.25]
        # TEMP
        for batch, (lr, hr, _, _1) in enumerate(self.loader_train):
            lr_bic = F.interpolate(lr, scale_factor=2)
            index = random.randint(0, 4)
            noise = torch.randn(lr.size()) * noise_level[index]
            lr_noisy = lr + noise
            lr, hr, lr_bic, lr_noisy = self.prepare(lr, hr, lr_bic, lr_noisy)
            batch_size = lr.shape[0]

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            bottleneck_Feats_S, p2_S, d2_S, s_pred = self.modelS(lr_noisy)

            with torch.no_grad():
                bottleneck_Feats_T, p2_T, d4_T, t_pred = self.modelT(lr)

            loss_s = self.loss(s_pred, hr)
            bottleneck_Feats_T_resized = F.interpolate(bottleneck_Feats_T, size=(32, 32), mode='bilinear',
                                                       align_corners=False)
            conv_reduce_channels = nn.Conv2d(1024, 256, kernel_size=1).to(device)
            conv_reduce_channels2 = nn.Conv2d(128, 128, kernel_size=1).to(device)
            conv_reduce_channels3 = nn.Conv2d(64, 64, kernel_size=1).to(device)


            bottleneck_Feats_T_reduced = conv_reduce_channels(bottleneck_Feats_T_resized).to(device)

            # Calculate bottleneck to bottleneck contrastive loss
            info_nce_loss = CLoss()

            bottleneck_Feats_S = bottleneck_Feats_S.view(batch_size, -1)
            bottleneck_Feats_T_reduced = bottleneck_Feats_T_reduced.view(batch_size, -1)

            p2_T = conv_reduce_channels2(p2_T).to(device)

            p2_S = p2_S.view(batch_size, -1)
            p2_T = p2_T.view(batch_size, -1)

            d2_S = conv_reduce_channels3(d2_S).to(device)
            d2_S = d2_S.view(batch_size, -1)
            d4_T = d4_T.view(batch_size, -1)

            b_b_Con_loss = info_nce_loss(bottleneck_Feats_S, bottleneck_Feats_T_reduced)
            loss_P2 = info_nce_loss(p2_S, p2_T)
            loss_decoder = info_nce_loss(d2_S, d4_T)

            loss_pmd = 0.8 * self.loss(s_pred, t_pred) + 0.2 * F.mse_loss((lr_bic - t_pred), (lr_bic - s_pred))
            # loss = 0.5 * loss_s + 0.5 * loss_pmd
            # loss = loss_s + loss_pmd + 0.1 * (b_b_Con_loss + loss_decoder + loss_P2)
            loss = loss_s + loss_pmd + 0.1 * b_b_Con_loss

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            # self.scheduler.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    # self.loss.display_loss(batch),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        return loss_s.item(), loss_pmd.item(), 0.1 * b_b_Con_loss.item(), loss.item()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.modelS.eval()
        if not self.args.test_only:
            self.lossv.start_log()

        scale = self.args.scale
        timer_test = utility.timer()

        # if self.args.save_results: self.ckp.begin_background()
        noise_level = [0.05, 0.1, 0.15, 0.2, 0.25]
        for (lr, hr, filename, params) in tqdm(self.loader_test, ncols=80):
            lr_bic = F.interpolate(lr, scale_factor=2)
            index = random.randint(0, 4)
            noise = torch.randn(lr.size()) * noise_level[index]
            lr_noisy = lr + noise
            lr, hr, lr_bic, lr_noisy = self.prepare(lr, hr, lr_bic, lr_noisy)

            with torch.no_grad():
                bottleneck_Feats_S, p2_S, d2_S, s_pred = self.modelS(lr_noisy)
            if not self.args.test_only:
                lossv = self.lossv(s_pred, hr)

            save_list = [s_pred]
            if not self.args.apply_field_data:
                self.ckp.log[-1] += utility.calc_psnr(
                    s_pred, hr, scale
                )

            if self.args.save_results:
                self.ckp.save_results(filename[0], save_list, params)

        if not self.args.apply_field_data:
            self.ckp.log[-1] /= len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                'Seis', 
                scale,
                self.ckp.log[-1],
                best[0],
                best[1] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if not self.args.test_only:
            self.lossv.end_log(len(self.loader_test))


        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        return [a.to(device) for a in args]

    def terminate(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args.epochs

class Trainer_single():
    def __init__(self, args, loader, my_model,
                my_loss=None, my_lossv=None, ckp=None):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp  # ckp: checkpoint
        self.loader_train = loader.loader_train #
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.lossv = my_lossv
        self.optimizer = utility.make_optimizer(args, self.model)
        # self.optimizer = torch.optim.Adam(self.modelS.parameters(), betas=(0.9, 0.999), lr=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 100, 120], gamma=0.1)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        # enoch = self.scheduler.last_epoch + 1
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        # lr = self.optimizer.param_groups[0]['lr']

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        for batch, (lr, hr, _, _1) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)

            loss = self.loss(sr, hr)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            # self.scheduler.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1)
        )
        self.model.eval()
        if not self.args.test_only:
            self.lossv.start_log()

        scale = self.args.scale
        timer_test = utility.timer()

        # if self.args.save_results: self.ckp.begin_background()
        for (lr, hr, filename, params) in tqdm(self.loader_test, ncols=80):
            lr, hr = self.prepare(lr, hr)

            _, _, _, sr = self.model(lr)
            if not self.args.test_only:
                lossv = self.lossv(sr, hr)

            save_list = [sr]
            if not self.args.apply_field_data:
                self.ckp.log[-1] += utility.calc_psnr(
                    sr, hr, scale
                )

            if self.args.save_results:
                self.ckp.save_results(filename[0], save_list, params)

        if not self.args.apply_field_data:
            self.ckp.log[-1] /= len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                'Seis',
                scale,
                self.ckp.log[-1],
                best[0],
                best[1] + 1
                )
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if not self.args.test_only:
            self.lossv.end_log(len(self.loader_test))

        # if self.args.save_results:
        #     self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        return [a.to(device) for a in args]

    def terminate(self):
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args.epochs
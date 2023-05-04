# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/19 15:28
import argparse
import os
from tqdm import tqdm
import pickle
import json

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from Load31_data import RAIL_Data
from config import GlobalConfig
from single_model_gaf import Decoder

torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

OPTIONS = argparse.ArgumentParser()
OPTIONS.add_argument('--id', type=str, default='rec_fusion_1_1', help='Unique experiment identifier.')
OPTIONS.add_argument('--device', type=str, default='cuda', help='Device to use')
OPTIONS.add_argument('--epochs', type=int, default=500, help='Number of train epochs.')
OPTIONS.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
OPTIONS.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
OPTIONS.add_argument('--batch_size', type=int, default=4, help='Batch size')
OPTIONS.add_argument('--logdir', type=str, default='logdir_Sample75', help='Directory to log data to.')

args = OPTIONS.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

write = SummaryWriter(log_dir=args.logdir)


class Engine(object):
    def __init__(self, cur_epoch=0, cur_iter=0):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.best_valid_epoch = cur_epoch
        self.train_loss = []
        self.train_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []
        self.best_valid = 1e10

    def train(self):
        loss_epoch = 0.
        num_batch = 0

        model.train()
        correct = 0
        total = 0
        acc_epoch = 0.
        # Train iter
        for data in tqdm(dataloader_train):
            # zero gradients
            for p in model.parameters():
                p.grad = None
            # create batch and move to GPU
            gaf0_in = data['GAF0s']
            gaf1_in = data['GAF1s']
            gaf2_in = data['GAF2s']
            gaf3_in = data['GAF3s']
            gaf4_in = data['GAF4s']
            gaf5_in = data['GAF5s']
            gaf6_in = data['GAF6s']
            gaf7_in = data['GAF7s']
            gaf8_in = data['GAF8s']
            gaf9_in = data['GAF9s']
            gaf10_in = data['GAF10s']
            gaf11_in = data['GAF11s']
            gaf12_in = data['GAF12s']
            gaf13_in = data['GAF13s']
            gaf14_in = data['GAF14s']
            gaf15_in = data['GAF15s']
            gaf16_in = data['GAF16s']
            gaf17_in = data['GAF17s']
            gaf18_in = data['GAF18s']
            gaf19_in = data['GAF19s']
            gaf20_in = data['GAF20s']
            gaf21_in = data['GAF21s']
            gaf22_in = data['GAF22s']
            gaf23_in = data['GAF23s']
            gaf24_in = data['GAF24s']
            gaf25_in = data['GAF25s']
            gaf26_in = data['GAF26s']
            gaf27_in = data['GAF27s']
            gaf28_in = data['GAF28s']
            gaf29_in = data['GAF29s']

            gaf0 = []
            gaf1 = []
            gaf2 = []
            gaf3 = []
            gaf4 = []
            gaf5 = []
            gaf6 = []
            gaf7 = []
            gaf8 = []
            gaf9 = []
            gaf10 = []
            gaf11 = []
            gaf12 = []
            gaf13 = []
            gaf14 = []
            gaf15 = []
            gaf16 = []
            gaf17 = []
            gaf18 = []
            gaf19 = []
            gaf20 = []
            gaf21 = []
            gaf22 = []
            gaf23 = []
            gaf24 = []
            gaf25 = []
            gaf26 = []
            gaf27 = []
            gaf28 = []
            gaf29 = []

            for i in range(config.seq_len):
                gaf0.append(gaf0_in[i].to(args.device, dtype=torch.float32))
                gaf1.append(gaf1_in[i].to(args.device, dtype=torch.float32))
                gaf2.append(gaf2_in[i].to(args.device, dtype=torch.float32))
                gaf3.append(gaf3_in[i].to(args.device, dtype=torch.float32))
                gaf4.append(gaf4_in[i].to(args.device, dtype=torch.float32))
                gaf5.append(gaf5_in[i].to(args.device, dtype=torch.float32))
                gaf6.append(gaf6_in[i].to(args.device, dtype=torch.float32))
                gaf7.append(gaf7_in[i].to(args.device, dtype=torch.float32))
                gaf8.append(gaf8_in[i].to(args.device, dtype=torch.float32))
                gaf9.append(gaf9_in[i].to(args.device, dtype=torch.float32))
                gaf10.append(gaf10_in[i].to(args.device, dtype=torch.float32))
                gaf11.append(gaf11_in[i].to(args.device, dtype=torch.float32))
                gaf12.append(gaf12_in[i].to(args.device, dtype=torch.float32))
                gaf13.append(gaf13_in[i].to(args.device, dtype=torch.float32))
                gaf14.append(gaf14_in[i].to(args.device, dtype=torch.float32))
                gaf15.append(gaf15_in[i].to(args.device, dtype=torch.float32))
                gaf16.append(gaf16_in[i].to(args.device, dtype=torch.float32))
                gaf17.append(gaf17_in[i].to(args.device, dtype=torch.float32))
                gaf18.append(gaf18_in[i].to(args.device, dtype=torch.float32))
                gaf19.append(gaf19_in[i].to(args.device, dtype=torch.float32))
                gaf20.append(gaf20_in[i].to(args.device, dtype=torch.float32))
                gaf21.append(gaf21_in[i].to(args.device, dtype=torch.float32))
                gaf22.append(gaf22_in[i].to(args.device, dtype=torch.float32))
                gaf23.append(gaf23_in[i].to(args.device, dtype=torch.float32))
                gaf24.append(gaf24_in[i].to(args.device, dtype=torch.float32))
                gaf25.append(gaf25_in[i].to(args.device, dtype=torch.float32))
                gaf26.append(gaf26_in[i].to(args.device, dtype=torch.float32))
                gaf27.append(gaf27_in[i].to(args.device, dtype=torch.float32))
                gaf28.append(gaf28_in[i].to(args.device, dtype=torch.float32))
                gaf29.append(gaf29_in[i].to(args.device, dtype=torch.float32))

            label = data['Label'][0].to(args.device, dtype=torch.float32)
            output = model(gaf0 + gaf1 + gaf2 + gaf3 + gaf4 + gaf5 + gaf6 + gaf7 + gaf8 + gaf9 +
                           gaf10 + gaf11 + gaf12 + gaf13 + gaf14 + gaf15 + gaf16 + gaf17 + gaf18 + gaf19 +
                           gaf20 + gaf21 + gaf22 + gaf23 + gaf24 + gaf25 + gaf26 + gaf27 + gaf28 + gaf29)

            _, pre = torch.max(output.data, dim=1)
            total += label.size(0)
            correct += (pre == label).sum()
            train_acc = 100 * correct / total
            acc_epoch += float(train_acc)

            loss = F.cross_entropy(output, label.long())
            loss.backward()

            loss_epoch += float(loss.item())

            num_batch += 1
            optimizer.step()

            write.add_scalar('train_loss', loss.item(), self.cur_iter)
            write.add_scalar('train_acc', train_acc.item(), self.cur_epoch)
            self.cur_iter += 1

        acc_epoch = acc_epoch / num_batch
        loss_epoch = loss_epoch / num_batch
        self.train_loss.append(loss_epoch)
        self.train_accuracy.append(acc_epoch)
        self.cur_epoch += 1

    def validation(self):
        model.eval()

        with torch.no_grad():
            num_batch = 0
            val_epoch = 0.
            total = 0
            correct = 0
            # Validation iter
            for batch_num, data in enumerate(tqdm(dataloader_valid), 0):

                # create batch and move to GPU

                gaf0_in = data['GAF0s']
                gaf1_in = data['GAF1s']
                gaf2_in = data['GAF2s']
                gaf3_in = data['GAF3s']
                gaf4_in = data['GAF4s']
                gaf5_in = data['GAF5s']
                gaf6_in = data['GAF6s']
                gaf7_in = data['GAF7s']
                gaf8_in = data['GAF8s']
                gaf9_in = data['GAF9s']
                gaf10_in = data['GAF10s']
                gaf11_in = data['GAF11s']
                gaf12_in = data['GAF12s']
                gaf13_in = data['GAF13s']
                gaf14_in = data['GAF14s']
                gaf15_in = data['GAF15s']
                gaf16_in = data['GAF16s']
                gaf17_in = data['GAF17s']
                gaf18_in = data['GAF18s']
                gaf19_in = data['GAF19s']
                gaf20_in = data['GAF20s']
                gaf21_in = data['GAF21s']
                gaf22_in = data['GAF22s']
                gaf23_in = data['GAF23s']
                gaf24_in = data['GAF24s']
                gaf25_in = data['GAF25s']
                gaf26_in = data['GAF26s']
                gaf27_in = data['GAF27s']
                gaf28_in = data['GAF28s']
                gaf29_in = data['GAF29s']

                gaf0 = []
                gaf1 = []
                gaf2 = []
                gaf3 = []
                gaf4 = []
                gaf5 = []
                gaf6 = []
                gaf7 = []
                gaf8 = []
                gaf9 = []
                gaf10 = []
                gaf11 = []
                gaf12 = []
                gaf13 = []
                gaf14 = []
                gaf15 = []
                gaf16 = []
                gaf17 = []
                gaf18 = []
                gaf19 = []
                gaf20 = []
                gaf21 = []
                gaf22 = []
                gaf23 = []
                gaf24 = []
                gaf25 = []
                gaf26 = []
                gaf27 = []
                gaf28 = []
                gaf29 = []
                for i in range(config.seq_len):
                    gaf0.append(gaf0_in[i].to(args.device, dtype=torch.float32))
                    gaf1.append(gaf1_in[i].to(args.device, dtype=torch.float32))
                    gaf2.append(gaf2_in[i].to(args.device, dtype=torch.float32))
                    gaf3.append(gaf3_in[i].to(args.device, dtype=torch.float32))
                    gaf4.append(gaf4_in[i].to(args.device, dtype=torch.float32))
                    gaf5.append(gaf5_in[i].to(args.device, dtype=torch.float32))
                    gaf6.append(gaf6_in[i].to(args.device, dtype=torch.float32))
                    gaf7.append(gaf7_in[i].to(args.device, dtype=torch.float32))
                    gaf8.append(gaf8_in[i].to(args.device, dtype=torch.float32))
                    gaf9.append(gaf9_in[i].to(args.device, dtype=torch.float32))
                    gaf10.append(gaf10_in[i].to(args.device, dtype=torch.float32))
                    gaf11.append(gaf11_in[i].to(args.device, dtype=torch.float32))
                    gaf12.append(gaf12_in[i].to(args.device, dtype=torch.float32))
                    gaf13.append(gaf13_in[i].to(args.device, dtype=torch.float32))
                    gaf14.append(gaf14_in[i].to(args.device, dtype=torch.float32))
                    gaf15.append(gaf15_in[i].to(args.device, dtype=torch.float32))
                    gaf16.append(gaf16_in[i].to(args.device, dtype=torch.float32))
                    gaf17.append(gaf17_in[i].to(args.device, dtype=torch.float32))
                    gaf18.append(gaf18_in[i].to(args.device, dtype=torch.float32))
                    gaf19.append(gaf19_in[i].to(args.device, dtype=torch.float32))
                    gaf20.append(gaf20_in[i].to(args.device, dtype=torch.float32))
                    gaf21.append(gaf21_in[i].to(args.device, dtype=torch.float32))
                    gaf22.append(gaf22_in[i].to(args.device, dtype=torch.float32))
                    gaf23.append(gaf23_in[i].to(args.device, dtype=torch.float32))
                    gaf24.append(gaf24_in[i].to(args.device, dtype=torch.float32))
                    gaf25.append(gaf25_in[i].to(args.device, dtype=torch.float32))
                    gaf26.append(gaf26_in[i].to(args.device, dtype=torch.float32))
                    gaf27.append(gaf27_in[i].to(args.device, dtype=torch.float32))
                    gaf28.append(gaf28_in[i].to(args.device, dtype=torch.float32))
                    gaf29.append(gaf29_in[i].to(args.device, dtype=torch.float32))
                label = data['Label'][0].to(args.device, dtype=torch.float32)
                output = model(gaf0 + gaf1 + gaf2 + gaf3 + gaf4 + gaf5 + gaf6 + gaf7 + gaf8 + gaf9 +
                               gaf10 + gaf11 + gaf12 + gaf13 + gaf14 + gaf15 + gaf16 + gaf17 + gaf18 + gaf19 +
                               gaf20 + gaf21 + gaf22 + gaf23 + gaf24 + gaf25 + gaf26 + gaf27 + gaf28 + gaf29)

                _, pre = torch.max(output.data, dim=1)
                total += label.size(0)
                correct += (pre == label).sum()

                val_epoch += float(F.cross_entropy(output, label.long()))

                num_batch += 1
            val_loss = val_epoch / float(num_batch)
            val_acc = 100 * correct / total
            tqdm.write(
                f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:  ' + f'Valid_loss: {val_loss:3.3f}, Valid_acc: {val_acc:3.3f}')

            write.add_scalar('valid_loss', val_loss, self.cur_epoch)
            write.add_scalar('valid_acc', val_acc.item(), self.cur_epoch)
            self.valid_loss.append(val_loss)
            self.valid_accuracy.append(val_acc.item())

    def save(self):
        save_best = False
        if self.valid_loss[-1] <= self.best_valid:
            self.best_valid = self.valid_loss[-1]
            self.best_valid_epoch = self.cur_epoch
            save_best = True

        # Creat a dic of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'best_valid_loss': self.best_valid,
            'best_valid_epoch': self.best_valid_epoch,
            'train_loss': self.train_loss,
            'train_acc': self.train_accuracy,
            'valid_loss': self.valid_loss,
            'valid_acc': self.valid_accuracy
        }

        # Save ckpt for every epoch
        # torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth' % self.cur_epoch))

        # Save the recent model/optimizer states
        # torch.save(model.state_dict(), os.path.join(args.logdir, 'model_pth'))
        # torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('======Save recent model======>')

        if save_best:
            torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write('=====Overwrite best model=====>')


if __name__ == '__main__':
    # config
    config = GlobalConfig

    # Data
    Plk_data = pickle.load(open(config.root10, 'rb'))
    train_dic, valid_dic, test_dic = Plk_data[config.train], Plk_data[config.valid], Plk_data[config.test]
    train_set = RAIL_Data(train_dic, config=config)
    valid_dic = RAIL_Data(valid_dic, config=config)

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dataloader_valid = DataLoader(valid_dic, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = Decoder(config, args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # sceduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    trainer = Engine()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    # Creat logdir
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
        print('Creat dir: ', args.logdir)

    elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
        print('Loading checkpoint from' + args.logdir)
        with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
            log_table = json.load(f)

        # Load Variables
        trainer.cur_epoch = log_table['epoch']
        if 'iter' in log_table: trainer.cur_iter = log_table['iter']
        trainer.best_valid = log_table['best_valid']
        trainer.train_loss = log_table['train_loss']
        trainer.train_accuracy = log_table['train_acc']
        trainer.valid_loss = log_table['valid_loss']
        trainer.valid_accuracy = log_table['valid_acc']

        # Load checkpoint
        # model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
        # optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

    # Log args
    with open(os.path.join(args.logdir, 'arg.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for epoch in range(trainer.cur_epoch, args.epochs):
        trainer.train()
        if epoch % args.val_every == 0:
            trainer.validation()
            trainer.save()

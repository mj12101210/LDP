import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from model_channel_semi_aug import Model_channel_semi_aug
from mini_imagenet import Mini_ImageNet
from samplers import Categories_sampler
import few_shot as fs
import utils
from utils import get_log_file, print_and_log


def main():
    print('>> CUDA device is avaiable, input the GPU index:')
    gpu_index = input()
    args.gpu_index = gpu_index
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    if torch.cuda.is_available() is False:
        print('NO CUDA')
        return

    log_file = get_log_file(args.save_path, 'train')
    print_and_log(log_file, "Options: %s" % args)
    print_and_log(log_file, "Save path: %s\n" % args.save_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model = Model_channel_semi_aug(args, encoder=args.pretrained_resnet_path).to(device)

    optimizer = torch.optim.SGD(train_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print_and_log(log_file, 'Optimizer:\n%s\n' % optimizer)

    train_set = Mini_ImageNet('train')
    # 此处加入了u_unlabeled
    train_sampler = Categories_sampler(train_set.label, args.train_batch_num, args.n_way, args.k_shot + args.k_query, args.u_unlabeled)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    print_and_log(log_file, 'train dataset: {} (x{}), {}'.format(train_set[0][0].shape, len(train_set), len(train_set.wnids)))

    val_set = Mini_ImageNet('val')
    val_sampler = Categories_sampler(val_set.label, args.val_batch_num, args.n_way, args.k_shot + args.k_query, args.u_unlabeled)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    print_and_log(log_file, 'val dataset: {} (x{}), {}'.format(val_set[0][0].shape, len(val_set), len(val_set.wnids)))
    
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    max_va = 0.

    for epoch in range(1, args.epoch + 1):

        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'va']
        aves = {k: utils.Averager() for k in aves_keys}

        val_acc_list = []

        # train
        train_model.train()

        for task_images, _ in tqdm(train_loader, desc='train', leave=False):
            
            # x_support: torch.Size([5, 1, 3, 84, 84])
            # x_query:   torch.Size([75, 3, 84, 84])
            # x_unlabel: torch.Size([75, 3, 84, 84])
            x_support, x_query, x_unlabel = fs.split_shot_query_unlabel(task_images.cuda(), args.n_way, args.k_shot, args.k_query, args.u_unlabeled)
            # x_support, x_query, x_unlabel = fs.split_shot_query_unlabel(task_images, args.n_way, args.k_shot, args.k_query, args.u_unlabeled)
            y_query = fs.make_nk_label(args.n_way, args.k_query).to(device)

            # torch.Size([75, 5])
            logits = train_model(x_support, x_query, x_unlabel, args.T)
            loss = F.cross_entropy(logits, y_query)
            acc = utils.compute_acc(logits, y_query)

            # 清空过往梯度
            optimizer.zero_grad()
            # 反向传播，计算当前梯度
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            # 根据梯度更新网络参数
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)


        train_model.eval()

        for (task_images, _) in tqdm(val_loader, desc='val', leave=False):
            x_support, x_query, x_unlabel = fs.split_shot_query_unlabel(task_images.to(device), args.n_way, args.k_shot, args.k_query, args.u_unlabeled)
            y_query = fs.make_nk_label(args.n_way, args.k_query).to(device)

            with torch.no_grad():
                logits = train_model(x_support, x_query, x_unlabel, args.T)
                acc = utils.compute_acc(logits, y_query)

            aves['va'].add(acc)
            val_acc_list.append(acc)

        val_confidence = (1.96 * torch.Tensor(val_acc_list).std()) / np.sqrt(len(val_acc_list))

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * args.epoch)

        log_str = 'epoch {}: train: loss: {:.4f} | acc: {:.4f}; val: acc: {:.4f}+/-{:.4f}'.format(epoch, aves['tl'].item(), aves['ta'].item(), aves['va'].item(), val_confidence.item())
        log_str += '\t{} {}/{}'.format(t_epoch, t_used, t_estimate)
        print_and_log(log_file, log_str)

        torch.save(train_model.state_dict(), os.path.join(args.save_path, 'epoch-{}.pt'.format(epoch)))

        if aves['va'].item() > max_va:
            max_va = aves['va'].item()
            torch.save(train_model.state_dict(), os.path.join(args.save_path, 'max-va.pt'))

    torch.save(train_model.state_dict(), os.path.join(args.save_path, 'epoch-last.pt'))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", "-s", default='save', help="Path to save.")
    parser.add_argument("--pretrained_resnet_path", default="pre_trained_model.pt",
                        help="Path to pretrained feature extractor model(miniimagenet).")
    parser.add_argument("--epoch", type=int, default=40, help="Number of epoch.")
    parser.add_argument("--train_batch_num", type=int, default=500, help="Number of batch.")
    parser.add_argument("--val_batch_num", type=int, default=500, help="Number of batch.")
    parser.add_argument("--n_way", type=int, default=5, help="Way of single dataset task.")
    parser.add_argument("--k_shot", type=int, default=5, help="Shots per class for context of single dataset task.")
    parser.add_argument("--k_query", type=int, default=15, help="Shots per class for target  of single dataset task.")

    # 引入无标签数据
    parser.add_argument("--u_unlabeled", type=int, default=15, help="unlabeled sample per class of a single task.")

    parser.add_argument('--T', default=0.5, type=float)

    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", "-wd", type=float, default=5.e-4, help="Weight decay.")

    args = parser.parse_args()



    main()
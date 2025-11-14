import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    log_file = get_log_file(args.save_path, 'test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_and_log(log_file, "\nOptions: %s\n" % args)

    test_model = Model_channel_semi_aug(args)
    test_model.load_state_dict(torch.load(args.test_model_path))
    test_model = test_model.to(device)
    
    test_set = Mini_ImageNet('test')
    test_sampler = Categories_sampler(test_set.label, args.test_batch_num, args.n_way, args.k_shot + args.k_query, args.u_unlabeled)
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=8, pin_memory=True)
    print_and_log(log_file, 'test dataset: {} (x{}), {}'.format(test_set[0][0].shape, len(test_set), len(test_set.wnids)))

    test_model.eval()

    test_acc = utils.Averager()
    test_acc_list = []
    
    for (task_images, _) in tqdm(test_loader, desc='test', leave=False):

        x_support, x_query, x_unlabel = fs.split_shot_query_unlabel(task_images.to(device), args.n_way, args.k_shot,
                                                                 args.k_query, args.u_unlabeled)
        y_query = fs.make_nk_label(args.n_way, args.k_query).to(device)
        
        with torch.no_grad():
            # logits = test_model(x_support, x_query)
            logits = test_model(x_support, x_query, x_unlabel, args.T)
            acc = utils.compute_acc(logits, y_query)
        
        test_acc.add(acc)
        test_acc_list.append(acc)

    test_confidence = (1.96 * torch.Tensor(test_acc_list).std()) / np.sqrt(len(test_acc_list))

    print_and_log(log_file, 'test acc: {:.4f}+/-{:.4f}'.format(test_acc.item(), test_confidence.item()))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", "-s", default='save', help="Path to save.")
    parser.add_argument("--test_model_path", "-m", default='save/max-va.pt',
                        help="Path to model to load and test.")
    parser.add_argument("--test_batch_num", type=int, default=10000, help="Number of batch.")
    parser.add_argument("--n_way", type=int, default=5, help="Way of single dataset task.")
    parser.add_argument("--k_shot", type=int, default=1, help="Shots per class for context of single dataset task.")
    parser.add_argument("--k_query", type=int, default=15, help="Shots per class for target  of single dataset task.")

    # 引入无标签数据
    parser.add_argument("--u_unlabeled", type=int, default=15, help="unlabeled sample per class of a single task.")

    parser.add_argument('--T', default=0.5, type=float)

    args = parser.parse_args()
    main()
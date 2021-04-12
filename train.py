import math
import os
import random
import pickle
import argparse
from collections import deque
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import matplotlib.pyplot as plt

def get_time_dif(start_time):
    """get the running time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class GetTriplePair(IterableDataset):
    def __init__(self, item_size, user_list, pair, shuffle, num_epochs):
        self.item_size = item_size
        self.user_list = user_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        self.start_list_index = None
        self.num_workers = 1
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]

                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.item_size)
        while j in self.user_list[u]:
            j = np.random.randint(self.item_size)
        return u, i, j


class MF(nn.Module):
    def __init__(self, user_size, item_size, dim, reg, reg_adv, eps):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))  # User embedding
        self.H = nn.Parameter(torch.empty(item_size, dim))  # Item embedding
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.reg = reg
        self.user_size = user_size
        self.item_size = item_size
        self.dim = dim
        self.reg_adv = reg_adv
        self.eps = eps
        self.update_u = None
        self.update_i = None
        self.update_j = None

    def forward(self, u, i, j, epoch):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
            epoch

        Returns:
            torch.FloatTensor
        """

        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        u.retain_grad()
        u_clone = u.data.clone()
        i.retain_grad()
        i_clone = i.data.clone()
        j.retain_grad()
        j_clone = j.data.clone()
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)

        x_uij =torch.clamp(x_ui - x_uj,min=-80.0,max=1e8)
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.reg * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        loss = -log_prob + regularization

        if epoch not in range(args.epochs, args.adv_epoch + args.epochs):
            """Normal training"""
            loss.backward()
            return loss

        else:
            """Adversarial training:
                    1.Backward to get grads
                    2.Construct adversarial perturbation
                    3.Add adversarial perturbation to embeddings
                    4.Calculate APR loss
            """
            # Backward to get grads
            loss.backward(retain_graph=True)
            grad_u = u.grad
            grad_i = i.grad
            grad_j = j.grad

            # Construct adversarial perturbation
            if grad_u is not None:
                delta_u = nn.functional.normalize(grad_u, p=2, dim=1, eps=self.eps)
            else:
                delta_u = torch.rand(u.size())
            if grad_i is not None:
                delta_i = nn.functional.normalize(grad_i, p=2, dim=1, eps=self.eps)
            else:
                delta_i = torch.rand(i.size())
            if grad_j is not None:
                delta_j = nn.functional.normalize(grad_j, p=2, dim=1, eps=self.eps)
            else:
                delta_j = torch.rand(j.size())

            # Add adversarial perturbation to embeddings
            x_ui_adv = torch.mul(u + delta_u, i + delta_i).sum(dim=1)
            x_uj_adv = torch.mul(u + delta_u, j + delta_j).sum(dim=1)
            x_uij_adv = torch.clamp(x_ui_adv - x_uj_adv,min=-80.0,max=1e8)

            # Calculate APR loss
            log_prob = F.logsigmoid(x_uij_adv).sum()
            adv_loss = self.reg_adv *(-log_prob) + loss
            adv_loss.backward()

            # Restore embedding data (and update)
            u.data = u_clone
            i.data = i_clone
            j.data = j_clone

            return adv_loss


def evaluate_k(user_emb, item_emb, train_user_list, test_user_list, klist, batch=512):
    """Compute HR and NDCG at k.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) HR and NDCG at k
    """

    # Calculate max k value
    max_k = max(klist)
    result = None
    for i in range(0, user_emb.shape[0], batch):

        # Construct mask for each batch
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i + j])), value=torch.tensor(0.0))

        # Get current result
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))

        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

    result = result.cpu()

    # Sort indice and get HR_NDCG_topk
    HRs, NDCGs = [], []
    for k in klist:
        ndcg, hr = 0, 0
        for i in range(user_emb.shape[0]):
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            hr += val / max([len(test), 1])
            pred = list(pred)
            x = int(test_user_list[i][0])
            if pred.count(x) != 0:
                position = pred.index(x)
                ndcg += math.log(2) / math.log(position + 2) if position < k else 0
            else:
                ndcg += 0
        NDCGs.append(ndcg / user_emb.shape[0])
        HRs.append(hr / user_emb.shape[0])
        NDCGs.append(ndcg / user_emb.shape[0])
    return HRs, NDCGs


def main(args):
    # Initialize seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']

    # Create dataset, model, optimizer
    dataset = GetTriplePair(item_size, train_user_list, train_pair, True, args.epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = MF(user_size, item_size, args.dim, args.reg, args.reg_adv, args.eps)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    start_time = time.time()
    eval_best_loss = float('inf')
    optimizer.zero_grad()
    epoch = 0
    HR_history = []
    NDCG_history = []
    for u, i, j in loader:
        if epoch in range(args.epochs + args.adv_epoch):
            loss = model(u, i, j, epoch)
            optimizer.step()
            HR_list, NDCG_list = evaluate_k(model.W.detach(),
                                                               model.H.detach(),
                                                               train_user_list,
                                                               test_user_list,
                                                               klist=[50, 100])
            if epoch % args.verbose == (args.verbose - 1):
                if epoch in range(args.epochs):
                    print('BPR-MF Epoch [{}/{}]'.format(epoch + 1, args.epochs + args.adv_epoch))
                if epoch in range(args.epochs, args.adv_epoch + args.epochs):
                    print('AMF Epoch [{}/{}]'.format(epoch + 1, args.epochs + args.adv_epoch))
                print('loss: %.4f' % loss)
                print('HR@50: %.4f, HR@100: %.4f, NDCG@50: %.4f, NDCG@100: %.4f' % (
                    HR_list[0], HR_list[1], NDCG_list[0], NDCG_list[1]))
            HR_history.append(HR_list[1])
            NDCG_history.append(NDCG_list[1])
            if epoch % 100 == 0:
                if loss < eval_best_loss:
                    eval_best_loss = loss
                    dirname = os.path.dirname(os.path.abspath(args.model))
                    os.makedirs(dirname, exist_ok=True)
                    torch.save(model.state_dict(), args.model)
                    time_dif = get_time_dif(start_time)
                    print("time", time_dif)
            epoch += 1
        else:
            break

    fig_HR = plt.figure(edgecolor='blue')
    ax1 = fig_HR.add_subplot(111)
    plt.ylabel('HR@100')
    plt.xlabel('Epoch')
    plt.title('Yelp')
    ax1.plot(range(len(HR_history)), HR_history, c=np.array([255, 71, 90]) / 255.)
    plt.show()
    fig_P = plt.figure(edgecolor='blue')
    ax1 = fig_P.add_subplot(111)
    plt.ylabel('NDCG@100')
    plt.xlabel('Epoch')
    plt.title('Yelp')
    ax1.plot(range(len(NDCG_history)), NDCG_history, c=np.array([255, 71, 90]) / 255.)
    plt.show()


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join('preprocessed', 'ml-1m.pickle'),
                        help="File path for data")
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=64,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default= 0.00025,
                        help="Learning rate")
    parser.add_argument('--reg',
                        type=float,
                        default=0,
                        help="Regularization for user and item embeddings.")
    # Training
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=2000,
                        help="Batch size in one iteration")
    parser.add_argument('--verbose',
                        type=int,
                        default=20,
                        help="Evaluate per X epochs")
    parser.add_argument('--eval_every',
                        type=int,
                        default=20,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--model',
                        type=str,
                        default=os.path.join('output', 'bpr.pt'),
                        help="File path for model")
    parser.add_argument('--reg_adv', type=float, default=1,
                        help='Regularization for adversarial loss')
    parser.add_argument('--adv_epoch', type=int, default=400,
                        help='Add APR in epoch X, when adv_epoch is 0, it\'s equivalent to pure AMF.\n '
                             'And when adv_epoch is larger than epochs, it\'s equivalent to pure MF model. ')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon for adversarial weights.')
    args = parser.parse_args()
    main(args)


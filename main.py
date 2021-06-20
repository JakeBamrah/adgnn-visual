"""
This repository is an extension of prior work:

    - Auto-Metric Graph Neural Network Based on a Meta-learning Strategy
    - Few shot GNN
    - Population GCN

focusing on Alzheimer's disease diagnosis.

This repository is for research purposes only and is not permitted for
commercial use. Please see repository for futher details.
"""

import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import pickle
import random
import textwrap
from pathlib import Path
from utils import io_utils

import visualize
import gnn_models as models
from predict import Predict
from data import DataGenerator


RESULT_PATH = Path("results/")
DATA_PATH = Path("data/")

# tensorboard summary writer
WRITER_PATH = 'runs/patient_prediction_scratch/'
tb = SummaryWriter(WRITER_PATH)


parser = argparse.ArgumentParser(description='AMGNN')
parser.add_argument('--metric_network', type=str, default='gnn', metavar='N',
                    help='gnn')
parser.add_argument('--dataset', type=str, default='AD', metavar='N',
                    help='AD')
parser.add_argument('--test_N_way', type=int, default=3, metavar='N')
parser.add_argument('--train_N_way', type=int, default=3, metavar='N')
parser.add_argument('--test_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--train_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--feature_num', type=int, default=31, metavar='N',
                    help='feature number of one sample')
parser.add_argument('--clinical_feature_num', type=int, default=4, metavar='N',
                    help='clinical feature number of one sample')
parser.add_argument('--w_feature_num', type=int, default=27, metavar='N',
                    help='feature number for w computation')
parser.add_argument('--w_feature_list', type=int, default=5, metavar='N',
                   help='feature list for w computation')
# 0-4,1-9，2-5,3-13,4-9，5-14,6-18
# 0-4,1-9，2-10,3-13,4-14，5-19,6-23
parser.add_argument('--iterations', type=int, default=300, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                    help='Decreasing the learning rate every x iterations')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_train', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_interval', type=int, default=200, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--random_seed', type=int, default=2019, metavar='N')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('GPU:', args.cuda)
random_seed = args.random_seed


def setup_seed(seed=random_seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizers, lr, iter, writer=None):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    if writer:
        writer.add_scalar("Learning rate", new_lr, iter)





def train_batch(model, data):
    """Train a model on selected data sample"""
    [amgnn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi] = data

    # NOTE: separate features per batch
    # slice the first four features which are our risk factors
    z_clinical = batch_x[:, 0, 0, 0:args.clinical_feature_num]
    zi_s_clinical = [batch_xi[:,0,0,0:args.clinical_feature_num] for batch_xi in batches_xi]

    # slice the remaining 27 features after our clinical / risk factors
    z_mri_feature = batch_x[:, :, :, args.clinical_feature_num:]
    zi_s_mri_feature = [batch_xi[:, :, :, args.clinical_feature_num:] for batch_xi in batches_xi]

    adj = amgnn.compute_adj(z_clinical, zi_s_clinical)

    inputs = [z_clinical, z_mri_feature, zi_s_clinical, zi_s_mri_feature, labels_yi, oracles_yi, adj]
    out_metric, out_logits = amgnn(*inputs)
    logsoft_prob = softmax_module.forward(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()

    return loss


def test_one_shot(args, fold, test_root, model, test_samples=50, partition='test', io_path= 'run.log', write_model_graph=False):
    io = io_utils.IOStream(io_path)

    io.cprint('\n**** TESTING BEGIN ***' )
    root = test_root
    data_loader = DataGenerator(root, keys=['CN','MCI','AD'])
    [amgnn, softmax_module] = model
    amgnn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples / args.batch_size_test) # 320 / 64

    for i in range(iterations):
        data = data_loader.get_task_batch(
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                cuda=args.cuda
        )

        Y, y_pred, labels_x_cpu = predict.predict_nodes_using_one_shot(data)
        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1

    labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    loss_test_f = float(loss_test)
    del loss_test

    message = textwrap.dedent("""
    ***ITERATION FINISHED***
    Loss: {}
    Correct: {}
    Total: {}
    Accuracy: {:.3f}%
    """.format(loss_test_f, correct, total, 100.0 * (correct / total)))
    io.cprint(message)

    amgnn.train()
    accuracy = 100 * (correct / total)

    return correct, accuracy, loss_test_f


if __name__ =='__main__':
    now = datetime.datetime.now()
    now_format = now.strftime('%Y-%m-%d-%H-%M')
    save_path = RESULT_PATH / now_format

    print(now_format)

    if save_path not in os.listdir(RESULT_PATH):
        os.makedirs(save_path)
    io = io_utils.IOStream(RESULT_PATH / 'run.log')
    print('The result will be saved in :', save_path)
    setup_seed(args.random_seed)

    amgnn = models.create_models(args, cnn_dim1=2)

    # initialise softmax and prediction modules
    softmax_module = models.SoftmaxModule()
    predict = Predict(amgnn, softmax_module, args)
    io.cprint(str(amgnn))


    # NOTE: CNN dimension where one CNN is used for learning the edge weight from the absolute difference
    # between each feature of the feature nodes - see notes 1b.
    io.cprint(str(amgnn))

    if args.cuda:
        amgnn.cuda()

    weight_decay = 0

    opt_amgnn = optim.Adam(amgnn.parameters(), lr=args.lr, weight_decay=weight_decay)
    amgnn.train()
    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    test_acc = 0

    for batch_idx in range(args.iterations):

        root = DATA_PATH / 'AD_3_CLASS_TRAIN.pkl'
        data_loader = DataGenerator(root, keys=['CN', 'MCI','AD'])
        data = data_loader.get_task_batch(
                batch_size=args.batch_size_train,
                n_way=args.train_N_way,
                num_shots=args.train_N_shots,
                cuda=args.cuda
        )
        [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi] = data

        opt_amgnn.zero_grad()

        # train model
        loss_d_metric = train_batch(model=[amgnn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi])
        opt_amgnn.step()

        adjust_learning_rate(optimizers=[opt_amgnn], lr=args.lr, iter=batch_idx, writer=tb)

        # test result output
        counter += 1
        total_loss += loss_d_metric.item()
        if batch_idx % args.log_interval == 0:
            display_str = 'Train Iter: {}'.format(batch_idx)
            display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss / counter)
            io.cprint(display_str)
            counter = 0
            total_loss = 0


        # test trained model performance
        if (batch_idx + 1) % args.log_interval == 0:

            test_samples = 320
            test_root = DATA_PATH / 'AD_3_CLASS_TEST.pkl'
            test_correct, test_acc_aux, test_loss_ = test_one_shot(
                            args, 0, test_root, model=[amgnn, softmax_module],
                            test_samples=test_samples, partition='test',
                            io_path=save_path / 'run.log'
                        )

            # record testing metrics for batch
            tb.add_scalar("Loss", test_loss_, batch_idx)
            tb.add_scalar("Correct", test_correct, batch_idx)
            tb.add_scalar("Accuracy", test_acc_aux, batch_idx)

            tb = visualize.record_amgnn_bias_metrics(amgnn, tb)


            amgnn.train()

            if test_acc_aux is not None and test_acc_aux >= test_acc:
                test_acc = test_acc_aux
                # val_acc = val_acc_aux
                torch.save(amgnn, save_path / 'amgnn_best_model.pkl')
            if args.dataset == 'AD':
                io.cprint("Best test accuracy {:.4f} \n".format(test_acc))

    tb.close()

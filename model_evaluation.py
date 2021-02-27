import argparse,torch
import os , hashlib
import data,statistics
import random
import math
import model_ablation
import numpy as np
import copy
import torch.nn as nn
import matplotlib.mlab as mlab
from scipy.stats import norm

import matplotlib.pyplot as plt
from collections import defaultdict

from utils import batchify, get_batch, repackage_hidden
from splitcross import SplitCrossEntropyLoss

from pathlib import Path


path = Path(__file__).parent.absolute()

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--cuda', action='store_true',default=False,
                    help='use CUDA')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--model_name', type=str,
                    help='path to model in *.pt format. e.g. PTB-mts.pt ')
parser.add_argument('--baseline_model', type=str,
                    help='path to baseline model in *.pt format. e.g. PTB.pt ')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--model_performance', action='store_true',default=False,
                    help='Calculate model performance on the test set')
parser.add_argument('--significance_testing', action='store_true',default=False,
                    help='Performance significance testing on baseline and MTS model')
parser.add_argument('--unit_ablation', action='store_true',default=False,
                    help='Evaluate model performance with unit ablation for layer 2')

args = parser.parse_args()

criterion = SplitCrossEntropyLoss(400, splits=[], verbose=False)
#entropy_calc = Entropy_calculation(400, splits=[], verbose=False)

seed = 141
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def model_load(fn):
    #global model
    with open(fn, 'rb') as f:
        model, criterion_m, optim = torch.load(f) #, map_location=torch.device('cpu')) #,
    return model

class Vocabulary():
    def __init__(self,data_path=None):

        fn = 'corpus.{}.data'.format(hashlib.md5(data_path.encode()).hexdigest())

        if os.path.exists(fn):
            print('Loading cached dataset...')
            self.corpus = torch.load(fn)
        else:
            print('Producing dataset...')
            self.corpus = data.Corpus(args.data)

        self.UNK_ind = self.corpus.dictionary.word2idx['<unk>']
        self.get_high_low_freq_idx()
        self.ntokens = len(self.corpus.dictionary)


    def load_data(self):
        test_tensor = self.corpus.test
        test_batch_size = 1
        test_data = batchify(test_tensor, test_batch_size, args)

        return test_data, test_batch_size

    def get_word2idx(self):
        return self.corpus.dictionary.word2idx

    def get_high_low_freq_idx(self):
        vocab_dict = self.corpus.dictionary.counter #counts with token_ids as key
        self.keys_above_10K, self.keys_in1K_10K, self.keys_in100_1K, self.keys_in100 = set(), set(), set(), set()

        for token_id in vocab_dict.keys():
            if vocab_dict[token_id] >= 10000:
                self.keys_above_10K.add(token_id)

            if vocab_dict[token_id] < 10000 and vocab_dict[token_id] >= 1000:
                self.keys_in1K_10K.add(token_id)

            if vocab_dict[token_id] < 1000 and vocab_dict[token_id] >= 100:
                self.keys_in100_1K.add(token_id)

            if vocab_dict[token_id] < 100:
                self.keys_in100.add(token_id)

        return

class Evaluate():
    def __init__(self, data_source, batch_size=10):
        self.data_source = data_source
        self.batch_size = batch_size

    def test_loss(self,model, args):

        model.eval()

        total_loss = 0
        hidden = model.init_hidden(self.batch_size)
        for i in range(0, self.data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(self.data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss.item() / len(self.data_source)

    def entropy_for_target_vec(self, model, args):
        model.eval()
        target_vec, entropy_vec = [], []

        hidden = model.init_hidden(self.batch_size)

        for i in range(0, self.data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(self.data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)

            x = len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets, return_avg=False).data

            target_vec.extend(targets.cpu().numpy().tolist())
            entropy_vec.extend(x.squeeze(1).cpu().numpy().tolist())

            hidden = repackage_hidden(hidden)

        return entropy_vec, len(self.data_source) - 1, target_vec

    def model_ppl_unit_ablation(self,model, args, partial_output=False):

        model.eval()
        target_vec, entropy_vec = [], []

        hidden = model.init_hidden(self.batch_size)

        for i in range(0, self.data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(self.data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden, partial_output)

            x = len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets, return_avg=False).data

            target_vec.extend(targets.cpu().numpy().tolist())
            entropy_vec.extend(x.squeeze(1).cpu().numpy().tolist())

            hidden = repackage_hidden(hidden)

        return entropy_vec, len(self.data_source) - 1, target_vec

def generate_bootstrapped_matrix(tokens_in_test_set):
    start_ind_arr = list(range(0, tokens_in_test_set, 100))[:-1]
    Boot_matrix = [random.choices(start_ind_arr, k=len(start_ind_arr)) for _ in range(10000)]
    Boot_matrix.append(start_ind_arr)

    return Boot_matrix

def plot_unit_ablation(bin1, bin2, bin3, bin4):
    import scipy
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.4))
    xticks = np.arange(0, len(bin1))
    timescale = ((scipy.stats.invgamma.isf(np.linspace(0, 1, 1151), a=0.56, scale=1)[1:]));

    # xticklabels = np.array(["{:.0e}".format(x) for x in timescale[50*xticks]])
    xticklabels = np.array(['{:2.1f}'.format(x) for x in timescale[50 * xticks]])
    xticklabels[0] = '360K'
    from matplotlib import cm

    colors = [cm.jet(x) for x in [0.71, 0.8, 0.9, 0.99]]

    ax.plot(bin1, '.-', color=colors[0], label='>10K')
    ax.plot(bin2, '.-', color=colors[1], label='1K-10K')
    ax.plot(bin3, '.-', color=colors[2], label='100-1K')
    ax.plot(bin4, '.-', color=colors[3], label='<100')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim([0.975, 1.08])
    ax.set_xlabel('Average timescale of ablated units (timesteps)')

    # Location of ablated chunk of 50 units from left(high timescale) to right(low timescale)
    ax.set_ylabel('Ratio of ppl with and w/o unit ablation')
    # ax.set_title('Change in ppx across different frequency bins vs. location of ablated chunk')
    ax.legend(fontsize=16, frameon=False)
    # ax.set_axis_bgcolor('white')
    # ax.grid(color='0.85')
    ax.axhline(y=1, color='gray', linestyle='--')
    for item in ([ax.title, ax.xaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    ax.yaxis.label.set_fontsize(17)
    plt.setp(ax.get_xticklabels(), fontsize=11, rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig('frequency_bins_' + data_type + '.eps', format='eps')
    fig.savefig('frequency_bins_' + data_type + '.png')  # frequency_bins_wiki

    return

def bootstrap_evaluation(entropy_loss, l):
    # Create an array of 10000x824
    start_ind_arr = list(range(0, l, 100))[:-1]
    Boot_matrix = generate_bootstrapped_matrix(l)
    h = [math.exp(sum([sum(entropy_loss[s:100 + s]) for s in row]) / (100 * (len(start_ind_arr)))) for row in
         Boot_matrix]
    return round(statistics.mean(h), 2),  round(statistics.stdev(h), 2)

def high_low_freq_bin_evaluation(entropy_loss, target_vec):

    above_10K_loss, in1K_10K_loss, in100_1K_loss, in100_loss = 0, 0, 0, 0
    above_10K_count, in1K_10K_count, in100_1K_count, in100_count = 0, 0, 0, 0

    for target_ind in range(len(target_vec)):
        if target_vec[target_ind] in vocab.keys_above_10K:
            above_10K_loss += entropy_loss[target_ind]
            above_10K_count += 1
        elif target_vec[target_ind] in vocab.keys_in1K_10K:
            in1K_10K_loss += entropy_loss[target_ind]
            in1K_10K_count += 1
        elif target_vec[target_ind] in vocab.keys_in100_1K:
            in100_1K_loss += entropy_loss[target_ind]
            in100_1K_count += 1
        elif target_vec[target_ind] in vocab.keys_in100:
            in100_loss += entropy_loss[target_ind]
            in100_count += 1

    a = round(math.exp((above_10K_loss) / above_10K_count), 2)
    b = round(math.exp((in1K_10K_loss) / in1K_10K_count), 2)
    c = round(math.exp((in100_1K_loss) / in100_1K_count), 2)
    d = round(math.exp((in100_loss) / in100_count), 2)

    return d,c,b,a

def test_model_performance(model_name, test_data , test_batch_size):

    model = model_load(model_name)
    eval = Evaluate(test_data, test_batch_size)
    test_loss = eval.test_loss(model, args)

    ppl, bpc = round(math.exp(test_loss),2), round(test_loss / math.log(2),2)
    print("Word perplexity:",ppl, "BPC:",bpc)

    entropy_loss, l, target_vec = eval.entropy_for_target_vec(model, args)

    a,b,c,d, = high_low_freq_bin_evaluation(entropy_loss, target_vec)
    print("Model ppl: from low to high frequency bin:",a, b, c, d)

    m , v = bootstrap_evaluation(entropy_loss, l)
    print("Bootstrap test performance - Mean ppl: ",m,"Standard deviation of ppl across 10000 samples: ",v)

    return

def statistics_significance(entropy_baseline, entropy_mts, target_vec, Boot_matrix):

    ppl_mts_bin = [[] for _ in range(5)]
    ppl_baseline_bin = [[] for _ in range(5)]

    target_vec_bin = []

    for target_ind in range(len(target_vec)):
        if target_vec[target_ind] in vocab.keys_above_10K:
            target_vec_bin.append(1)

        elif target_vec[target_ind] in vocab.keys_in1K_10K:
            target_vec_bin.append(2)

        elif target_vec[target_ind] in vocab.keys_in100_1K:
            target_vec_bin.append(3)

        elif target_vec[target_ind] in vocab.keys_in100:
            target_vec_bin.append(4)

    target_vec_bin = np.array(target_vec_bin)

    for row in Boot_matrix:
        target_vec_bin_indices = []
        target_vec_sample = np.hstack(target_vec_bin[s:s + 100] for s in row)

        for j in range(1, 5):
            a = np.where(target_vec_sample == j)[0]
            target_vec_bin_indices.append(a)

        entropy_vec_baseline_eg = np.hstack(entropy_baseline[s:s + 100] for s in row)
        entropy_vec_mts_eg = np.hstack(entropy_mts[s:s + 100] for s in row)

        for i in range(1, 5):
            ppl_baseline_bin[i - 1].append(math.exp(np.mean(entropy_vec_baseline_eg[target_vec_bin_indices[i - 1]])))
            ppl_mts_bin[i - 1].append(math.exp(np.mean(entropy_vec_mts_eg[target_vec_bin_indices[i - 1]])))

        ppl_baseline_bin[i].append(math.exp(np.mean(entropy_vec_baseline_eg)))
        ppl_mts_bin[i].append(math.exp(np.mean(entropy_vec_mts_eg)))

    assert (round(math.exp(np.mean(entropy_vec_mts_eg)),2) == 57.58 and round(math.exp(np.mean(entropy_vec_baseline_eg)),2) == 58.98)
    print('Mean perplexity difference for')
    for index,name  in enumerate( ['tokens >10K frequency' , 'tokens with 1K-10K frequency' ,'tokens with 100-1K frequency',  'tokens <100 frequency' , 'all the tokens'] ):
        sorted_ppl_diff_vec=np.sort(np.array(ppl_baseline_bin[index])-np.array(ppl_mts_bin[index]))
        print(name, ':',  round(np.mean(sorted_ppl_diff_vec),2) , 'with CI [',round(sorted_ppl_diff_vec[4],2),round(sorted_ppl_diff_vec[9995],2),'] and standard dev:', round(np.std(sorted_ppl_diff_vec),2)  )

    return

def significance_testing(model_name, baseline_name , test_data , test_batch_size):

    mts_model = model_load(model_name)
    baseline_model = model_load(baseline_name)

    eval = Evaluate(test_data, test_batch_size)
    entropy_loss_mts, l_mts, target_vec_mts = eval.entropy_for_target_vec(mts_model, args)
    entropy_loss_baseline, l_baseline, target_vec_baseline = eval.entropy_for_target_vec(baseline_model, args)

    assert (l_mts==l_baseline and target_vec_mts==target_vec_baseline)

    mts_ppl = round(math.exp(sum(entropy_loss_mts)/l_mts),2)
    baseline_ppl = round(math.exp(sum(entropy_loss_baseline)/l_baseline),2)
    #assert (mts_ppl==57.58 and baseline_ppl==58.98)

    # Create an array of 10000x824
    Boot_matrix = generate_bootstrapped_matrix(l_mts)
    statistics_significance(entropy_loss_baseline, entropy_loss_mts, target_vec_mts, Boot_matrix)

    return

#ptb: --dropouti 0.4 --dropouth 0.25
#wiki: --dropouth 0.2


def information_routing(model_name, test_data, test_batch_size):

    model = model_load(model_name)
    eval = Evaluate(test_data, test_batch_size)

    ntokens = vocab.ntokens
    print((ntokens))
    assert(ntokens==10000)#33278

    model_ablated_l1 = model_ablation.RNNModel('LSTM', ntokens, 400, 1150, 3, 0.4, args.dropouth,args.dropouti, 0.1, 0.5, True)

    l1 = dict(model_ablated_l1.named_parameters())
    l2 = dict(model.named_parameters())

    for prm_name in l1.keys():
        l1[prm_name].data = copy.deepcopy(l2[prm_name].data)

    entropy_loss, l, target_vec = eval.model_ppl_unit_ablation(model_ablated_l1, args, partial_output=False)  # data/wikitext-2'
    ppl = round(math.exp(sum(entropy_loss) / l), 2)
    print(ppl)
    assert (ppl == 57.58)#66.06

    d_org, c_org, b_org, a_org = high_low_freq_bin_evaluation(entropy_loss, target_vec)
    print("Model ppl: from low to high frequency bin:", d_org, c_org, b_org, a_org)

    #assert(d_org == 1902.37 and c_org == 167.97 and b_org == 26.68 and a_org == 6.79)
    bin1, bin2, bin3, bin4 = [], [], [], []

    for i in range(1,24):

        entropy_loss_abl, l_abl, target_vec_abl = eval.model_ppl_unit_ablation(model_ablated_l1,args,partial_output=i)#data/wikitext-2'
        d, c, b, a = high_low_freq_bin_evaluation(entropy_loss_abl, target_vec_abl)
        print("Model ppl: from low to high frequency bin:", d/d_org, c/c_org, b/b_org, a/a_org)

        bin1.append(a/a_org)
        bin2.append(b/b_org)
        bin3.append(c/c_org)
        bin4.append(d/d_org)

    np.save('bin1.txt',np.array(bin1))
    np.save('bin2.txt',np.array(bin2))
    np.save('bin3.txt',np.array(bin3))
    np.save('bin4.txt',np.array(bin4))

    plot_unit_ablation(bin1, bin2, bin3, bin4)

    return

vocab = Vocabulary(args.data)
test_data , test_batch_size = vocab.load_data()  #data/penn or data/wikitext-2

if args.model_performance: test_model_performance(args.model_name, test_data, test_batch_size)
if args.significance_testing: significance_testing(args.model_name, args.baseline_model,test_data, test_batch_size)
if args.unit_ablation: information_routing(args.model_name, test_data, test_batch_size)

# python model_evaluation --data data/penn --dropouti 0.4 --dropouth 0.25  --model_name pTB.pt
# python model_evaluation --data data/wiki  --dropouth 0.2  --model_name Wiki.pt

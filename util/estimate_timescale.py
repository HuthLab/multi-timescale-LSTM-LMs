import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import pickle

import data
#import model as model
import model_for_viz as model_viz
import copy
#import scipy.stats
import matplotlib .pyplot as plt

from pathlib import Path
from collections import defaultdict
import sys
path = Path(__file__).parent.absolute()
sys.path.append(str(path)+'/cottoncandy')

from utils import batchify, get_batch, repackage_hidden

from splitcross import SplitCrossEntropyLoss
criterion = SplitCrossEntropyLoss(400, splits=[], verbose=False)

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')


parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')


parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

parser.add_argument('--seed', type=int, default=None, 
                    help='random seed') #1111
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--trained_model',type=str, help='Model to test')

args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
"""
np.random.seed(args.seed) 
torch.manual_seed(args.seed) # 
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else: #
        torch.cuda.manual_seed(args.seed)#
"""
###############################################################################
# function to load model
def model_load(fn):
    global model_pretrained, criterion, optimizer

    with open(fn, 'rb') as f:
        model_pretrained, criterion, optimizer = torch.load(f)

    return
###############################################################################                                                             
# Load data
import os
import hashlib

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

train_tensor = corpus.train
test_tensor  = corpus.test
val_tensor   = corpus.valid
ntokens = len(corpus.dictionary)

eval_batch_size = 10
test_batch_size = 1
    
train_data = batchify(train_tensor, args.batch_size, args)
val_data   = batchify(val_tensor, eval_batch_size, args)
test_data  = batchify(test_tensor, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

model = model_viz.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                               args.dropouti, args.dropoute, args.wdrop, args.tied)

###
if args.cuda:
    model.cuda()


def forget_gate_estimation(model_name , model_in, data_source, batch_size=10,averaging=True, sorting_forget_gate=True):
    # Turn on evaluation mode which disables dropout.                                                                                      
    
    model_in.eval()
    hidden = model_in.init_hidden(batch_size)
    
    for i in range(0, data_source.size(0) - 1, args.bptt):
        
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model_in(data, hidden)
        int2word = corpus.dictionary.idx2word

        hidden = repackage_hidden(hidden)
                

        if i!=350: continue
        
        label = np.array([int2word[ind] for ind in data[:,0:1].detach().cpu().numpy().transpose()[0]])
        
        for layer, rnn in enumerate(model_in.rnns):
                        
            out = np.array( [rnn.module.forgetgate[t][0][0] for t in range(len(rnn.module.forgetgate)) ]) 
            
            
            if sorting_forget_gate:
                mean_out = np.mean(out, axis=0)
                Index_arr = np.argsort(mean_out)
                out  = out[:, Index_arr]
            

            if averaging:
                out_s = np.hstack( [np.mean(out[:,j:j+10], axis = 1)[:,np.newaxis] for j in range(0,out.shape[1], 10)] )
            
            else:
                out_s = np.hstack( [out[:,j][:,np.newaxis] for j in range(0,out.shape[1], 10)])

            #cci.upload_npy_array('shivangi/example/%s/layer%d/output'%(model_name,layer),out_s)
            np.save('%s_forget_gate_layer_%d_output'%(model_name,layer),out_s)
        
        return
        
    return 

def timescale_estimation(model_name , model_in, data_source, batch_size=1):

    model_in.eval()
    hidden = model_in.init_hidden(batch_size)
    
    mean_forget_gate = []
    count = 0
    
    for i in range(0, data_source.size(0) - 1, args.bptt):
        
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model_in(data, hidden)
        hidden = repackage_hidden(hidden)
        count += 1
        
        for layer, rnn in enumerate(model_in.rnns):
              
            out = np.array( [rnn.module.forgetgate[t][0][0] for t in range(len(rnn.module.forgetgate)) ]) 
            mean_out = np.mean(out, axis=0)
            
            if i==0:    
                mean_forget_gate.append(mean_out)
            else: 
                mean_forget_gate[layer] += mean_out

    for layer in [0,1,2]:
        mean_forget_gate[layer] /= float(count)
        timescale = -1/ np.log(mean_forget_gate[layer])
        np.save('%s_timescale_layer_%d_output' % (model_name, layer+1), timescale)

    return

def plot_timescale(model_name):
    # ptb models
    import scipy
    timescale = scipy.stats.invgamma.isf(np.linspace(0, 1, 1151), a=0.56, scale=1)[1:]

    fig, ax = plt.subplots(figsize=(6, 4))
    for layer in range(1, 2):
        x = np.load('%s_timescale_layer_%d_output.npy' % (model_name, layer+1))

        ax.plot((x), label='Estimated', linewidth=3)
        ax.plot((timescale), 'r', label='Assigned', linewidth=3)
        ax.set_xlabel('Units')
        ax.set_ylabel('Timescale (T)')
        ax.set_yscale('log')

        for item in ([ax.title, ax.xaxis.label,
                      ax.yaxis.label] + ax.legend().get_texts() + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

    plt.tight_layout()
    plt.savefig('Estimated_timescale_invgamma.eps', format='eps')
    plt.savefig('Estimated_timescale_invgamma.png')


def evaluate(model_in, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model_in.eval()

    total_loss = 0
    hidden = model_in.init_hidden(batch_size)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model_in(data, hidden)
        total_loss += len(data) * criterion(model_in.decoder.weight, model_in.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)

    return total_loss.item() / len(data_source)

#test copied model
def test_model(test_model):


    with open(test_model, 'rb') as f:
        model_pretrained, criterion,o = torch.load(f)

    pretrained_dict = dict(model_pretrained.named_parameters())
    model_dict = dict(model.named_parameters())

    for key in model_dict.keys():
        model_dict[key].data = copy.deepcopy(pretrained_dict[key].data)

    #evaluate the model here first
    test_loss = evaluate(model, test_data, test_batch_size)


    ppl, bpc = round(math.exp(test_loss),2), round(test_loss / math.log(2),2)
    print("Word perplexity:",ppl, "BPC:",bpc)

    #assert(ppl == 57.58)
    plot_title = test_model.split('.')[0]
    timescale_estimation(plot_title, model, test_data, test_batch_size)

    #plot_timescale(plot_title)
    # forget_gate_estimation(plot_title , model, test_data, test_batch_size)

    return

test_model(args.trained_model)






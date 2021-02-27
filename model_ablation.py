import torch
import torch.nn as nn
import numpy as np

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import scipy.stats

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]

            ## start edit: steps for mts model bias assigments 

            ## STEP1: make a list of size of hidden layers - useful for init step
            hid_dim  = [nhid if l != nlayers -1 else ( ninp if tie_weights else nhid) for l in range(nlayers)]

            ## STEP2: create bias values depending on type of init we want

            chrono_bias = [np.zeros(hid_dim[l]) for l in range(nlayers)]
            multi_timescale = True

            if multi_timescale:
                #layer 0 with half units of timescale 3 and half of timescale 4
                half_length = int(0.5*hid_dim[0]) ;
                timescale_first_half, timescale_second_half = 3,4
                #calculate bias values from timescale and store in an array 
                chrono_bias[0][:half_length] =  -1 * np.log(np.exp(1/timescale_first_half)-1)  
                chrono_bias[0][half_length:] =  -1 * np.log(np.exp(1/timescale_second_half)-1) 

                #layer 1 with timescale sampled from an inverse gamma distribution
                timescale_invgamma  = scipy.stats.invgamma.isf(np.linspace(0, 1, 1151),a=0.56,scale=1)[1:]
                #calculate bias values from timescales and store in an array
                chrono_bias[1] =  -1 * np.log(np.exp(1/timescale_invgamma)-1)
  
            
            ## STEP 3: assign bias values to the layers-first half is input gate bias, second half is forget gate for both i to h and h to h

            for l in range(nlayers-1): #Assign biases for only first two layers 
                self.rnns[l].bias_ih_l0.data[0:hid_dim[l]*2] = torch.tensor(np.zeros(hid_dim[l]*2),dtype=torch.float)
                self.rnns[l].bias_hh_l0.data[0:hid_dim[l]*2] = torch.from_numpy(np.hstack((-1*chrono_bias[l], chrono_bias[l] )).astype(np.float32))
                    
            ## STEP 4: fix the bias - if we want to fix the bias instead of just init them 
            fixed_weights = True
            if fixed_weights:
                for l in range(nlayers-1):
                    print(l)
                    self.rnns[l].bias_ih_l0.requires_grad = False 
                    self.rnns[l].bias_hh_l0.requires_grad = False
    

            ##end edit
            ###
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden,partial_output, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        timescale_invgamma = scipy.stats.invgamma.isf(np.linspace(0, 1, 1151), a=0.56, scale=1)[1:]
        np.save('timescale_invgamma.txt', timescale_invgamma)
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            #print('Partial', partial_output)
            if partial_output:
                if l == 2:
                    i = partial_output
                    current_input[:,:,(i-1)*50:(i)*50] = torch.tensor(np.zeros(50) , dtype=torch.float)
                    #print(np.mean(timescale_invgamma[(i-1)*50:(i)*50]))
                    #print(torch.sum(raw_output[:,:,(i-1)*50:(i)*50] ))
                    #print(torch.sum(raw_output[:,:,(i)*50:] ))
                    #print(i*50)
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

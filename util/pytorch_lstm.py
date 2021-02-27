import torch
import torch.nn as nn 
from torch import Tensor

import torch.nn.functional as F

class LSTMOld(torch.nn.Module):
    def __init__(self,input_size,out_size,stack=1,dropout=0):
        super(LSTMOld, self).__init__()
        for layer in range(stack):
            self.weight_ih_l0 = torch.nn.Parameter(torch.Tensor(4 * out_size, input_size))
            self.weight_hh_l0 = torch.nn.Parameter(torch.Tensor(4 * out_size, out_size))
            self.bias_ih_l0   = torch.nn.Parameter(torch.Tensor(4 * out_size))
            self.bias_hh_l0   = torch.nn.Parameter(torch.Tensor(4 * out_size))
            self.forgetgate, self.inputgate,self.outputgate , self.cellgate = {},{},{},{}
            self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
                

    def forward(self,x, hidden):
        h,c=[],[]
        for t in range(len(x)):
            if t==0:  
                ht,ct = hidden[0], hidden[1] 
            else: 
                ht,ct = h[t-1],c[t-1]
            gates = F.linear(x[t:t+1,:,:], self.weight_ih_l0, self.bias_ih_l0) + F.linear(ht, self.weight_hh_l0, self.bias_hh_l0)
            
            self.inputgate[t],self.forgetgate[t],self.cellgate[t],self.outputgate[t] = gates.chunk(4,-1)
            
            self.forgetgate[t]  =  torch.sigmoid(self.forgetgate[t])
            self.inputgate[t]   =  torch.sigmoid(self.inputgate[t])
            self.outputgate[t]  =  torch.sigmoid(self.outputgate[t])
            self.cellgate[t]    =  torch.tanh(self.cellgate[t])

            ct = (self.forgetgate[t]*ct) + (self.inputgate[t]*self.cellgate[t])
            ht = self.outputgate[t] * torch.tanh(ct) 
            h.append(ht), c.append(ct)

        return torch.cat(h, dim=0) , [h[len(x)-1], c[len(x)-1]]


class LSTMNew(torch.nn.Module):
    def __init__(self,input_size,out_size,stack=1,dropout=0):
        super(LSTMNew, self).__init__()
        for layer in range(stack):
            self.weight_ih_l0 = torch.nn.Parameter(torch.Tensor(4 * out_size, input_size))
            self.weight_hh_l0 = torch.nn.Parameter(torch.Tensor(4 * out_size, out_size))
            self.bias_ih_l0   = torch.nn.Parameter(torch.Tensor(4 * out_size))
            self.bias_hh_l0   = torch.nn.Parameter(torch.Tensor(4 * out_size))
            self.forgetgate, self.inputgate,self.outputgate , self.cellgate = {},{},{},{}
            self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
                
    def forward(self,x, hidden):
        h,c=[],[]
       
        for t in range(len(x)):
            if t==0:  ht,ct = hidden[0], hidden[1] 
            else: ht,ct = h[t-1],c[t-1]
            
            gates = F.linear(x[t:t+1,:,:], self.weight_ih_l0, self.bias_ih_l0) + F.linear(ht, self.weight_hh_l0, self.bias_hh_l0)
            
            it, ft, chat_t, ot = gates.chunk(4,-1)
            
            ft      =  torch.sigmoid(ft)
            it      =  torch.sigmoid(it)
            ot      =  torch.sigmoid(ot)
            chat_t  =  torch.tanh(chat_t)
            
            
            self.forgetgate[t] = ft.detach().cpu().numpy()
            self.inputgate[t]  = it.detach().cpu().numpy()
            self.outputgate[t] = ot.detach().cpu().numpy()

            ct = (ft*ct) + (it*chat_t)
            ht = ot * torch.tanh(ct) 
            
            h.append(ht), c.append(ct)
            
        return torch.cat(h, dim=0) , [h[len(x)-1], c[len(x)-1]]

if __name__ == '__main__':

    import torch 
    from model_lstm import LSTMNew
    import numpy as np
    in_size,out_size,bs,seq_len = 2,4,3,5


    LSTM_torch = torch.nn.LSTM(in_size,out_size)
    LSTM_naive = LSTMNew(in_size,out_size);

    #data for LSTM
    x = torch.from_numpy(np.random.normal(0,15,(seq_len,bs,in_size)).astype(np.float32) );
    h = torch.from_numpy(11*np.random.random((1,bs,out_size)).astype(np.float32) );
    c = torch.from_numpy(10*np.random.random((1,bs,out_size)).astype(np.float32) );
    out = []
    #Init both LSTMs with same val 
    for LSTM in [LSTM_torch]:
        LSTM.weight_ih_l0.data =torch.from_numpy(1*np.random.normal( 0,1,(4*out_size,in_size)).astype(np.float32));
        LSTM.bias_ih_l0.data =torch.from_numpy(0*np.random.uniform(-10,10, (4*out_size)).astype(np.float32));
        LSTM.weight_hh_l0.data =torch.from_numpy(1*np.random.normal( 0,1,(4*out_size,out_size)).astype(np.float32));
        LSTM.bias_hh_l0.data =torch.from_numpy(0*np.random.uniform( -10,10,(4*out_size)).astype(np.float32));

    LSTM_naive.weight_ih_l0.data = LSTM_torch.weight_ih_l0.data    
    LSTM_naive.bias_ih_l0.data =    LSTM_torch.bias_ih_l0.data
    LSTM_naive.weight_hh_l0.data =  LSTM_torch.weight_hh_l0.data 
    LSTM_naive.bias_hh_l0.data = LSTM_torch.bias_hh_l0.data
    #x = torch.from_numpy(np.ones((seq_len,bs,in_size)).astype(np.float32) );
    
    for LSTM in [LSTM_torch, LSTM_naive]:
        y, [hf, cf] = LSTM(x,[h,c])
        out.append(y)
        print('LSTM is',LSTM, y.size())
    
    print('Output diff is', torch.sum(out[0]-out[1]).data)

"""
def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy

class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self._backend.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
"""

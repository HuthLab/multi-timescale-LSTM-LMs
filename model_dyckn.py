import torch.nn as nn
import torch
from scipy.stats import invgamma


class BaselineLSTMModel(nn.Module):
    def __init__(self, hidden_state_size, input_size=1, output_size=1, num_layers=1):
        super(BaselineLSTMModel, self).__init__()
        self.batch_first = True
        self.layers = num_layers
        # self.embed = nn.Embedding(input_size, input_size)
        self.embed = nn.Embedding.from_pretrained(torch.eye(input_size), freeze=True)  # one hot encoding
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_state_size, batch_first=self.batch_first,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_state_size, output_size)

    def forward(self, input, lengths, hidden_state=None):
        e = self.embed(input)
        e_packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=self.batch_first, enforce_sorted=True)
        out_packed, hidden = self.lstm(e_packed, hidden_state)
        out_rnn, lens = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=self.batch_first)
        output = torch.sigmoid(self.linear(out_rnn))
        return output, hidden, lens


class MTSLSTMModel(nn.Module):
    def __init__(self, hidden_state_size, input_size=1, output_size=1, init_bias=None, fixed_bias=True, num_layers=1):
        super(MTSLSTMModel, self).__init__()
        self.batch_first = True
        self.embed = nn.Embedding.from_pretrained(torch.eye(input_size), freeze=True)  # one hot encoding
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_state_size, batch_first=self.batch_first)
        self.linear = nn.Linear(hidden_state_size, output_size)
        # Custom initialization?
        if init_bias is not None:
            self.lstm.bias_ih_l0.data = torch.zeros((4*hidden_state_size))
            init_bias(self.lstm.bias_hh_l0, hidden_state_size)
        # Fixed bias vectors?
        self.lstm.bias_ih_l0.requires_grad = not fixed_bias
        self.lstm.bias_hh_l0.requires_grad = not fixed_bias

    def forward(self, input, lengths, hidden_state=None):
        e = self.embed(input)
        e_packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=self.batch_first, enforce_sorted=True)
        out_packed, hidden = self.lstm(e_packed, hidden_state)
        out_rnn, lens = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=self.batch_first)
        output = torch.sigmoid(self.linear(out_rnn))
        return output, hidden, lens


class LSTMBiasInvGammaInitializer:
    def __init__(self, alpha, scale=1.0):
        """
        Functor to initialize biases of LSTMs with an inv gamma distribution
        b_i = -b_f
        b_f = sampled from InvGamma(alpha, beta) based on number of hidden state size
        b_g, and b_o aren't touched

        :param alpha: parameter alpha for inverse gamma distribution
        :param scale: parameter scale for inverse gamma distribution (default: 1)
        """
        self.alpha = alpha
        self.scale = scale

    def __call__(self, bias, hidden_state):
        # LSTM cell order in bias: [ b_i | b_f | b_g | b_o ]
        # use hidden_state + 2 to avoid the 0 and 1 in linspace that will produce inf and -inf in the log().
        timescales = torch.tensor(invgamma.isf(torch.linspace(0, 1, hidden_state+2).numpy(),
                                               a=self.alpha,
                                               scale=self.scale)[1:-1])
        bias[hidden_state:2 * hidden_state].data.copy_(-torch.log(torch.exp(1 / timescales) - 1))
        bias[:hidden_state].data.copy_(torch.log(torch.exp(1 / timescales) - 1))

import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
    '''An LSTM class that serves as the row encoder
        
        A for loop upon implemention in the training code is required for each row.
        eg. output of CNN is batch x height x width x dimension
            for each height:
                encode(batch x width x dim)

        Args: 
        batch size - inputs have batch first (0th element)
        input size - size of the dimension from convolution (512 in this case)
        hidden size - size of the hidden layer of the encoder LSTM
        num_layers - number of LSTM layers
        gpu - default is to use gpu need to specify False if using cpu

        inputs:
        inputs: input from CNN of size [batch x width x D(512 from convolution)]
        hidden: hidden of size [layers * direction x batch size x hidden size]


        outputs:
        of size batch x width x encoder_hidden * 2 (because bidirectional)

        the encoder is hard coded to have a bidirectional LSTM


    '''

    def __init__(self, batch_size, inputs_size, hidden_size, num_layers, gpu=True):
        super(EncoderLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.inputs_size = inputs_size #the third dimension of the tensor from cnn is 512
        self.num_layers = num_layers
        self.gpu = gpu

        self.lstm = nn.LSTM(inputs_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
    
    def forward(self, inputs, hidden):
        
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self):
        #initiates the hidden and cell states respectivly
        #h_0 and c_0 (num_layer * num_directions (2), batch, hidden_size)
        if self.gpu:
            return (Variable(torch.zeros((2, self.batch_size, self.hidden_size)).cuda()),
                    Variable(torch.zeros((2, self.batch_size, self.hidden_size)).cuda()))
        else:
            return (Variable(torch.zeros((2, self.batch_size, self.hidden_size))),
                    Variable(torch.zeros((2, self.batch_size, self.hidden_size))))

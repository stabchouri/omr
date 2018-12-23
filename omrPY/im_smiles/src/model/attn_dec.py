import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from attention import Attn



class ATTNDecoder(nn.Module):
	'''
	A class for using an attention decoder.

	Args:
	batch size - size of the batch. in this model the batch is the 0th element
	inputs size - the output size from the encoder
	vocab size - the size of the vocabulary output that is possible
	hidden size - hidden size of the LSTM. for this model it will be of the size encoder size * 2 since
	             the previous context is concat with the embedding
	dropout(optional) - dropout of the embedded layer. default is 0.01
	gpu(optional) - gpu is default
	
	inputs:
	prev input: previous input to the decoder. size is [batch size]
	previous context: the previous context from the last iteration (right now not implemented but still here because other attention mechanisms may use them)
	encoder out: the output from the encoder. the shape of this has to be [batch x length x dim]
		(output from the encoder is a list of encoded rows that are then cat along the width dim = shape of batch x height x width x dim
		this is reshaped to batch x length x dim)
	hidden: the hidden from the previous iteration


	outputs:
	output: probability over the vocab size (batch x vocab size)
	context: the applied attention
	hidden: the hidden layer of the lstm (batch x 1 x hidden size)
	attn_weights
	'''

	def __init__(self, batch_size, inputs_size, vocab_size, hidden_size,\
		dropout_p=0.01, gpu=True):
		super(ATTNDecoder, self).__init__()

		self.num_layers = 1
		self.batch_size = batch_size
		self.inputs_size = inputs_size
		self.hidden_size = hidden_size
		self.dropout_p = dropout_p
		self.vocab_size = vocab_size
		self.gpu = gpu


		self.embedding = nn.Embedding(vocab_size, inputs_size, padding_idx=0)
		self.dropout = nn.Dropout(dropout_p)
		self.gru = nn.GRU(inputs_size * 2, hidden_size, batch_first=True)
		self.attention = Attn('concat', hidden_size)
		self.out = nn.Linear(hidden_size, vocab_size)

	def forward(self, prev_input, prev_context, encoder_out, hidden):

		if self.gpu:
			embedded = self.embedding(prev_input.cuda()).view(self.batch_size, 1, -1) #batch x hidden -> batch size x 1 x hidden
			embedded = self.dropout(embedded).cuda() #batch x 1 x hidden
		else:
			embedded = self.embedding(prev_input).view(self.batch_size, 1, -1)
			embedded = self.dropout(embedded)
		
		attn_weights = self.attention(hidden, encoder_out)
		context = attn_weights.bmm(encoder_out)
		rnn_input = torch.cat((embedded, context), 2)
		output, hidden = self.gru(rnn_input, hidden)
		output = self.out(output) #if chage from crossentropy loss log_softmax needs to be added

		
		return output, context, hidden, attn_weights


	def init(self):
		
		if self.gpu:
			return Variable(torch.zeros((self.num_layers, self.batch_size, self.hidden_size)).cuda()),\
					Variable(torch.zeros((self.batch_size, self.num_layers, self.hidden_size)).cuda())
		else:
			return Variable(torch.zeros((self.num_layers, self.batch_size, self.hidden_size))),\
					Variable(torch.zeros((self.batch_size, self.num_layers, self.hidden_size)))




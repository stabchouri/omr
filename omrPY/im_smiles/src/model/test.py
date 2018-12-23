import sys, time, random, logging, operator
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils  
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as tv
from PIL import Image
from cnn import CNNModel
from encoder import EncoderLSTM
from attn_dec import ATTNDecoder
sys.path.append('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles')
from data_gen import data_generator
from train_utils import *

'''
Script horribly set up right now. Is set to only test one image. Used in debugging to make sure outputs are reasonable
TODO:
needs bucket generations / usage of validation set
set up for cpu usage?


'''

image_directory = '/scratch2/sophiat/chem-ie-TJS_omrPY/omr/data/processed/out1_flipped_processed'
enc_hidden_size = 256
dec_hidden_size = 256
enc_layers = 1

img_list = [100]

img_batch = torch.FloatTensor() #empty tensorls for batches

for i in img_list:
	old = Image.open(image_directory+'/'+str(i)+'.png').convert('L')
	loader = tv.Compose([tv.ToTensor()])
	single_img = loader(old).unsqueeze(0)
	img_batch = torch.cat((img_batch, single_img), dim=0)
	print("single image size="+str(old.size))

print('Images loaded shape: {}'.format(img_batch.shape))

batch_size = img_batch.size(0)

idx_vocab, vocab_idx = data_generator.Vocab_Idx('/scratch2/sophiat/chem-ie-TJS_omrPY/omr/data/smiles_vocab.txt').gen()
target_loc = '/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/data/out1.txt'
target_tens = torch.Tensor()
lens=[]
print(vocab_idx)
with open(target_loc, 'r') as f:
	lines = f.readlines()
	target_text = [lines[l-1].split() for l in img_list]
	print(target_text)
	max_length = max([len(x) for x in target_text])
	for i in target_text:
		print('i='+str(i))
		i += ['null'] * (max_length - len(i))
		target_tens = torch.cat((target_tens, torch.Tensor([vocab_idx[x] for x in i]).unsqueeze(0)), dim=0)

print(target_tens)

vocab_size = len(idx_vocab)

input_tensor = Variable(img_batch.cuda())
target_tensor =  Variable(target_tens.cuda())

def test(inp_tensor, target_tensor, convolution, encoder, decoder, gpu):
	
	#CNN
	conv_out = convolution.forward(inp_tensor)
	convolution.parameters()
	
	batch_size = conv_out.size(0) #batch_size
	img_ht = conv_out.size(1) #height
	img_wid = conv_out.size(2) #width
	conv_size = conv_out.size(3) #512
	
	rows_enc = []
	enc_hid = encoder.init_hidden()
	for i in range(conv_out.size(1)):
		enc_outs, enc_hid = encoder.forward(conv_out[:,i], enc_hid) #shape of enc_out (batch x W x D)
		rows_enc.append(enc_outs) #list of enc_outs which are the rows of the convolution so list of number h's [tensor(batch x w x d)]

	#V_t = torch.stack(V_cap, dim=1).view(batch_size, -1, enc_hidden_size*2)
	stacked = torch.stack(rows_enc, dim=1).view(batch_size, -1, enc_hidden_size*2)
	enc_total = stacked[:,:,:enc_hidden_size] + stacked[:,:,enc_hidden_size:] # sums fw and bw lstm for all the rows

	prev_in = Variable(torch.zeros((batch_size)).long())
	top_p = np.empty((1,1))
	prediction = torch.LongTensor()
	dec_hid, prev_cont  = decoder.init()

	for i in range(target_tensor.size(1)):
		dec_output, prev_cont, dec_hid, attn_weights = decoder.forward(prev_in, prev_cont, enc_total, dec_hid)
		topv, topi = dec_output.topk(1)
		top_p = np.append(top_p, topi.cpu().squeeze(1).data.numpy(), axis=1)
		prediction = torch.cat((prediction, topi.data.cpu()))
		#print(topi.squeeze(1).size())
		prev_in = topi.squeeze().detach()
		#print(prev_in)

	num_correct = float((prediction.view(-1) == target_tens.view(-1).long()).sum())
	accuracy = (num_correct / operator.mul(*target_tens.shape)) * 100.
	print('Number correct: {}, Accuracy: {}'.format(num_correct, accuracy))


	return top_p






if __name__ == '__main__':
	conv = CNNModel().cuda()
	#self, batch_size, inputs_size, img_w, hidden_size, num_layers
	enc = EncoderLSTM(batch_size, 512, 46, enc_hidden_size, enc_layers, gpu=True).cuda().eval()
	#self, batch_size, inputs_size, vocab_size, hidden_size, max_decoder_l, dropout_p=0.01
	dec = ATTNDecoder(batch_size, enc_hidden_size ,1,  vocab_size, dec_hidden_size, gpu=True).cuda().eval()
	conv.load_state_dict(torch.load('/Users/thomasstruble/Documents/GitHub/chem-ie/omrPY/im_smiles/src/model/model_conv'))
	enc.load_state_dict(torch.load('/Users/thomasstruble/Documents/GitHub/chem-ie/omrPY/im_smiles/src/model/model_enc' ))
	dec.load_state_dict(torch.load('/Users/thomasstruble/Documents/GitHub/chem-ie/omrPY/im_smiles/src/model/model_dec' ))

	pred = test(input_tensor, target_tens, conv, enc, dec, gpu=True)
	print(pred)

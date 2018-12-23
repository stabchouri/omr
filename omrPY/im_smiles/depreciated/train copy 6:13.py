import sys, time, random, logging
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
sys.path.append('/Users/thomasstruble/Documents/GitHub/chem-ie/omrPY/im_smiles/data_gen')
from data_gen import Vocab_Idx
from train_utils import *
from tqdm import tqdm

'''
TODO:
-implement the bucketing data generation
-need to make sure vocabulary list is correct
-verify code can be used on multiple gpu's -- right now confirmed to work on cpu(slow) and one gpu
-

'''

logging.basicConfig(filename='tests.log', filemode='w', level=logging.INFO)
BASE_DIR ='/Users/thomasstruble/Documents/GitHub/chem-ie/'

image_directory = BASE_DIR + 'omr/data/processed/png_processed'
max_length = 100
enc_hidden_size = 256
dec_hidden_size = 256
enc_layers = 1
batch_size = 5

img_lists = list(chunks(range(0,5000),batch_size)) 
idx_vocab, vocab_idx = Vocab_Idx(BASE_DIR + 'omrPY/data/smiles_vocab.txt').gen()
target_loc = BASE_DIR + 'omr/data/smiles_tokens.txt'

vocab_size = len(idx_vocab) #index 0 is null an is used for the padding of sequences


def train(inp_tensor, target_tensor, convolution, encoder, decoder, cnn_optimizer, encoder_optimizer,\
	decoder_optimizer, criterion, teacher_forcing_ratio, learning_rate, gpu=False):
	
	lr = learning_rate
	loss = 0


	#CNN
	conv_out = convolution.forward(inp_tensor)
	#shape at this point is tensor(batch x height x width x D(512))
	
	#Row encoder. takes all the height dimensions and runs them through LSTM
	rows_enc = []
	enc_hid = encoder.init_hidden()
	for i in range(conv_out.size(1)):
		enc_outs, enc_hid = encoder.forward(conv_out[:,i], enc_hid) #shape of enc_out (batch x W x D)
		rows_enc.append(enc_outs) #list of enc_outs which are the rows of the convolution so list of number h's [tensor(batch x w x d)]

	stacked = torch.stack(rows_enc, dim=1).view(batch_size, -1, enc_hidden_size*2) #[batch x H x W x D*2] -> [batch x L(h+w) x D*2]
	enc_total = stacked[:,:,:enc_hidden_size] + stacked[:,:,enc_hidden_size:] # sums fw and bw lstm for all the L [batch x L x D]

	#Decoder
	prev_in = Variable(torch.zeros((batch_size)).long())
	top_p = np.empty((batch_size,1))
	dec_hid, prev_cont  = decoder.init()

	teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	
	if teacher_forcing:
		for i in range(target_tensor.size(1)):
			dec_output, prev_cont, dec_hid, attn_weights = decoder.forward(prev_in, prev_cont, enc_total, dec_hid)
			topv, topi = dec_output.squeeze(1).topk(1)
			top_p = np.append(top_p, topi.cpu().data.numpy(), axis=1)
			prev_in = target_tensor[:,i].long() # teacher forcing
			loss += criterion(dec_output.squeeze(1), target_tensor[:,i].long())
	else:
		for i in range(target_tensor.size(1)):
			dec_output, prev_cont, dec_hid, attn_weights = decoder.forward(prev_in, prev_cont, enc_total, dec_hid)
			topv, topi = dec_output.topk(1)
			top_p = np.append(top_p, topi.cpu().squeeze(1).data.numpy(), axis=1)
			prev_in = topi.squeeze().detach()
			loss += criterion(dec_output.squeeze(1), target_tensor[:,i].long())
	
	#zero the gradients before we caluclate the gradients w.r.t the loss in the backward pass
	cnn_optimizer.zero_grad()
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	loss.backward()

	#gradient clipping
	torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.25)
	for p in encoder.parameters():
		p.data.add_(-lr, p.grad.data)
	torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.25)
	for p in decoder.parameters():
		p.data.add_(-lr, p.grad.data)
	
	cnn_optimizer.step()
	encoder_optimizer.step()
	decoder_optimizer.step()
	

	return loss.item() / target_tensor.size(1), top_p


def trainIters(convolution, encoder, decoder, epochs, print_every=1000, plot_every=100, learning_rate=0.01, teacher_forcing_ratio=0.3, gpu=False):
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	cnn_optimizer = optim.SGD(convolution.parameters(), lr=learning_rate)
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


	if gpu:
		criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()
		#criterion = nn.NLLLoss().cuda()
	else:
		criterion = nn.CrossEntropyLoss()
		#criterion = nn.NLLLoss()

	for iters in range(1, epochs + 1):
		dropped = 0
		for batch_num, imgs in enumerate(tqdm(img_lists)):
			img_batch = torch.FloatTensor() #empty tensors for batches
			for img in imgs:
				old = Image.open(image_directory+'/'+str(img+1)+'.png').convert('L')
				loader = tv.Compose([tv.ToTensor()])
				single_img = loader(old).unsqueeze(0)
				img_batch = torch.cat((img_batch, single_img), dim=0)

			target_tens = torch.Tensor()
			lens=[]
			#TODO bucket data loader here
			try:
				with open(target_loc, 'r') as f:
					lines = f.readlines()
					target_text = [lines[l-1].split() for l in imgs]
					max_length = max([len(x) for x in target_text])
					for i in target_text:
						i += ['null'] * (max_length - len(i))
						target_tens = torch.cat((target_tens, torch.Tensor([vocab_idx[x] for x in i]).unsqueeze(0)), dim=0)
			except:
				logging.warning('Dropped vector: {}, Total Dropped: {}'.format(imgs, dropped))
				dropped += 1
				continue

			if max_length > 100:
				logging.warning('Dropped vector: {}, Total Dropped: {}'.format(imgs, dropped))
				dropped += 1
				continue

			if gpu:
				input_tensor = Variable(img_batch.cuda(), requires_grad=False)
				target_tensor =  Variable(target_tens.cuda(), requires_grad=False)
			else:
				input_tensor = Variable(img_batch)
				target_tensor =  Variable(target_tens)

			loss, top_preds = train(input_tensor, target_tensor, convolution, encoder,
						decoder, cnn_optimizer, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, gpu, learning_rate)
			print_loss_total += loss
			plot_loss_total += loss


			if batch_num % print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0
				print('{} ({} {}) {}'.format(timeSince(start, (iters / float(epochs))),
											iters, iters / epochs * 100, print_loss_avg))
				logging.info('Top P: {}, target: {}'.format(top_preds * 10, target_tensor.cpu().data.numpy()))
				logging.info('{} ({} {}) {}'.format(timeSince(start, (iters / float(epochs))),
											iters, iters / epochs * 100, print_loss_avg))
				torch.save(convolution.state_dict(),BASE_DIR + 'omrPY/im_smiles/src/model/model_conv' )
				torch.save(encoder.state_dict(),BASE_DIR + 'omrPY/im_smiles/src/model/model_enc' )
				torch.save(decoder.state_dict(),BASE_DIR + 'omrPY/im_smiles/src/model/model_dec' )

			if iters % plot_every == 0:
				plot_loss_avg = plot_loss_total / plot_every
				plot_losses.append(plot_loss_avg)
				plot_loss_total = 0



if __name__ == '__main__':
	gpu=False
	if gpu:
		conv = CNNModel().cuda()
		enc = EncoderLSTM(batch_size, 512, enc_hidden_size, enc_layers, gpu=True).cuda()
		dec = ATTNDecoder(batch_size, enc_hidden_size, vocab_size, dec_hidden_size, dropout_p=0.01, gpu=True).cuda()
		trainIters(conv, enc, dec, 1, print_every=100, plot_every=10, teacher_forcing_ratio=1.0, gpu=True)
	else:
		conv = CNNModel()
		enc = EncoderLSTM(batch_size, 512, enc_hidden_size, enc_layers, gpu=False)
		dec = ATTNDecoder(batch_size, enc_hidden_size, vocab_size, dec_hidden_size, dropout_p=0.01, gpu=False)
		trainIters(conv, enc, dec, 1, print_every=100, plot_every=100, teacher_forcing_ratio=1.0, gpu=False)

	torch.save(conv.state_dict(),BASE_DIR + 'omrPY/im_smiles/src/model/model_conv')
	torch.save(enc.state_dict(),BASE_DIR + 'omrPY/im_smiles/src/model/model_conv')
	torch.save(dec.state_dict(),BASE_DIR + 'omrPY/im_smiles/src/model/model_conv')
	
	
	
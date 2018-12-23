#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:02:09 2018

@author: sophiatabchouri
"""

import sys, time, random, logging, operator
import numpy as np
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
sys.path.append('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/data_gen')
#from data_gen import Vocab_Idx
from train_utils import *
from data_generator import DataGenerator, Vocab_Idx

'''
Script horribly set up right now. Is set to only test one image. Used in debugging to make sure outputs are reasonable
TODO:
needs bucket generations / usage of validation set
set up for cpu usage?
'''

img_feed = '/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/data/train_filter.lst'
target_smiles = '/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/data/smiles.120k'
vocab_directory = '/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/data/smiles_vocab.txt'
idx_vocab, vocab_idx = Vocab_Idx('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/data/smiles_vocab.txt').gen()
vocab_size = len(idx_vocab)

enc_hidden_size = 256
dec_hidden_size = 256
enc_layers = 1
BATCH_SIZE = 5


def get_batch_tensors(line_nums):
    img_feed_f = open(img_feed,'r')
    imgs = img_feed_f.readlines()
    img_batch = torch.FloatTensor()
    target_tens = torch.Tensor()
    target_text = []
    
    with open(target_smiles, 'r') as f:
        lines = f.readlines()
        for i in imgs:
            
            img_file,line_num = i.split()
            
            #add image to data tensor
            img_path = "/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/data/"+img_file
            old = Image.open(img_path).convert('L')
            loader = tv.Compose([tv.ToTensor()])
            single_img = loader(old).unsqueeze(0)
            img_batch = torch.cat((img_batch, single_img), dim=0)
            
            #add test label to list 
            target_text.append(lines[int(line_num)].split())
        
        
        max_length = max([len(x) for x in target_text])
        for i in target_text:
            i += ['null'] * (max_length - len(i))
            target_tens = torch.cat((target_tens, torch.Tensor([vocab_idx[x] for x in i]).unsqueeze(0)), dim=0)
            
    
    #print('Images loaded shape: {}'.format(img_batch.shape))
    
    input_tensor = Variable(img_batch.cuda())
    target_tensor =  Variable(target_tens.cuda())
    
    return (input_tensor,target_tensor)

def test_iters():
    s_batch = DataGenerator(img_feed, target_smiles, vocab_directory)
    for shape in s_batch.file_dict.keys():
        batch_num = 1
        for b in s_batch.gen_batches(shape, batch_size=BATCH_SIZE):
            img_batch, target_tens = b
            
	    input_tensor = Variable(img_batch.cuda(), requires_grad=False)
	    target_tensor =  Variable(target_tens.cuda(), requires_grad=False)
    	    
	    preds = test(input_tensor, target_tensor, conv, enc, dec)
    print(preds)
    return preds

    
def test(inp_tensor, target_tensor, convolution, encoder, decoder):
	
    #CNN
    conv_out = convolution.forward(inp_tensor)
	
    batch_size = conv_out.size(0) #batch_size
    img_ht = conv_out.size(1) #height
    img_wid = conv_out.size(2) #width
    conv_size = conv_out.size(3) #512
    
    rows_enc = []
    enc_hid = encoder.init_hidden()
    for i in range(conv_out.size(1)):
        enc_outs, enc_hid = encoder.forward(conv_out[:,i], enc_hid) #shape of enc_out (batch x W x D)
        rows_enc.append(enc_outs) #list of enc_outs which are the rows of the convolution so list of number h's [tensor(batch x w x d)]
    stacked = torch.stack(rows_enc, dim=1).view(batch_size, -1, enc_hidden_size*2)
    enc_total = stacked[:,:,:enc_hidden_size] + stacked[:,:,enc_hidden_size:] # sums fw and bw lstm for all the rows

    prev_in = Variable(torch.zeros((batch_size)).long())
    top_p = np.empty((BATCH_SIZE,1))
    prediction = torch.LongTensor()
    dec_hid, prev_cont  = decoder.init()

    for i in range(target_tensor.size(1)):
        dec_output, prev_cont, dec_hid, attn_weights = decoder.forward(prev_in, prev_cont, enc_total, dec_hid)
        topv, topi = dec_output.topk(1)
        top_p = np.append(top_p, topi.cpu().squeeze(1).data.numpy(), axis=1)
        prediction = torch.cat((prediction, topi.data.cpu()))
        prev_in = topi.squeeze().detach()
        
    # print('prediction: {}'.format(prediction.view(target_tensor.size(0), -1)))
    # print('target: {}'.format(target_tensor.cpu().long().data))
     
    #print('prediction=',str(prediction.view(-1).shape))
    #print('target =', str(target_tensor.view(-1).shape))
    #print('prediction')
    #print(prediction.view(-1))
    #print('target')
    #print(target_tensor.view(-1))
    #	print('dec output=',dec_output.squeeze(1).shape)
    #	print('target =', target_tensor[i,:].shape)
    #	print('dec output')
    #	print(dec_output.squeeze(1))
    #	print('target')
    #	print(target_tensor[i,:].long())
    num_correct = float((prediction.view(-1) == target_tensor.cpu().view(-1).long().data).sum())
    accuracy = (num_correct / operator.mul(*target_tensor.shape)) * 100.
    print('Number correct: {}, Accuracy: {}'.format(num_correct, accuracy))
#    return top_p



if __name__ == '__main__':
    conv = CNNModel().cuda()
    #self, batch_size, inputs_size, img_w, hidden_size, num_layers
    enc = EncoderLSTM(BATCH_SIZE, 512, enc_hidden_size, enc_layers, gpu=True).cuda().eval()
    #self, batch_size, inputs_size, vocab_size, hidden_size, max_decoder_l, dropout_p=0.01
    dec = ATTNDecoder(BATCH_SIZE, enc_hidden_size, vocab_size, dec_hidden_size, gpu=True).cuda().eval()
    conv.load_state_dict(torch.load('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/model_conv'))
    enc.load_state_dict(torch.load('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/model_enc'))
    dec.load_state_dict(torch.load('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/model_dec'))

    pred = test_iters()
    print(pred)

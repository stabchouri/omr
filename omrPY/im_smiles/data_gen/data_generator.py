import os, sys, logging
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
import torch
import torchvision.transforms as tv
sys.path.append('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model')
from train_utils import *


class Vocab_Idx(object):

    def __init__(self, vocab_file):
        self.vocab_file = vocab_file


    def gen(self):
        with open(self.vocab_file) as f:
            idx_vocab = {i:j.split()[0] for i,j in enumerate(f)}
            vocab_idx = {j:i for i,j in idx_vocab.iteritems()}
        

        return idx_vocab, vocab_idx

class DataGenerator:
	'''
	Class that groups images by size for faster training and feed it to the model
	Args:
	feed_file - file that has image path and smiles index for training
	smiles_tokens - a file that has the smiles strings that have been tokenized
	vocab - a file that has one vocab token per line (0 is null)

	'''

	def __init__(self, feed_file, smiles_tokens, vocab, max_len=100):
		self.feed_file = feed_file
		self.smiles_tokens = smiles_tokens
		self.vocab = vocab
		self.max_len = max_len
		self.dropped = []
		
		with open(self.feed_file, 'r') as f:
			file_names = f.readlines()
			random.shuffle(file_names)
			self.file_dict = {}
			size_list = []
			for i in file_names:
				file = i.strip().split()
				img = Image.open('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/data/'+file[0])
				if img.size in self.file_dict:
					self.file_dict[img.size].append(file)
				else:
					self.file_dict[img.size] = [file]

		self.idx_vocab, self.vocab_idx = Vocab_Idx(self.vocab).gen()

	#generator for batches - hopefully will keep memory usage down 
	def gen_batches(self, im_size, batch_size):
		for i in range(0, len(self.file_dict[im_size]), batch_size):
			batch = self.file_dict[im_size][i: i + batch_size]
			sort_batch = self.sort(batch) #sort from longest to shortest and pad with zeros
			img_tensor, target_tensor = self.data_to_tensors(sort_batch, batch_size)
			
			yield img_tensor, target_tensor


	def data_to_tensors(self, batch, batch_size):
		#input should be a sorted batch by highest size to lowest size
		#print('data to tensors')
		img_batch = torch.FloatTensor()
		target_batch = torch.FloatTensor()

		#find the max length of the batch so that we can pad up the that size
		lengs = [len(x[1].split()) if len(x[1].split()) < self.max_len else 0 for x in batch] 
		max_length = max(lengs)

		for entry in batch:
			smiles = entry[1].split()
			#drop all sequences less than the designated max length
			if len(smiles) > self.max_len:
				self.dropped.append(entry[0])
				continue
			else:
				#every once an awhile there is a token that is not in the vocab for some reason. This will skip it and log it
				try:
					smiles += ['null'] * (max_length - len(smiles))
					#print('here')
					#print(smiles)
					#print('ok')
					b = [self.vocab_idx[x] for x in smiles]
					#print('b='+str(b))
					target_batch = torch.cat((target_batch, torch.Tensor([self.vocab_idx[x] for x in smiles]).unsqueeze(0)), dim=0)
				except:
					#TODO logger
					continue

				old_im = Image.open('/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/im_smiles/src/model/data/'+entry[0]).convert('L')
				loader = tv.Compose([tv.ToTensor()])
				single_img = loader(old_im).unsqueeze(0)
				img_batch = torch.cat((img_batch, single_img), dim=0)

		
		# ***** bad workaround for now but padding tensors up to batch size *****
		print(img_batch.size())
		img_dif = batch_size - img_batch.size()[0]
		targ_dif = batch_size - target_batch.size()[0]
		for i in range(img_dif):
			img_batch = torch.cat((img_batch, torch.zeros(1, 1, img_batch.size()[2], img_batch.size()[3])), dim=0)
		for i in range(targ_dif):
			target_batch = torch.cat((target_batch, torch.zeros(1, target_batch.size()[1])), dim=0)

		return img_batch, target_batch
				

	def sort(self, batch):
		with open(self.smiles_tokens, 'r') as smiles:
			lines = smiles.readlines()
		unsorted_batch = []
		for i in batch:
			unsorted_batch.append([i[0], lines[int(i[1])-1]])
		sorted_batch = sorted(unsorted_batch, key = lambda x: len(x[1]), reverse=True)

		return sorted_batch



if __name__ == '__main__':
	feed ='/scratch2/sophiat/chem-ie-TJS_omrPY/omr/data/new.txt'
	smiles = '/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/data/smiles_tokens.txt'
	vocab = '/scratch2/sophiat/chem-ie-TJS_omrPY/omrPY/data/smiles_vocab.txt'
	# for i in DataGenerator(feed, smiles, vocab).gen_batches((800,800), batch_size=10):
	# 	new, k = i
	# 	print('generator: {}, {}'.format(new.shape, k.shape))

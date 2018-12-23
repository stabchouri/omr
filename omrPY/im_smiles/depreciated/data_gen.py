#adapted from https://github.com/da03/Attention-OCR/blob/master/src/data_util/data_gen.py

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from bucket import BucketData



class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 #img_width_range = (12, 320),
                 img_width_range = (360, 800),
                 word_len = 300):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """
        self.idx_vocab, self.vocab_idx = Vocab_Idx(data_root + 'smiles_vocab.txt').gen()
        img_height = 160
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            #self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
                             #(int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
                             #(int(math.ceil(img_width_range[1] / 4)), word_len + 2)]
            self.bucket_specs = [(13, 46),
                                 (21, 51),
                                 (26, 63),
                                 (101, 101),
                                 (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]


        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                img_path, lex = l.strip().split()
                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]
                    print('width: {}'.format(img_bw.shape))

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, img_path)# os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def read_data(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        for c in lex:
            #assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            word.append(
                ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
            #word.append(self.vocab_idx[c])

        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word

def group_by_size():
    with open('/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/new.txt', 'r') as f:
        file_names = f.readlines()
        file_dict = {}
        size_list = []
        for i in file_names:
            file = i.strip().split()[0]
            img = Image.open(file)
            if img.size in file_dict:
                file_dict[img.size].append(file)
            else:
                file_dict[img.size] = [file]
    return file_dict


class Vocab_Idx(object):

    def __init__(self, vocab_file):
        self.vocab_file = vocab_file


    def gen(self):
        with open(self.vocab_file) as f:
            idx_vocab = {i:j.split()[0] for i,j in enumerate(f)}
            vocab_idx = {j:i for i,j in idx_vocab.iteritems()}
        

        return idx_vocab, vocab_idx

def text_file_gen():
    with open('/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/new.txt', 'w') as f:
        with open('/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/out1.txt', 'r') as t:
            lines = t.readlines()
            for i in range(1, 9000):
                f.write('/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/processed/png_processed/' + str(i) +'.png' + ' ' + str(i) + '\n')

def test_gen():
    img_height = 160
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = DataGen('/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/', '/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/new.txt')
    count = 0
    for batch in s_gen.gen(10):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == img_height
    print(count)

def test_read():
    img_path = '/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/processed/png_processed/1.png'
    lexs = 'CC'
    s_gen = DataGen('/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/', '/Users/thomasstruble/Documents/GitHub/chem-ie/omr/data/test_dsr2_filter.lst')
    print(s_gen.read_data(img_path, lexs))




if __name__ == '__main__':
    text_file_gen()
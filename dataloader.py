from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from joblib import Parallel, delayed
import json
import h5py
import os
import warnings
warnings.filterwarnings('ignore')
import tables
# import multiprocessing
# from multiprocessing import Process
# from multiprocessing.dummy import Pool as ThreadPool
# from pathos.multiprocessing import Pool
# import pathos.pools as pp
import numpy as np
import random
import torch
from torchvision import transforms as trn

# from flair.data import Sentence
# from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# def unwrap_self(args):
#     cls, arg = args
#     return cls.get_batch_one(arg)
def func(*args, **kwargs):
    return DataLoader.get_batch_one(args, kwargs)

def get_img(args):
    h5_image_file, ix = args
    temp_h5 = tables.open_file(h5_image_file, mode='r')
    img = np.array(temp_h5.root.images[ix, :, :, :])
    img_batch = preprocess(torch.from_numpy(img.astype('float32') / 255.0)).numpy()
    temp_h5.close()
    return img_batch

def get_sen_embed(args):
    h5_sen_file, sen_ix = args
    temp_h5 = h5py.File(h5_sen_file, mode='r')
    sen_embed = np.stack(temp_h5['average'][sen_ix, :]).transpose()
    temp_h5.close()
    return sen_embed

def get_sen_pointer(args):
    h5_sen_pointer_file, sen_ix = args
    temp_h5 = h5py.File(h5_sen_pointer_file, mode='r')
    sen_pointer = np.stack(temp_h5['average'][sen_ix, :]).transpose()
    temp_h5.close()
    return sen_pointer

def combine(args):
    h5_image_file, ix, h5_sen_file, sen_ix = args
    arg1 = h5_image_file, ix
    arg2 = h5_sen_file, sen_ix
    img = get_img(arg1)
    sen = get_sen_embed(arg2)
    return img, sen

def combine3(args):
    h5_image_file, ix, h5_sen_file, sen_ix, h5_sen_pointer_file, sen_pointer_ix = args
    arg1 = h5_image_file, ix
    arg2 = h5_sen_file, sen_ix
    arg3 = h5_sen_pointer_file, sen_pointer_ix
    img = get_img(arg1)
    sen = get_sen_embed(arg2)
    sen_pointer = get_sen_pointer(arg3)
    return img, sen, sen_pointer


class DataLoader:
    
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = self.opt.seq_per_img
        self.num_thread = 1
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        self.max_word_length = self.opt.word_length

        print('DataLoader loading h5 file: ', opt.input_label_h5, opt.input_image_h5)
        self.h5_label_file = tables.open_file(self.opt.input_label_h5, driver="H5FD_CORE")
        self.h5_image_file = tables.open_file(self.opt.input_image_h5, mode='r')

        if opt.word_embed_att:
            print('DataLoader loading h5 file: ', self.opt.related_w_h5)
            self.related_word_file = tables.open_file(self.opt.related_w_h5, driver='H5FD_CORE')
            if opt.pointer_matching:
                print('DataLoader loading h5 file(for matching supervision): ', self.opt.pointer_matching_h5)
                self.match_file = tables.open_file(self.opt.pointer_matching_h5, driver='H5FD_CORE')
            if opt.index_size != -1:
                print('DataLoader loading h5 file: ', self.opt.retr_w_index_h5)
                self.retr_w_index_file = tables.open_file(self.opt.retr_w_index_h5, driver='H5FD_CORE')

        # load sentence embedding
        if 'sentence_embed' in opt:
            if opt.sen_sim_init:
                print('DataLoader loading h5 file: ', self.opt.sim_h5)
                self.sim_sent_file = tables.open_file(self.opt.sim_h5, driver='H5FD_CORE')
            if opt.sentence_embed:
                self.h5_sen_embed_file = h5py.File(self.opt.sentence_embed, mode='r')
                self.sen_embed_keys = json.load(open(self.opt.sentence_embed.split('.h5')[0] + '_keys.json'))
        else:
            self.opt.sentence_embed = False

        # extract image size from dataset
        images_size = self.h5_image_file.root.images.shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'
        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        print('read %d images of size %dx%dx%d' %(self.num_images,
                    self.num_channels, self.max_image_size, self.max_image_size))

        # load in the sequence data
        seq_size = self.h5_label_file.root.labels.shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        self.label_start_ix = np.array(self.h5_label_file.root.label_start_ix)
        self.label_end_ix = np.array(self.h5_label_file.root.label_end_ix)

        # id to image name(article id) dictionary
        if 'goodnews' in opt.dataset:
            self.id_to_keys = {i['id']: i['file_path'].split('/')[1].split('_')[0] for i in self.info['images']}
        elif 'breakingnews' in opt.dataset:
            self.id_to_keys = {i['id']: i['file_path'].split('/')[1].split('_')[0].replace('a', '').replace('n', '') for i in self.info['images']}
        else:
            print('dataset error', opt.dataset)
            exit(0)

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            # ix == info['images'][ix]['id']
            assert ix == self.info['images'][ix]['id']
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        # self.shuffle stores the index for image id
        self.shuffle = {'train': np.random.permutation(np.arange(len(self.split_ix['train']))),  # disrupt the training order
                        'val': np.arange(len(self.split_ix['val'])),
                        'test': np.arange(len(self.split_ix['test']))}

    def __len__(self):
        return len(self.split_ix['train'])

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    """def get_batch_one(self, split):
        split_ix = self.split_ix[split]
        batch_size = 1
        img_batch = np.ndarray([batch_size, 3, 256, 256], dtype='float32')
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='float32')
        if self.opt.sentence_embed:
            sen_embed_batch = np.zeros(
                [batch_size * self.seq_per_img, self.opt.sentence_length + 1, self.opt.sentence_embed_size],
                dtype='float32')
        max_index = len(split_ix)
        infos = []
        b_id = self.shuffle[split][self.iterators[split]: self.iterators[split] + batch_size][0]
        self.iterators[split] += batch_size
        if self.iterators[split] >= max_index:
            np.random.shuffle(self.shuffle[split])
            self.iterators[split] = 0

        i=0
        ix = split_ix[b_id]

        # fetch image
        # img = self.load_image(self.image_info[ix]['filename'])
        # img = np.array(self.h5_image_file['images'][ix, :, :, :])
        img = np.array(self.h5_image_file.root.images[ix, :, :, :])
        img_batch[i] = preprocess(torch.from_numpy(img.astype('float32') / 255.0)).numpy()

        # fetch the sequence labels # ix1==ix2 when a image only has one caption
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < self.seq_per_img:  # seq_per_img defaults to 1
            # we need to subsample (with replacement)
            seq = np.zeros([self.seq_per_img, self.seq_length], dtype='int')
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                # seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                seq[q, :] = self.h5_label_file.root.labels[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            # seq = self.h5_label_file['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]
            seq = self.h5_label_file.root.labels[ixl: ixl + self.seq_per_img, :self.seq_length]

        label_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img, 1: self.seq_length + 1] = seq

        # record associated info as well
        info_dict = {}
        info_dict['id'] = self.info['images'][ix]['id']
        info_dict['file_path'] = self.info['images'][ix]['file_path']
        # fetch sen_embed
        if self.opt.sentence_embed:
            # for q in range(self.seq_per_img):
            key = self.id_to_keys[info_dict['id']]
            sen_ix = self.sen_embed_keys.index(key)
            sen_embed = np.stack(self.h5_sen_embed_file['average'][sen_ix, :]).transpose()
            # sen_embed = np.stack(self.h5_sen_embed_file.root.average[sen_ix, :]).transpose()
            sen_embed_batch[i, :len(sen_embed), :] = sen_embed
        infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        if self.opt.sentence_embed:
            data['sen_embed'] = sen_embed_batch

        data['images'] = img_batch
        data['labels'] = label_batch
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix)}
        data['infos'] = infos

        return data"""

    def __call__(self, split):
        return self.get_batch_one(split)

    def get_batch(self, split, batch_size=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size

        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='float32')

        if self.opt.sen_sim_init:
            sim_batch = np.zeros([batch_size * self.seq_per_img, self.opt.sentence_length + 1], dtype='float32')
        if self.opt.word_embed_att:
            related_word_batch = np.zeros([batch_size * self.seq_per_img, self.opt.word_length], dtype='int')
            if self.opt.word_mask:
                word_mask_batch = np.zeros([batch_size * self.seq_per_img, self.opt.word_length], dtype='float32')
            if self.opt.pointer_matching:
                match_label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length, self.opt.match_gold_num], dtype='int')
                match_mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length, self.opt.match_gold_num], dtype='float32')
            if self.opt.index_size != -1:
                retr_w_index_label_batch = np.zeros([batch_size * self.seq_per_img, self.opt.word_length], dtype='int')

        max_index = len(split_ix)
        wrapped = False
        infos = []
        batch_ids = self.shuffle[split][self.iterators[split]: self.iterators[split]+batch_size].tolist()
        self.iterators[split] += batch_size
        if self.iterators[split] >= max_index:
            if split=='train':
                np.random.shuffle(self.shuffle[split])
            self.iterators[split] = 0
            wrapped = True
            if len(batch_ids) != batch_size:
                leftover = batch_size - len(batch_ids)
                batch_ids.extend(self.shuffle[split][self.iterators[split]: self.iterators[split]+leftover])
                self.iterators[split] += leftover
        #combine
        if self.opt.sentence_embed:
            # self.id_to_keys: id to image name(article id)
            # keys = [self.id_to_keys[self.info['images'][i]['id']] for i, b_id in enumerate(batch_ids)]
            # Anwen Hu 2019/08/06
            keys = [self.id_to_keys[split_ix[b_id]] for i, b_id in enumerate(batch_ids)]  # article id
            # print(keys)
            sen_ixs = [self.sen_embed_keys.index(key) for key in keys]
            combined = Parallel(n_jobs=self.num_thread, verbose=0, backend="loky")(
                map(delayed(combine), [(self.opt.input_image_h5, split_ix[b_id], self.opt.sentence_embed, s
                                        ) for b_id, s in zip(batch_ids, sen_ixs)]))
            img_batch = [c[0] for c in combined]
            sen = [c[1] for c in combined]
            if vars(self.opt).get('sentence_embed_method', None) == 'fc' or \
                    vars(self.opt).get('sentence_embed_method', None) == 'fc_max':
                sen_embed_batch = [np.pad(a, ((0, self.opt.sentence_length + 1 - len(a)), (0, 0)),
                                          'constant', constant_values=0) for a in sen]
            else:
                sen_embed_batch = [np.pad(a, ((0, self.opt.sentence_length - len(a)), (0, 0)),
                                          'constant', constant_values=0) if len(a)<self.opt.sentence_length else a[:self.opt.sentence_length] for a in sen]
                sen_embed_batch = np.array(sen_embed_batch, dtype=np.float32)
        else:
            img_batch = [get_img((self.opt.input_image_h5, split_ix[b_id])) for b_id in batch_ids]

        img_batch = np.array(img_batch)

        # batch_ids store the indexes for image id
        for i, b_id in enumerate(batch_ids):
            ix = split_ix[b_id]  # split_ix[] store the image ids for each data split
            ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image, nacp == 1
            assert ncap == 1, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < self.seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')
                for q in range(self.seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.h5_label_file.root.labels[ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                seq = self.h5_label_file.root.labels[ixl: ixl + self.seq_per_img, :self.seq_length]

            label_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = seq
            if self.opt.sen_sim_init:
                sim_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img] = self.sim_sent_file.root.sim_sent_ids[ix1]
            if self.opt.word_embed_att:
                if 'retr' in self.opt.related_w_h5 or 'ran' in self.opt.related_w_h5:
                    related_word_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img] = self.related_word_file.root.retr_word_ids[ix1][:self.max_word_length]
                else:
                    print('invalid w h5')
                    exit(0)
                if self.opt.pointer_matching:
                    match_label_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img] = self.match_file.root.att_index[ix1]  # label start from 1
                    match_mask = [[0.0] * self.opt.match_gold_num] * self.seq_length  # seq_length * gold_num
                    for m, indexes in enumerate(self.match_file.root.att_index[ix1]):
                        for n, index in enumerate(indexes):
                            if index > 0:
                                match_mask[m][n] = 1.0
                    match_mask_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img] = np.array(match_mask)
                if self.opt.index_size != -1:
                    retr_w_index_label_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img] = self.retr_w_index_file.root.retr_w_index[ix1][:self.max_word_length]

            # record associated info as well
            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        # generate word mask
        if self.opt.word_mask:
            word_nonzeros = np.array(list(map(lambda x: (x != 0).sum(), related_word_batch)))
            for ix, row in enumerate(word_mask_batch):
                row[:word_nonzeros[ix]] = 1

        data = {}
        if self.opt.sentence_embed:
            data['sen_embed'] = sen_embed_batch

        data['images'] = img_batch
        data['labels'] = label_batch  # batch*seq_length + 1, the start of caption start for index 1
        data['masks'] = mask_batch

        if self.opt.sen_sim_init:
            data['sim'] = sim_batch  # relevant sentences
        if self.opt.word_embed_att:
            data['sim_words'] = related_word_batch  # words of relevant sentences, for word-level attention
            if self.opt.word_mask:
                data['word_masks'] = word_mask_batch
            if self.opt.pointer_matching:
                data['match_labels'] = match_label_batch
                data['match_masks'] = match_mask_batch
            if self.opt.index_size != -1:
                data['sim_words_index'] = retr_w_index_label_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos
        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0

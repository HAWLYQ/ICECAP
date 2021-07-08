from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
import pickle

import opts
import models
from dataloader import *
# from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

dataset = 'goodnews'  # breakingnews/goodnews
data_dir = './' + dataset + '_data/'
# Input arguments and options
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=dataset, choices=['breakingnews', 'goodnews'])
parser.add_argument('--save_name', type=str, default=dataset, help='the name for saving the output file')
parser.add_argument('--split', type=str, default='test', help='val/test')

# Input paths
parser.add_argument('--model_path', type=str,
                    default='./' + dataset + '_save/show_attend_tell_watt_glove/model-best.pth',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model_path', type=str,
                    default='./' + dataset + '_save/show_attend_tell_watt_glove/model-cnn-best.pth',
                    help='path to cnn model to evaluate')
parser.add_argument('--infos_path', type=str,
                    default='./' + dataset + '_save/show_attend_tell_watt_glove/infos_-best.pkl',
                    help='path to infos to evaluate')

# file paths
parser.add_argument('--input_json', type=str, default=data_dir + dataset + '_cap_basic.json', #data_dir + dataset + '_cap_basic.json'
                    help='basic information, including id2word dictionary and filepath of each image')
parser.add_argument('--input_label_h5', type=str, default=data_dir + dataset + '_cap_label.h5', # data_dir + dataset + '_cap_label.h5'
                    help='h5file containing ground truth for caption generation')
parser.add_argument('--pointer_matching_h5', type=str, default=data_dir + dataset + '_att200_g5_wm_label.h5', # data_dir + dataset + '_att200_g5_wm_label.h5'
                    help='h5file containing ground truth for word-level matching')
parser.add_argument('--sentence_embed', type=str, default=data_dir + dataset + '_articles_full_TBB.h5',
                    help='sentence-level features')
parser.add_argument('--emb_npy', default=data_dir + dataset + '_vocab_emb.npy', # data_dir + dataset + '_vocab_emb.npy'
                    help='initialized embedding file')
parser.add_argument('--related_w_h5', type=str,
                    default=data_dir + dataset + '_retr10_words300_word_ids.h5',  # './data/template_newf_retr10_words300_time_label.h5'
                    help='h5file containing id sequence of retrieved 10 sentences')
parser.add_argument('--retr_w_index_h5', type=str,
                    default=data_dir + dataset + '_retr10_words300_serial_ids.h5', # data_dir + dataset + '_retr10_words300_serial_ids.h5'
                    help='h5file containing serial number of named entities in sentences')
parser.add_argument('--input_image_h5', type=str, default=data_dir + dataset + '_image.h5',  # data_dir + dataset + '_image.h5'
                    help='h5file containing the preprocessed image')


parser.add_argument('--img_init', type=bool, default=True,
                    help='whether to use image feature as init state of decoder')
parser.add_argument('--sen_init', type=bool, default=True,
                    help='whether to use sentence embedding as init state of decoder')
parser.add_argument('--sen_init_type', type=str, default='sum',  # avg/sum
                    help='do sum or average operation to get the global article feature')
parser.add_argument('--sen_sim_init', type=bool, default=False,
                    help='whether to use top k most similar sentence embedding as init state of decoder')


# for word-level attention
parser.add_argument('--word_embed_att', type=bool, default=True,
                    help='Use word-level attention or not')
parser.add_argument('--return_w_attention', type=bool, default=True,
                    help='set this parameter to True when useing word-level attention to choose named entity ')

# for word-level matching
parser.add_argument('--pointer_matching', type=bool, default=False, help='whether to use word-level match')
parser.add_argument('--pointer_matching_weight', type=float, default=0.2, help='weight for word-level matching loss')


parser.add_argument('--word_mask', type=bool, default=False, help='whether to mask the padding in word-level attention')

# sentence-level attention
parser.add_argument('--sentence_embed_att', type=bool, default=False,
                    help='Use sentence-level attention or not')
parser.add_argument('--sentence_embed_method', type=str, default='',  # default is fc
                    help='choose which method to use, available options are conv, conv_deep, fc, bnews, fc_max')

# Basic options
parser.add_argument('--batch_size', type=int, default=64,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=1,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                    help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=0.0,
                    help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',  # /home/abiten/Desktop/Thesis/europana/test/
                    help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                    help='In case the image paths have to be preprended with a root path to an image folder')


parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--return_attention', type=bool, default=False,
                    help='This should only be run when sentence attention architecture is used. When set to True, '
                         'it will write the attention weights for article and images to json')

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = pickle.load(f, encoding='iso-8859-1')

# override and collect parameters
if len(opt.input_label_h5) == 0:
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_image_h5) == 0:
    opt.input_image_h5 = infos['opt'].input_image_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id

ignore = ["id", "batch_size", "beam_size", "start_from", "dataset"]  # , "language_eval"
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            pass
            # assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping
if 'index_size' not in vars(opt):
    opt.index_size = -1
# print('index size:', opt.index_size)
# print('word length', opt.word_length)

# Setup the model
cnn_model = utils.build_cnn(opt)
cnn_model.load_state_dict(torch.load(opt.cnn_model_path))
cnn_model.cuda()
cnn_model.eval()
model = models.setup(opt)
print('load model from', opt.model_path)
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()

if opt.pointer_matching:
    crit = utils.LanguageModelMatchCriterion(opt)
else:
    crit = utils.LanguageModelCriterion()

opt.seq_per_img = 1
# Create the Data Loader instance
loader = DataLoader(opt)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, loader,
                                                            vars(opt), return_attention=opt.return_attention,
                                                            return_w_attention=opt.return_w_attention)

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    print('template captions save to',
          'vis/' + opt.split + '_vis_' + vars(infos['opt'])['caption_model'] + '_' + vars(opt)['save_name'] + '.json')
    json.dump(split_predictions, open('vis/' + opt.split + '_vis_' + vars(infos['opt'])['caption_model'] +
                                      '_' + vars(opt)['save_name'] + '.json', 'w'))

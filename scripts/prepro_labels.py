"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
import tqdm
import math
import stop_words
from difflib import SequenceMatcher

def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def build_vocab_watt(imgs, articles, params):
    """
    add articles to generate vocabulary and move templates to th top of vocabulary
    :param imgs:
    :param articles:
    :param params:
    :return:
    """
    templates = ['ORDINAL_', 'LOC_', 'PRODUCT_', 'NORP_', 'WORK_OF_ART_', 'LANGUAGE_', 'MONEY_',
                 'PERCENT_', 'PERSON_', 'FAC_', 'CARDINAL_', 'GPE_', 'TIME_', 'DATE_', 'ORG_', 'LAW_', 'EVENT_',
                 'QUANTITY_']
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    template_counts = {}
    print('counting words in captions and related sentences...')
    for img in tqdm.tqdm(imgs):
        if 'breakingnews' in params['input_json']:
            article_id = img['imgid'].split('_')[0].replace('n', '').replace('a', '')
        else:
            article_id = img['imgid'].split('_')[0]
        assert len(img['sentences']) == 1
        # captions
        for sent in img['sentences']:
            for w in sent['tokens']:
                if w in templates:
                    template_counts[w] = template_counts.get(w, 0) + 1
                else:
                    counts[w] = counts.get(w, 0) + 1
        # related sentences
        sim_sentences = [id_s[0] for id_s in img['sim_sentences']]
        retr_sentences = img['retrieved_sentences']
        sent_ids = set(sim_sentences+retr_sentences)
        for sent_id in sent_ids:
            sent = articles[article_id]['article_ner'][sent_id]
            for w in sent.split(' '):
                if w.split('-')[0] + '_' in templates:
                    w = w.split('-')[0] + '_'
                    template_counts[w] = template_counts.get(w, 0) + 1
                else:
                    counts[w] = counts.get(w, 0) + 1
    print('vocab size:', len([w for w, n in counts.items() if n > count_thr]))
    """print('counting words in articles...')
    for id, article in tqdm.tqdm(articles.items()):
        for sent in article['article_ner']:
            for w in sent.split(' '):
                if w.split('-')[0]+'_' in templates:
                    w = w.split('-')[0]+'_'
                    template_counts[w] = template_counts.get(w, 0) + 1
                else:
                    counts[w] = counts.get(w, 0) + 1
    print('vocab size:', len([w for w, n in counts.items() if n > count_thr]))"""
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    ctw =sorted([(count, tw) for tw, count in template_counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:10])))
    print('top templates and their counts:')
    print('\n'.join(map(str, ctw[:10])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    template_vocab = [w for w, n in template_counts.items()]  # keep all templates
    print('template size:', len(template_vocab))
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    all_vocab = template_vocab + vocab
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        all_vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr or w in templates else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return all_vocab


def encode_captions(imgs, params, wtoi):
    """
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

    max_length = params['max_length']
    # min_length = params['min_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            # if len(s) <= min_length:
            #   continue
            # else:
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        assert counter-1 == img['cocoid']
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def BM25_score(cap, sent, df_dict, stopwords, dataset):
    """
    calculate BM25 score between caption and sentences in the article
    take into account the named entity rather than the pointer
    remove number and stopwords
    :param cap:
    :param doc:
    :param df_dict:
    :param ave_sen_len:
    :param stopwords:
    :return:
    """
    if dataset == 'breakingnews':
        N = 2423309
        ave_sen_len = 20
    else:
        N = 5953950
        ave_sen_len = 20
    k1 = 2.0
    k2 = 1.0
    b = 0.75
    sent_tf = {}
    cap_tf = {}
    score = 0
    cleaned_cap = []
    # remove number and stop words
    for token in cap:
        token = token.lower()
        if not is_number(token) and token not in stopwords:
            cleaned_cap.append(token)
            cap_tf[token] = cap_tf.get(token, 0) + 1

    for token in sent:
        token = token.lower()
        # ignore number and stop words
        if not is_number(token) and token not in stopwords:
            sent_tf[token] = sent_tf.get(token, 0) + 1
    for token in cleaned_cap:
        df = df_dict.get(token, 0)
        qf = cap_tf[token]
        W = math.log((N - df + 0.5) / (df + 0.5), 2)
        K = k1 * (1 - b + b * len(sent) / ave_sen_len)
        tf = sent_tf.get(token, 0)
        try:
            token_score = round((W * tf * (k1 + 1) / (tf + K)) * (qf * (k2 + 1) / (qf + k2)), 2)
        except TypeError as e:
            # print('token:%s' % token)
            print('W:%.4f, tf:%d, K:%.4f, qf:%d' % (W, tf, K, qf))
            exit(0)
        score = score + token_score
    # sorted_socres = sorted([(index, score) for index, score in scores.items()], reverse=True, key=lambda e: e[1])
    return score

def organize_ner(ner, stopwords):
    new = {}
    for k, pointers in ner.iteritems():  # k: word  v:list(label)
        value = ' '.join(k.split())
        if value not in stopwords:
            value = value.encode('ascii', errors='ignore').decode('ascii')
            for pointer in pointers:
                new[pointer] = value
            """try:
                value.encode('ascii')
            except UnicodeEncodeError as e:
                print value
                exit(0)"""

            # print type(value)
            # exit(0)
    return new


def to_pointers_cap(ent_txt, ent_type, article_ner_pointers, part=True, type_compromise=False):
    if ent_txt in article_ner_pointers.keys():  # fully matched
        for pointer in article_ner_pointers[ent_txt]:
            if ent_type in pointer:
                index = int(pointer.split('-')[1])
                # print(ent_txt, ent_type, 'fully matched: ', pointer, 'index:', index)
                return index
        if type_compromise:
            return int(article_ner_pointers[ent_txt][0].split('-')[1])
        # print(ent_txt, ent_type, article_ner_pointers[ent_txt])
        # txt in, type not in
        # return ent_txt
    if part:  # partly matched
        for key in article_ner_pointers.keys():  # belong
            if ent_txt in key:
                for pointer in article_ner_pointers[key]:
                    if ent_type in pointer:
                        index = int(pointer.split('-')[1])
                        # print(ent_txt, ent_type, 'partly matched:', key, pointer, 'index:', index)
                        return index
                if type_compromise:
                    return int(article_ner_pointers[key][0].split('-')[1])
        scores = []
        for key in article_ner_pointers.keys():
            s = SequenceMatcher(None, ent_txt, key)
            score = s.ratio()
            scores.append((key, score))
        sort_scores = sorted(scores, reverse=True, key=lambda e: e[1])
        if len(sort_scores) > 0:
            best_score = sort_scores[0]
            if best_score[1] > 0.8:  # 0:key, 1:score
                for pointer in article_ner_pointers[best_score[0]]:
                    if ent_type in pointer:
                        index = int(pointer.split('-')[1])
                        # print(ent_txt, ent_type, 'partly matched:',best_score[0], pointer, 'index:', index)
                        return index
                if type_compromise:
                    return int(article_ner_pointers[best_score[0]][0].split('-')[1])
        # print(ent_txt + '\t' + str(article_ner_pointers))
        # print(ent_txt, ent_type, 'not matched index:-1')
        return -1
    else:
        # txt not in
        # print(ent_txt, ent_type, 'not matched index:-1')
        return -1

def encode_related_sentences(imgs, articles, wtoi, params):
    named_entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE',
                      'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'LAW']
    named_entities_ = [name+'_' for name in named_entities]
    # all_sim_sents = []
    all_retr_sents = []
    all_sim_sents = []
    all_sim_words = []
    all_retr_words = []
    all_ran_words = {0:[], 1:[], 2:[], 3:[], 4:[]}

    print('encoding similar sentences...')
    retr_w_num = 0
    retr_unk_num = 0
    if params['name_att']:
        with open('../data/' + params['dataset'] + '_df.json', 'r') as f:
            df_dict = json.load(f)
        stopwords = stop_words.get_stop_words('en')
        candidates_dict = {}
        all_att_indexes = []
    if params['template_index']:
        all_template_index = []
        all_retr_words_index = []
        index_dict = {}
    for i, img in enumerate(tqdm.tqdm(imgs)):
        if params['name_att'] or params['template_index']:
            assert len(img['final_captions']) == 1
            final_caption = img['final_captions'][0]
            full = img['sentences_full'][0]['tokens']
            assert len(final_caption) == len(full)
        sim_sentences = [id_s[0] for id_s in img['sim_sentences']]
        retrieved_sentences = img['retrieved_sentences']
        random_sentences = img['random_sentences']
        if params['order'] == 'time':
            sim_sentences.sort(key=lambda x: x, reverse=False)
            retrieved_sentences.sort(key=lambda x: x, reverse=False)
        if params['retr_s']:
            retrieved_sentences_index = np.zeros(10, dtype='uint32')
            id_index = 0
            for sentence_id in retrieved_sentences:
                if sentence_id <= params['sentence_length']-1:# 0-61/0-53 is allowed
                    retrieved_sentences_index[id_index] = sentence_id+1  # start from 1, 0 for padding
                    id_index += 1
            all_retr_sents.append(retrieved_sentences_index)
        if params['sim_s']:
            sim_sentences_index = np.zeros(10, dtype='uint32')
            id_index = 0
            for sentence_id in sim_sentences:
                if sentence_id <= params['sentence_length'] - 1:  # 0-61/0-53 is allowed
                    sim_sentences_index[id_index] = sentence_id + 1  # start from 1, 0 for padding
                    id_index += 1
                if id_index == 10:
                    break
            all_sim_sents.append(sim_sentences_index)

        if 'breakingnews' in params['input_json']:
            article_id = img['imgid'].split('_')[0].replace('n', '').replace('a', '')
        else:
            article_id = img['imgid'].split('_')[0]
        sentences_str = articles[article_id]['article_ner']
        sim_sentences_str = []
        retrieved_sentences_str = []
        randoms_sentences_str = []
        if params['name_att']:
            scores = []
            raw_ner_dict = articles[article_id]['ner']
            ner_dict = organize_ner(raw_ner_dict, stopwords)
        if params['template_index']:
            raw_ner_dict = articles[article_id]['ner']
        # sentence-level
        # sen_ids = np.zeros(params['max_sent_num']+1, dtype='float32')
        if params['conca_sent']:
            if params['sim_w']:
                for id in sim_sentences:
                    # sen_ids[id] = 1.0
                    sim_sentences_str += sentences_str[id].split(' ')
                sim_sentences_str = sim_sentences_str[:params['max_word_num']]
            if params['retr_w']:
                # print(len(retrieved_sentences))
                for id in retrieved_sentences[:params['related_sent_num']]:
                    if params['name_att']:
                        score = BM25_score(final_caption, sentences_str[id].split(' '), df_dict, stopwords, params['dataset'])
                        scores += [score]*len(sentences_str[id].split(' '))
                    retrieved_sentences_str += sentences_str[id].split(' ')
                retrieved_sentences_str = retrieved_sentences_str[:params['max_word_num']]
            if params['ran_w']:
                for random_item in random_sentences:
                    random_sentences_str = []
                    for id in random_item:
                        random_sentences_str += sentences_str[id].split(' ')
                    randoms_sentences_str.append(random_sentences_str[:params['max_word_num']])

            if params['name_att']:
                scores = scores[:params['att_range']]
                att_indexes = np.zeros(shape=(params['max_length'], params['gold_num']), dtype='uint32') # idnex start from 1, 0 represents no index
                for m, token_c in enumerate(final_caption[:params['max_length']]):
                    if token_c in named_entities_:
                        nametype = token_c
                        name = full[m]
                        candidates = []
                        for n, token_s in enumerate(retrieved_sentences_str[:params['att_range']]):
                            if len(token_s.split('-')) > 1 and token_s.split('-')[0] in named_entities:
                                if token_s in ner_dict:
                                    name_s = ner_dict[token_s]
                                    if (name in name_s or name_s in name) and nametype == token_s.split('-')[0]+'_':
                                        candidates.append((n, name_s, scores[n]))
                        candidates_dict[len(candidates)] = candidates_dict.get(len(candidates), 0) + 1
                        if len(candidates) > 0:
                            candidates.sort(key=lambda x: x[2], reverse=True)
                            # att_index = candidates[0][0] + 1
                            att_index = [0] * params['gold_num']
                            for k, can in enumerate(candidates[:params['gold_num']]):
                                att_index[k] = can[0] + 1
                            att_indexes[m] = att_index

                        # att_weight = np.zeros(params['max_word_num'], dtype='float32')
                        # att_weight[att_index] = 1.0
                all_att_indexes.append(att_indexes)
            if params['template_index']:
                template_indexes = np.zeros(params['max_length'], dtype='uint32') # idnex for template caption, start from 1, 0 represents no index
                retrieved_words_indexes = np.zeros(params['max_word_num'], dtype='uint32') # idnex for retrieved words start from 1, 0 represents no index
                for m, token_c in enumerate(final_caption[:params['max_length']]):
                    if token_c in named_entities_:
                        if token_c == 'WORK_OF_ART_':
                            nametype = 'WORK_OF_ART'
                        else:
                            nametype = token_c.split('_')[0]
                        name = full[m]
                        # index==-1: not find
                        index = to_pointers_cap(name, nametype, raw_ner_dict, part=True, type_compromise=False)
                        index += 1
                        template_indexes[m] = index
                        # print(m)
                        index_dict[index] = index_dict.get(index, 0) + 1
                # print(template_indexes)
                for n, token_s in enumerate(retrieved_sentences_str[:params['max_word_num']]):
                    if len(token_s.split('-')) > 1 and token_s.split('-')[0] in named_entities:
                        index = int(token_s.split('-')[1])
                        index += 1
                        retrieved_words_indexes[n] = index
                        index_dict[index] = index_dict.get(index, 0) + 1
                all_template_index.append(template_indexes)
                all_retr_words_index.append(retrieved_words_indexes)
        # separate encoding
        else:
            for id in sim_sentences:
                # sen_ids[id] = 1.0
                cut_sim_sent = sentences_str[id].split(' ')[:params['single_sent_length']]
                if len(cut_sim_sent) < params['single_sent_length']:
                    cut_sim_sent += ['<PAD>']*(params['single_sent_length']-len(cut_sim_sent))
                sim_sentences_str += cut_sim_sent
            if len(sim_sentences) < params['related_sent_num']:
                sim_sentences_str += ['<PAD>']*params['single_sent_length']*(params['related_sent_num']-len(sim_sentences))
            for id in retrieved_sentences:
                cut_re_sent = sentences_str[id].split(' ')[:params['single_sent_length']]
                if len(cut_re_sent) < params['single_sent_length']:
                    cut_re_sent += ['<PAD>'] * (params['single_sent_length'] - len(cut_re_sent))
                retrieved_sentences_str += cut_re_sent
            if len(retrieved_sentences) < params['related_sent_num']:
                retrieved_sentences_str += ['<PAD>']*params['single_sent_length']*(params['related_sent_num']-len(retrieved_sentences))
            assert len(sim_sentences_str) == params['max_word_num']
            assert len(retrieved_sentences_str) == params['max_word_num']
        # sentence-level
        # all_sim_sents.append(sen_ids)
        # word-level BM25
        if params['sim_w']:
            word_ids = np.zeros(params['max_word_num'], dtype='uint32')
            for j, token in enumerate(sim_sentences_str):
                if len(token.split('-'))>1 and token.split('-')[0] in named_entities:  # replace pointer with template
                    # print(token)
                    token = token.split('-')[0] + '_'
                    # print(token, wtoi[token])
                    # exit(0)
                if token in wtoi:
                    word_ids[j] = wtoi[token]
                elif token == '<PAD>':
                    word_ids[j] = 0
                else:
                    word_ids[j] = wtoi['UNK']
            # word-level image retrieved
            all_sim_words.append(word_ids)
        if params['retr_w']:
            r_word_ids = np.zeros(params['max_word_num'], dtype='uint32')
            for j, token in enumerate(retrieved_sentences_str):
                if len(token.split('-')) > 1 and token.split('-')[0] in named_entities:  # replace pointer with template
                    token = token.split('-')[0] + '_'
                if token in wtoi:
                    retr_w_num += 1
                    r_word_ids[j] = wtoi[token]
                elif token == '<PAD>':
                    r_word_ids[j] = 0
                else:
                    retr_unk_num += 1
                    retr_w_num += 1
                    r_word_ids[j] = wtoi['UNK']
            # print(word_ids)
            # exit(0)
            all_retr_words.append(r_word_ids)
        if params['ran_w']:
            for k, random_sentences_str in enumerate(randoms_sentences_str):
                ran_word_ids = np.zeros(params['max_word_num'], dtype='uint32')
                for j, token in enumerate(random_sentences_str):
                    if len(token.split('-')) > 1 and token.split('-')[0] in named_entities:  # replace pointer with template
                        token = token.split('-')[0] + '_'
                    if token in wtoi:
                        ran_word_ids[j] = wtoi[token]
                    elif token == '<PAD>':
                        ran_word_ids[j] = 0
                    else:
                        ran_word_ids[j] = wtoi['UNK']
                all_ran_words[k].append(ran_word_ids)

    if params['retr_w']:
        print('retr unk %.4f' % (float(retr_unk_num)/retr_w_num))
    if params['conca_sent']:
        if params['sim_w']:
            sim_w_lb_name = params['sim_w_h5'] + str(params['max_word_num'])+'_' + params['order'] + '_label.h5'
        if params['retr_w']:
            retr_w_lb_name = params['retr_w_h5'] + str(params['max_word_num']) + '_' + params['order'] + '_label.h5'
        if params['ran_w']:
            ran_w_lb_name = params['ran_w_h5'] + str(params['max_word_num']) + '_' + params['order'] + '_label.h5'
    else:
        if params['sim_w']:
            sim_w_lb_name = params['sim_w_h5'] + str(params['max_word_num']) + '_' + params['order'] + '_short_label.h5'
        if params['retr_w']:
            retr_w_lb_name = params['retr_w_h5'] + str(params['max_word_num']) + '_' + params['order'] + '_short_label.h5'
    if params['retr_s']:
        retr_s_lb_name = params['retr_s_h5'] + '_' + params['order'] + '_label.h5'
        retr_s_lb = h5py.File(retr_s_lb_name, "w")
        retr_s_lb.create_dataset("retr_sent_ids", dtype='uint32', data=all_retr_sents)
        print('id of relevant snetences encoding done, save to', retr_s_lb_name)
    if params['sim_s']:
        sim_s_lb_name = params['sim_s_h5'] + '_' + params['order'] + '_label.h5'
        sim_s_lb = h5py.File(sim_s_lb_name, "w")
        sim_s_lb.create_dataset("retr_sent_ids", dtype='uint32', data=all_sim_sents)
        print('id of similar snetences encoding done, save to', sim_s_lb_name)
    if params['sim_w']:
        sim_w_lb = h5py.File(sim_w_lb_name, "w")
        sim_w_lb.create_dataset("sim_word_ids", dtype='uint32', data=all_sim_words)
        print('words of sim snetences encoding done, save to', sim_w_lb_name)
    if params['retr_w']:
        retr_w_lb = h5py.File(retr_w_lb_name, "w")
        retr_w_lb.create_dataset("retr_word_ids", dtype='uint32', data=all_retr_words)
        print('words of retrieved snetences encoding done, save to', retr_w_lb_name)
    if params['ran_w']:
        for i in range(5):
            ran_w_lb_name_ = ran_w_lb_name.replace('label', 'label'+str(i))
            ran_w_lb = h5py.File(ran_w_lb_name_, "w")
            ran_w_lb.create_dataset("retr_word_ids", dtype='uint32', data=all_ran_words[i])
            print('words of random snetences encoding done, save to', ran_w_lb_name_)

    if params['name_att']:
        print('candidate', candidates_dict)
        att_index_lb_name = params['att_index_h5'] + str(params['att_range']) + '_g' + str(params['gold_num']) + '_label.h5'
        att_index_lb = h5py.File(att_index_lb_name, "w")
        att_index_lb.create_dataset("att_index", dtype='uint32', data=all_att_indexes)
        print('name attention indexes encoding done, save to', att_index_lb_name)

    if params['template_index']:
        print('max index', max(index_dict.keys()))
        template_index_lb_name = params['template_index_h5'] + '_label.h5'
        retr_w_index_lb_name = params['retr_w_index_h5'] + str(params['max_word_num']) + '_label.h5'
        # template_index_lb = h5py.File(template_index_lb_name, "w")
        # template_index_lb.create_dataset("template_index", dtype='uint32', data=all_template_index)
        retr_w_index_lb = h5py.File(retr_w_index_lb_name, "w")
        retr_w_index_lb.create_dataset("retr_w_index", dtype='uint32', data=all_retr_words_index)
        # print('template indexes encoding done, save to', template_index_lb_name)
        print('word indexes(pointers) of retrieved sentences encoding done, save to', retr_w_index_lb_name)




def main(params, dataset):
    imgs = json.load(open(params['input_json'], 'r'))
    # imgs = imgs['images']
    with open(params['article_json'], 'r') as f:
         article_pointer = json.load(f)

    seed(123)  # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params)
    # vocab = build_vocab_watt(imgs, article_pointer, params)
    print('top 20 in vocab:', vocab[:20])
    exit(0)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    print('PERSON_', wtoi['PERSON_'])

    # encode related sentences
    encode_related_sentences(imgs, article_pointer, wtoi, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    # L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    """N = len(imgs)
    f_lb = h5py.File(params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img:
            if dataset == 'breakingnews':
                jimg['file_path'] = os.path.join(img['filepath'], img['filename'].split('/')[-1])
            else:
                jimg['file_path'] = os.path.join(img['filepath'], img['filename'])  # copy it over, might need
        if 'cocoid' in img: jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    dataset = 'breakingnews'

    if dataset == 'breakingnews':
        prefix = 'breakingnews_'
        parser.add_argument('--sentence_length', type=int, default=62,  # for breakingnews 62
                            help='hyperparameter to pad the values, this is used for both the sentence and word level')
    else:
        prefix = ''
        parser.add_argument('--sentence_length', type=int, default=54,  # for breakingnews 62
                            help='hyperparameter to pad the values, this is used for both the sentence and word level')
    # input json
    # news_dataset
    parser.add_argument('--dataset', default=dataset)
    parser.add_argument('--input_json', default='../data/'+prefix+'ttv_sim_retr20_ran_newf.json', # ttv_sim_retr_ran_newf.json
                        help='input json file to process into hdf5')
    parser.add_argument('--article_json', default='../data/' + prefix + 'article_pointer.json', help='used for encoding related sentences')
    # data_news
    # add '_a' means adding articles to create the vocabulary
    parser.add_argument('--output_json', default='../data/'+prefix+'cap_newf_basic.json', help='output json file')
    # data_news
    parser.add_argument('--output_h5', default='../data/'+prefix+'cap_newf', help='output h5 file')

    # options
    # parser.add_argument('--min_length', default=4, type=int,
    #                     help='min length of a caption, in number of words. captions lesser than this get clipped.')
    parser.add_argument('--max_length', default=31, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')

    parser.add_argument('--word_count_threshold', default=4, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    parser.add_argument('--order', default='time', help='time/score; the order for ranking related sentences')
    parser.add_argument('--retr_s', type=bool, default=False)
    parser.add_argument('--retr_s_h5', default='../data/' + prefix + 'template_newf_retr_sents', help='output h5 file')
    parser.add_argument('--sim_s', type=bool, default=False)
    parser.add_argument('--sim_s_h5', default='../data/' + prefix + 'template_newf_sim_sents', help='output h5 file')
    parser.add_argument('--sim_w', type=bool, default=False)
    parser.add_argument('--sim_w_h5', default='../data/' + prefix + 'template_newf_sim_words', help='output h5 file')
    parser.add_argument('--retr_w', type=bool, default=False)
    parser.add_argument('--retr_w_h5', default='../data/' + prefix + 'template_newf_retr10_words', help='output h5 file')
    parser.add_argument('--ran_w', type=bool, default=False)
    parser.add_argument('--ran_w_h5', default='../data/' + prefix + 'template_newf_ran_words', help='output h5 file')
    parser.add_argument('--max_word_num', default=300, type=int, help='max number of words in similar sentences')
    parser.add_argument('--conca_sent', type=bool, default=True, help='whether to encode the whole sequence or not')
    parser.add_argument('--single_sent_length', type=int, default='30', help='the max length for single sentence')
    parser.add_argument('--related_sent_num', type=int, default='10', help='the max number for related sentence')  # 6, 8, 10, 12, 14

    parser.add_argument('--name_att', type=bool, default=True)
    parser.add_argument('--att_index_h5', default='../data/' + prefix + 'att_index')
    parser.add_argument('--gold_num', default=5)
    parser.add_argument('--att_range', default=200)  # 120(20*6) 160(20*8), 200(20*10),240(20*12), 280(20*14)

    parser.add_argument('--template_index', type=bool, default=True)
    parser.add_argument('--template_index_h5',  default='../data/' + prefix + 'template_index')
    parser.add_argument('--retr_w_index_h5', default='../data/' + prefix + 'retr10_words300_index')





    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params, dataset)

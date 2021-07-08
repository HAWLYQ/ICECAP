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
import tqdm
import math
import stop_words
from difflib import SequenceMatcher
import json


def load_vocab(params):
    vocab_path = '../'+params['dataset']+'_data/'+params['dataset']+'_threshold4_vocab.json'
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    return vocab


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


def organize_ner(ner, stopwords):
    new = {}
    for k, pointers in ner.items():  # k: word  v:list(label)
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
    all_retr_words = []
    all_ran_words = {0:[], 1:[], 2:[], 3:[], 4:[]}

    print('encoding relevant sentences...')
    retr_w_num = 0
    retr_unk_num = 0
    # if params['name_att']:
    with open('../'+params['dataset']+'_data/' + params['dataset'] + '_df.json', 'r') as f:
        df_dict = json.load(f)
    stopwords = stop_words.get_stop_words('en')
    candidates_dict = {}
    all_att_indexes = []
    all_template_index = []
    all_retr_words_index = []
    index_dict = {}
    for i, img in enumerate(tqdm.tqdm(imgs)):
        # if params['name_att'] or params['template_index']:
        assert len(img['final_captions']) == 1
        final_caption = img['final_captions'][0]
        full = img['sentences_full'][0]['tokens']
        assert len(final_caption) == len(full)
        # sim_sentences = [id_s[0] for id_s in img['sim_sentences']]
        retrieved_sentences = img['retrieved_sentences']
        random_sentences = img['random_sentences']
        # rank retrieved sentence according time order
        retrieved_sentences.sort(key=lambda x: x, reverse=False)
        if params['dataset'] == 'breakingnews':
            article_id = img['imgid'].split('_')[0].replace('n', '').replace('a', '')
        else:
            article_id = img['imgid'].split('_')[0]
        sentences_str = articles[article_id]['article_ner']
        retrieved_sentences_str = []
        randoms_sentences_str = []
        # if params['name_att']:
        # store BM25 similar scores between gt caption and sentences, used for choose gt for word-level matching
        scores = []
        raw_ner_dict = articles[article_id]['ner']
        ner_dict = organize_ner(raw_ner_dict, stopwords)
        # if params['template_index']:
        raw_ner_dict = articles[article_id]['ner']

        # collect retrieved sentences
        for id in retrieved_sentences[:params['related_sent_num']]:
            score = BM25_score(final_caption, sentences_str[id].split(' '), df_dict, stopwords, params['dataset'])
            scores += [score]*len(sentences_str[id].split(' '))
            retrieved_sentences_str += sentences_str[id].split(' ')
        retrieved_sentences_str = retrieved_sentences_str[:params['max_word_num']]

        # collect randomly chosen sentences
        if params['ran_w']:
            for random_item in random_sentences:
                random_sentences_str = []
                for id in random_item:
                    random_sentences_str += sentences_str[id].split(' ')
                randoms_sentences_str.append(random_sentences_str[:params['max_word_num']])

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

        all_att_indexes.append(att_indexes)
        # if params['template_index']:
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
        # convert word sequence of retrieved sentences to id sequence
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

    print('retr unk %.4f' % (float(retr_unk_num)/retr_w_num))

    save_dir = '../'+params['dataset']+'_data/'
    # h5 file to store id sequence of concatenated retrieved sentences
    retr_w_lb_name = save_dir + params['dataset'] + '_retr10_words' + str(params['max_word_num']) +  '_word_ids.h5'
    retr_w_lb = h5py.File(retr_w_lb_name, "w")
    retr_w_lb.create_dataset("retr_word_ids", dtype='uint32', data=all_retr_words)
    print('word id of retrieved sentences save to', retr_w_lb_name)
    # print('max index', max(index_dict.keys()))
    # h5 file to store serial number of each named entity
    retr_w_index_lb_name = save_dir + params['dataset'] + '_retr10_words' + str(params['max_word_num']) + '_serial_ids.h5'
    retr_w_index_lb = h5py.File(retr_w_index_lb_name, "w")
    retr_w_index_lb.create_dataset("retr_w_index", dtype='uint32', data=all_retr_words_index)
    print('serial number of named entities in retrieved sentences save to', retr_w_index_lb_name)

    if params['ran_w']:
        # h5 file to store id sequence of concatenated retrieved sentences
        for i in range(5):
            ran_w_lb_name = save_dir + params['dataset'] + '_random_words' + str(params['max_word_num']) + '_word_ids.h5'
            ran_w_lb_name_ = ran_w_lb_name.replace('_word_ids', '_word_ids'+str(i))
            ran_w_lb = h5py.File(ran_w_lb_name_, "w")
            ran_w_lb.create_dataset("retr_word_ids", dtype='uint32', data=all_ran_words[i])
            print('words of random sentences save to', ran_w_lb_name_)

    # print('candidate', candidates_dict)
    # h5 file to store ground truth for word-level matching (raw named att_index)
    word_match_lb_name = save_dir + params['dataset'] + '_att' + str(params['att_range']) + '_g' + str(params['gold_num']) + '_wm_label.h5'
    word_match_lb = h5py.File(word_match_lb_name, "w")
    word_match_lb.create_dataset("att_index", dtype='uint32', data=all_att_indexes)
    print('gt word-level matching save to', word_match_lb_name)


def main(params):
    cap_json_path = '../'+params['dataset']+'_data/'+params['dataset']+'_ttv.json'
    imgs = json.load(open(cap_json_path, 'r'))
    article_json_path = '../'+params['dataset']+'_data/'+params['dataset']+'_article_icecap.json'
    with open(article_json_path, 'r') as f:
         articles = json.load(f)

    seed(123)  # make reproducible
    # create the vocab
    vocab = load_vocab(params)
    itow = {int(i): w for i, w in vocab.items()}  # a 1-indexed vocab translation table
    wtoi = {w: int(i) for i, w in vocab.items()}  # inverse table
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if w in wtoi.keys() else 'UNK' for w in txt]
            img['final_captions'].append(caption)
    # encode retrieved relevant sentences
    encode_related_sentences(imgs, articles, wtoi, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # N = len(imgs)
    # h5 file to store id sequence of gt caption
    cap_h5_path = '../'+params['dataset']+'_data/'+params['dataset']+'_cap_label.h5'
    f_lb = h5py.File(cap_h5_path, "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()
    print('gt caption save to', cap_h5_path)

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img:
            if params['dataset'] == 'breakingnews':
                jimg['file_path'] = os.path.join(img['filepath'], img['filename'].split('/')[-1])
            else:
                jimg['file_path'] = os.path.join(img['filepath'], img['filename'])  # copy it over, might need
        if 'cocoid' in img: jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)
    # store basic information (e.g. id2word dictionary and filepath of each image)
    basic_info_json_path = '../'+params['dataset']+'_data/'+params['dataset']+'_cap_basic.json'
    json.dump(out, open(basic_info_json_path, 'w'))
    print('basic info json save to', basic_info_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breakingnews', choices=['breakingnews', 'goodnews'])
    # some important parameters
    parser.add_argument('--max_length', default=31, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--related_sent_num', type=int, default=10,
                        help='the max number for retrieved sentence')
    parser.add_argument('--max_word_num', default=300, type=int,
                        help='max number of words in concatenated retrieved sentences')
    parser.add_argument('--gold_num', default=5,
                        help='the max number of possible gt for word-level matching')
    parser.add_argument('--att_range', default=200,
                        help='the scope of word-level attention or word-level matching')
    # optional
    parser.add_argument('--ran_w', type=bool, default=False,
                        help='whether to save id sequence of concatenated random sentences')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)

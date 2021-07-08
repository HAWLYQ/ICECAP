import tqdm
import h5py
import spacy
import numpy as np
import json
from collections import Counter
from nltk.tokenize import word_tokenize
import math
import sys


def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_h5(file_save, data):
    f_lb = h5py.File(file_save, "w")
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    ds = f_lb.create_dataset('average', (len(data), 300, ), dtype=dt)
    for i,d in tqdm.tqdm(enumerate(data)):
        ds[i] = d
    f_lb.close()


def getLog(frequency, a=10 ** -3):
    #     return 1/math.log(1.1) if frequency==1 else 1/math.log(frequency)
    return a / (a + math.log(1 + frequency))


def get_vector_avg_weighted_full(sent):
    # sent = nlp(unicode(sent))
    sent = nlp(sent)
    vectors = []
    weights = []
    for token in sent:
        if token.has_vector:
            frequency = count_full.get(token.text, 10)
            weight = getLog(frequency)
            vectors.append(token.vector)
            weights.append(weight)
    try:
        doc_vector = np.average(vectors, weights=weights, axis=0)
    except:
        doc_vector = sent.doc.vector
    return doc_vector


def get_weighted_avg_data(article, get_vector_avg_weighted):
    data, keys = [], []
    for k, v in tqdm.tqdm(article.items()):
        # print v, len(v)
        # Anwen Hu 2019/8/5
        v = v['sentence']
        # print v, len(v)
        if len(v) < sen_len + 1:
            temp = np.zeros([300, len(v)])
            for i, sents in enumerate(v):
                # print sents
                temp[:, i] = get_vector_avg_weighted(sents.lower())
            # exit(0)
        else:
            temp = np.zeros([300, sen_len + 1])
            for i, sents in enumerate(v[:sen_len]):
                temp[:, i] = get_vector_avg_weighted(sents.lower())
            temp[:, sen_len] = np.average([get_vector_avg_weighted(sents.lower()) for sents in v[sen_len:]])
        data.append(temp)
        keys.append(k)

    return keys, data


if __name__ == '__main__':
    dataset = 'breakingnews'  # breakingewns/goodnews
    np.random.seed(42)
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])
    print('nlp model loaded')
    count_full = Counter()
    if dataset == 'breakingnews':
        sen_len = 62
    else:
        sen_len = 54
    article_json_path = '../'+dataset+'_data/'+dataset+'_article.json'
    full = open_json(article_json_path)
    print('loaded', article_json_path)
    full_num = len(full.values())
    tokenized_num = 0
    for v_fu in full.values():
        for elm in v_fu['sentence']:
            count_full.update([t.lower() for t in word_tokenize(elm)])
        tokenized_num += 1
        sys.stdout.write(
            '\r Tokenizing articless: %d/%d articles processed...' % (tokenized_num, full_num))
        sys.stdout.flush()

    keys, data = get_weighted_avg_data(full, get_vector_avg_weighted_full)
    json.dump(keys, open('../'+dataset+'_data/'+dataset+'_articles_full_WeightedAvg_keys.json', 'w', encoding='utf-8'))
    save_h5('../'+dataset+'_data/'+dataset+'_articles_full_WeightedAvg.h5', data)
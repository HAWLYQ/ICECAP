from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import numpy as np
import json
import argparse
import stop_words

import tqdm

def open_json(path):
    print('read from', path)
    with open(path, "r") as f:
        return json.load(f)


def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    final_scores = {}
    all_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
            for m, s in zip(method, scores):
                all_scores[m] = s
        else:
            final_scores[method] = score
            all_scores[method] = scores
        print('%s done' % str(method))

    return final_scores, all_scores


def evaluate(ref, cand, get_scores=True):
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    truth = {}
    for i, caption in enumerate(ref):
        truth[i] = [caption]

    # compute bleu score
    final_scores = score(truth, hypo)

    #     print out scores
    print('Bleu_1:\t ;', final_scores[0]['Bleu_1'])
    print('Bleu_2:\t ;', final_scores[0]['Bleu_2'])
    print('Bleu_3:\t ;', final_scores[0]['Bleu_3'])
    print('Bleu_4:\t ;', final_scores[0]['Bleu_4'])
    print('METEOR:\t ;', final_scores[0]['METEOR'])
    print('ROUGE_L: ;', final_scores[0]['ROUGE_L'])
    print('CIDEr:\t ;', final_scores[0]['CIDEr'])

    if get_scores:
        return final_scores


# Anwen Hu 2019/08/13
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


def insert(cap, word_att, ner_dict, related_words, context=False):
    sen = []
    names = []
    templates = ['ORDINAL_', 'LOC_', 'PRODUCT_', 'NORP_', 'WORK_OF_ART_', 'LANGUAGE_', 'MONEY_',
                   'PERCENT_', 'PERSON_', 'FAC_', 'CARDINAL_', 'GPE_', 'TIME_', 'DATE_', 'ORG_', 'LAW_', 'EVENT_', 'QUANTITY_']
    for i, token in enumerate(cap):
        if token in templates:
            name = ''
            atts = word_att[i]  # word index (sorted by attention weight)
            for j, index in enumerate(atts):
                if context:
                    distance = [0, -1, 1]
                    for d in distance:
                        new_index = index + d
                        if 0 <= new_index < len(related_words):
                            word = related_words[new_index]
                            if len(word.split('-')) > 1 and word.split('-')[0] + '_' == token:
                                name = ner_dict[word]
                                break
                    if name != '':
                        break

                else:
                    try:
                        word = related_words[index]
                    except IndexError as e:
                        # print(index, len(related_words))
                        # exit(0)
                        continue
                    if len(word.split('-')) > 1 and word.split('-')[0]+'_' == token:
                        name = ner_dict[word]
                        break
            if name == '':  # no availale name that has the same entity type as the token
                name = token
            else:
                names.append([name, token[:-1]])  # -1 means to remove the final character '_'
            sen.append(name)
        else:
            sen.append(token)
    return sen, names


def insert_f1(ref_name, hypo_name):
    assert len(ref_name) == len(hypo_name)
    all_predict_num = 0
    all_ground_num = 0
    all_true_num = 0
    for i in range(len(ref_name)):
        ref = ref_name[i]
        hypo = hypo_name[i]
        predict_num = len(hypo)
        ground_num = len(ref)
        true_num = 0
        for item in hypo:
            if item in ref:
                true_num += 1
        all_predict_num += predict_num
        all_true_num += true_num
        all_ground_num += ground_num

    p = round(float(all_true_num) / all_predict_num, 4)
    r = round(float(all_true_num) / all_ground_num, 4)
    f1 = round(2 * p * r / (p + r), 4)
    print("Named Entity Generation p:%f r:%f f1:%f" % (p, r, f1))
    return 0


def main(params):
    word_length = 200
    dataset = params['dataset']
    split = params['split']
    template_path = params['template_path']
    ttv_items = open_json('./'+dataset+'_data/' + dataset + '_ttv.json')
    id_retr = {}
    for item in ttv_items:
        if item['split'] == split:
            if len(item['retrieved_sentences']) > 0:
                retrieved_sentences = item['retrieved_sentences']
                retrieved_sentences.sort(key=lambda x: x, reverse=False)
                id_retr[item['cocoid']] = retrieved_sentences[:10]
            else:
                id_retr[item['cocoid']] = []
    del ttv_items
    test_compact = open_json('./'+dataset+'_data/' + dataset + '_' + split + '.json')
    article_dataset = open_json('./'+dataset+'_data/' + dataset + '_article_icecap.json')
    stopwords = stop_words.get_stop_words('en')

    # Start the insertion process
    opt = parser.parse_args()
    output = open_json(template_path)
    if dataset == 'breakingnews':
        id_to_key = {h['image_id']: h['image_path'].split('/')[1].split('_')[0].replace('n', '').replace('a', '') for h
                     in output}
    else:
        id_to_key = {h['image_id']: h['image_path'].split('/')[1].split('_')[0] for h in output}
    id_to_index = {h['cocoid']: i for i, h in enumerate(test_compact)}
    ref = []
    hypo = []
    ref_name = []
    hypo_name = []
    related_non = 0
    for h in tqdm.tqdm(output):
        imgId = h['image_id']
        cap = h['caption'].split(' ')
        key = id_to_key[imgId]
        index = id_to_index[imgId]
        ref_name.append(test_compact[index]['sentences'][0]['names'])
        ref.append(test_compact[index]['sentences_full'][0]['raw'])
        related_sentences = id_retr[imgId]
        ner_articles = article_dataset[key]['article_ner']
        ner_dict = article_dataset[key]['ner']
        ner_dict = organize_ner(ner_dict, stopwords)
        related_words = []
        if len(related_sentences) > 0:
            for id in related_sentences:
                related_words += ner_articles[id].split(' ')
            related_words = related_words[:word_length]
            sorted_word_att_locs = h['sorted_word_att']
            top_att_weights = h['top_att_weight']
            for top_att_weight in top_att_weights:
                try:
                    assert np.sum(top_att_weight) <= 1.01
                except AssertionError:
                    print(np.sum(top_att_weight), top_att_weight)
                    exit(0)
            assert len(sorted_word_att_locs) == len(top_att_weights)
            sen, inserted_name = insert(cap, sorted_word_att_locs, ner_dict, related_words, context=False)
        else:
            inserted_name = []
            related_non += 1
            sen = cap
        hypo.append(' '.join(sen))
        hypo_name.append(inserted_name)
    # retrieve the reference sentences
    if params['dump']:
        json.dump(hypo, open(template_path.replace('.json', '_full.json'), 'w', encoding='utf-8'))
    insert_f1(ref_name, hypo_name)
    sc, scs = evaluate(ref, hypo)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='goodnews', choices=['breakingnews', 'goodnews'])
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    parser.add_argument('--template_path', type=str,
                        default='./vis/test_vis_show_attend_tell_watt_glove_goodnews.json',
                        help='template path to insert named entities according word-level matching distribution')
    parser.add_argument('--dump', type=bool, default=False, help='Save the inserted captions in a json file')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)


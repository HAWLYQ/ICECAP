import json
import tables
from tqdm import tqdm


def clean_newf(prefix='breakingewns_'):
    for split in ['val', 'test', 'ttv_sim_retr20_ran']:
        input_json = '/data6/haw/GoodNews-master/data/'+prefix+split+'_newf.json'
        raw = json.load(open(input_json, 'r', encoding='utf-8'))
        for item in raw:
            item.pop('sim_sentences')
            """if 'random_sentences' in item.keys():
                item.pop('random_sentences')"""
        if split == 'ttv_sim_retr20_ran':
            split = 'ttv'
        if prefix == 'breakingnews_':
            json.dump(raw, open('../breakingnews_data/'+prefix+split+'.json', 'w', encoding='UTF-8'))
        else:
            json.dump(raw, open('../goodnews_data/goodnews_' + split + '.json', 'w', encoding='UTF-8'))


def add_en(dataset='breakingnews_'):
    for split in ['val', 'test']:
        if dataset == 'breakingnews':
            en_json = '/data6/haw/GoodNews-master/data/'+dataset+'_'+split+'_en.json'
        else:
            en_json = '/data6/haw/GoodNews-master/data/' + split + '_en.json'
        imgs_en = json.load(open(en_json, 'r', encoding='utf-8'))
        img_sentences = {}
        for img in imgs_en:
            # print(img)
            imgid = img['imgid']
            sentences = img['sentences'] # contain names
            img_sentences[imgid] = sentences
        img_json = '../'+dataset+'_data/'+dataset+'_'+split+'.json'
        imgs = json.load(open(img_json, 'r', encoding='utf-8'))
        for img in imgs:
            img['sentences'] = img_sentences[img['imgid']]
        json.dump(imgs, open('../'+dataset+'_data/'+dataset+'_'+split+'.json', 'w', encoding='UTF-8'))


def compare_retr_words():
    old_path = '../data/cap_newf_basic.json'
    new_path = old_path# '../goodnews_data/goodnews_cap_basic.json'
    old_data = json.load(open(old_path, 'r', encoding='utf-8'))
    new_data = json.load(open(new_path, 'r', encoding='utf-8'))
    old_dict = old_data['ix_to_word']
    new_dict = new_data['ix_to_word']
    old_dict['0'] = '<PAD>'
    new_dict['0'] = '<PAD>'
    old_path = '../data/template_newf_retr10_words300_time_label.h5'
    new_path = '../goodnews_data/goodnews_retr10_words300_word_ids.h5'
    old_h5file = tables.open_file(old_path, driver='H5FD_CORE')
    new_h5file = tables.open_file(new_path, driver='H5FD_CORE')
    old_data = old_h5file.root.retr_word_ids
    new_data = new_h5file.root.retr_word_ids
    assert len(old_data) == len(new_data)
    for i in tqdm(range(len(old_data))):
        try:
            old_seq = old_data[i]
            new_seq = new_data[i]
            old_words = [old_dict[str(w_id)] for w_id in old_seq]
            new_words = [new_dict[str(w_id)] for w_id in new_seq]
            for j in range(len(old_seq)):
                assert old_words[j] == new_words[j]
        except AssertionError as e:
            print(i)
            print('old: ', old_data[i])
            print('new: ', new_data[i])
            exit(0)

def compare_basic():
    old_path = '../data/cap_newf_basic.json'
    new_path = '../goodnews_data/goodnews_cap_basic.json'
    old_data = json.load(open(old_path, 'r', encoding='utf-8'))
    new_data = json.load(open(new_path, 'r', encoding='utf-8'))
    old_dict = old_data['ix_to_word']
    new_dict = new_data['ix_to_word']
    for i in range(10):
        i += 1
        print(i, old_dict[str(i)], new_dict[str(i)])


def save_vocab():
    old_path = '../data/breakingnews_cap_newf_basic.json'
    old_data = json.load(open(old_path, 'r', encoding='utf-8'))
    old_dict = old_data['ix_to_word']
    json.dump(old_dict, open('../breakingnews_data/breakingnews_threshold4_vocab.json', 'w', encoding='utf-8'))


if __name__ == '__main__':
    # clean_newf(prefix='')
    # add_en(dataset='goodnews')
    # compare_retr_words()
    # compare_basic()
    save_vocab()



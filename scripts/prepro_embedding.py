import argparse
import json
import numpy as np
import tqdm


def main(params):
    input_json_path = '../'+params['dataset']+'_data/'+params['dataset']+'_cap_basic.json'
    print('loading json file: ', input_json_path)
    info = json.load(open(input_json_path))
    ix_to_word = info['ix_to_word']
    vocab_size = len(ix_to_word)
    print('vocab size is ', vocab_size)
    embedding_dict = {}
    with open(params['emb_path'], 'r', encoding='utf-8') as file:
        print('loading word embedding from', params['emb_path'])
        for line in tqdm.tqdm(file.readlines()):
            line = line.strip()
            word_emb = line.split()
            # print(line)
            word = word_emb[0]
            embedding = np.array(word_emb[1:], dtype=np.float32)
            embedding_dict[word] = embedding
    print("%s read done" % params['emb_path'])
    # embedding_matrix = np.array()
    # random_scale = np.sqrt(3.0 / dim)
    non_found_num = 0
    embedding_matrix = np.zeros([vocab_size + 1, 300], dtype=np.float32)
    # word index begin with 1
    # embedding_matrix[0] = np.random.uniform(-random_scale,random_scale,[1,dim]).astype(np.float32)
    for index, word in tqdm.tqdm(ix_to_word.items()):
        if word in embedding_dict:
            embedding_matrix[int(index)] = embedding_dict[word]
        # glove embedding is not case-sensitive
        # if words are lowercase, this step can be removed
        elif word.lower() in embedding_dict:
            embedding_matrix[int(index)] = embedding_dict[word.lower()]
        else:
            # print("%s not found in embedding file"%word)
            non_found_num += 1
            # word not in embedding file , use a random vector
            # shape => [dim,] (np will convert [1,dim] to [dim,])
            # embedding_matrix[index] = np.random.uniform(-random_scale, random_scale, [1, dim]).astype(np.float32)
            embedding_matrix[int(index)] = np.random.uniform(-0.1, 0.1, [1, 300])
    print("%d/%d words not found in embedding file" % (non_found_num, vocab_size))
    print('embedding matrix shape', embedding_matrix.shape)

    output_path = '../'+params['dataset']+'_data/'+params['dataset'] + '_vocab_emb.npy'
    np.save(output_path, embedding_matrix)
    print('embedding matrix saved to', output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breakingnews', choices=['breakingnews', 'goodnews'])
    parser.add_argument('--emb_path', default='../data/glove.42B.300d.txt', help='raw embedding file')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)

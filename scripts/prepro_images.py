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
from scipy.misc import imread, imresize
import sys


def main(params, prefix):
    input_json_path = '../'+params['dataset'] + '_data/'+params['dataset'] + '_cap_basic.json'
    imgs = json.load(open(input_json_path, 'r'))
    imgs = imgs['images']
    seed(123)  # make reproducible
    missed_writer = open(prefix + 'missed.txt', 'w')
    missed_num = 0
    # create output h5 file
    N = len(imgs)
    output_h5_path = '../'+params['dataset'] + '_data/'+params['dataset'] + '_image.h5'
    f = h5py.File(output_h5_path, "w")
    dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8')  # space for resized images
    for i, img in enumerate(imgs):
        # load the image
        try:
            I = imread(os.path.join(params['images_root'], params['dataset'] + '_' + img['file_path']))
        except IOError as e:
            missed_num += 1
            # print('missed', missed_num)
            missed_writer.write(img['file_path']+'\n')
            continue

        try:
            Ir = imresize(I, (256, 256))
        except:
            print('failed resizing image %s - see http://git.io/vBIE0' % (img['filepath'],))
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        # if i % 1000 == 0:
        sys.stdout.write('\rprocessing %d/%d (%.2f%% done) missed %d' % (i, N, i * 100.0 / N, missed_num))
        sys.stdout.flush()
    f.close()
    missed_writer.close()
    print('image wrote to ', output_h5_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breakingnews', choices=['breakingnews', 'goodnews'])
    # options
    parser.add_argument('--images_root', default='../',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params, prefix)

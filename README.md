#ICECAP: Information Concentrated Entity-aware Image Captioning (ACM MM2020)

By Anwen Hu, Shizhe Chen, Qin Jin

## Environment
* python 3.6  
* torch 1.4
* spacy 2.2.4
* [pycocoevalcap(python3)](https://github.com/ronghanghu/coco-caption)  
(put directory pycocoevalcap and pycocotools under ICECAP)

## Feature Preparation
###1. prepare article input and ground truth
download json files of [BreakingNews]() and [GoodNews]() and put them in directory ICECAP/$DATASET_data, which should contain follwing json files:
* $DATASET_$SPLIT.json: ground-truth caption ($SPLIT in [ttv, test,val], where 'ttv' means train+test+val)
* $DATASET_article_icecap.json: processed articles for ICECAP
* $DATASET_df.json: df file of each dataset
* $DATASET_article.json: raw articles
* $DATASET_threshold4_vocab.json: vocabulary file
>cd scripts

>python prepro_icecap_input.py --dataset $DATASET

This code will produce following files:
* $DATASET_cap_basic.json: basic information (id2word dictionary and filepath of each image)
* $DATASET_att200_g5_wm_label.h5: ground truth for word-level matching
* $DATASET_cap_label.h5: ground truth for caption generation
* $DATASET_retr10_words300_word_ids.h5: id sequence of retrieved 10 sentences (concatenated according time order)
* $DATASET_retr10_words300_serial_ids.h5: serial number of named entities in sentences


###2. prepare initialized word embeddings
download [glove.42B.300d](https://github.com/stanfordnlp/GloVe), and put it in directory ICECAP/data
> python prepro_embedding.py --dataset $DATASET

This code will produce the following file:
* $DATASET_vocab_emb.py: initialized embedding matrix

###3. prepare sentence-level features

> python prepro_articles_wavg.py

> python prepro_articles_tbb.py

(revise variable 'dataset' in these two python files to choose dataset)

This code will produce the following file :
* $DATASET_articles_full_TBB.h5: sentence-level features (proposed in [GoodNews](https://openaccess.thecvf.com/content_CVPR_2019/papers/Biten_Good_News_Everyone_Context_Driven_Entity-Aware_Captioning_for_News_Images_CVPR_2019_paper.pdf))

###4. prepare image input
download raw images of [BreakingNews](http://www.iri.upc.edu/groups/perception/#BreakingNews) and [GoodNews](https://github.com/furkanbiten/GoodNews)
> cd ../prepocess

> python resize_$DATASET_images.py

> python prepre_images.py --dataset $DATASET

This code will produce the following file:
* $DATASET_image.h5: resized image input

download [ResNet152]() to extract image features during training or inference


## Train
> python train.py 

Parameters are set in opts.py. The default parameters are set for ICECAP with the weight of word-level match setting to 0.2.
## Inference
> python eval.py

This code will generate template captions and calculate the word-level matching distribution. The output will save to $TEMPLATE_FILE_PATH.  

> python insert_by_word_match.py --dataset $DATASET --template_path $TEMPLATE_FILE_PATH 

This code will insert named entities according word-level matching distribution. 

If the model is ICECAP-M (drops the word-level matching), run insert_by_word_att.py to insert named entities as follows:

> python insert_by_word_att.py --dataset $DATASET --template_path $TEMPLATE_FILE_PATH 

## Citation

If you find this code useful for your research, please consider citing:
```bibtex
@inproceedings{DBLP:conf/mm/HuCJ20,
  author    = {Anwen Hu and
               Shizhe Chen and
               Qin Jin},
  title     = {{ICECAP:} Information Concentrated Entity-aware Image Captioning},
  booktitle = {{ACM} Multimedia},
  pages     = {4217--4225},
  publisher = {{ACM}},
  year      = {2020}
}
```


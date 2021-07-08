import argparse
import torchvision

torchvision.models.resnet152()


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    dataset = 'breakingnews'
    retr_num = 10
    data_dir = './' + dataset + '_data/'
    parser.add_argument('--dataset', type=str, default=dataset, choices=['breakingnews', 'goodnews'])

    # file paths
    parser.add_argument('--input_json', type=str, default=data_dir + dataset + '_cap_basic.json',
                        help='basic infomation, including id2word dictionary and filepath of each image')
    parser.add_argument('--input_label_h5', type=str, default=data_dir + dataset + '_cap_label.h5',
                        help='h5file containing ground truth for caption generation')
    parser.add_argument('--pointer_matching_h5', type=str, default=data_dir + dataset + '_att200_g5_wm_label.h5',
                        help='h5file containing ground truth for word-level matching')
    parser.add_argument('--sentence_embed', type=str, default=data_dir + dataset + '_articles_full_TBB.h5',
                        help='sentence-level features')
    parser.add_argument('--emb_npy', default=data_dir + dataset + '_vocab_emb.npy',
                        help='initialized embedding file')
    parser.add_argument('--related_w_h5', type=str,
                        default=data_dir + dataset + '_retr10_words300_word_ids.h5',
                        help='h5file containing id sequence of retrieved 10 sentences')
    parser.add_argument('--retr_w_index_h5', type=str,
                        default=data_dir + dataset + '_retr10_words300_serial_ids.h5',
                        help='h5file containing serial number of named entities in sentences')
    parser.add_argument('--input_image_h5', type=str, default=data_dir + dataset + '_image.h5',
                        help='h5file containing the preprocessed image')

    # `caption_model' decides which model to use and the save directory
    parser.add_argument('--caption_model', type=str, default="ICECAP_matchs02_retr10",
                        help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown, BNmodel, ICECAP*')

    parser.add_argument('--input_encoding_size', type=int, default=300,  # glove:300, random:512, # bpe 1024
                        help='the encoding size (word embedding size) of each token in the vocabulary.')
    parser.add_argument('--sentence_embed_size', type=int, default=300,
                        help='size for sentence embedding')
    parser.add_argument('--word_lstm_use', type=bool, default=True, help='whether to use word-level lstm layer')

    # for init state of decoder
    parser.add_argument('--img_init', type=bool, default=True,
                        help='whether to use image feature as init state of decoder')
    parser.add_argument('--sen_init', type=bool, default=True,
                        help='whether to use global article feature as init state of decoder')
    parser.add_argument('--sen_init_type', type=str, default='sum',  # avg/sum
                        help='whether to use sentence embedding as init state of decoder')
    parser.add_argument('--sen_sim_init', type=bool, default=False,
                        help='whether to use top k most similar sentence embedding as init state of decoder')

    # for word-level attention
    parser.add_argument('--word_embed_att', type=bool, default=True,
                        help='Use word-level attention or not')
    parser.add_argument('--index_size', type=int, default=140,
                        help='used for serial number embedding. Set index_size as -1 to drop serial number embedding')

    # for pointer matching
    parser.add_argument('--pointer_matching', type=bool, default=True, help='whether to match pointer')
    parser.add_argument('--pointer_matching_weight', type=float, default=0.2, help='weight for pointer matching loss')
    parser.add_argument('--match_gold_num', type=int, default=5, help='')
    parser.add_argument('--word_mask', type=bool, default=False,
                        help='whether to mask the padding in word-level attention')

    # for sent-level attention
    parser.add_argument('--sentence_embed_att', type=bool, default=False,
                        help='Use attention or not')
    parser.add_argument('--sentence_embed_method', type=str, default='',  # fc
                        help='choose which method to use, available options are fc_max, conv, conv_deep, fc, bnews, default fc')

    # some personalized parameters
    if dataset == 'breakingnews':
        parser.add_argument('--sentence_length', type=int, default=62,  # for breakingnews 62
                            help='max sentences num for an article')
        parser.add_argument('--word_length', type=int, default=str(retr_num * 20),  # 20*6 20*8, 20*10, 20*12, 20*14
                            help='max word num for an article')
        parser.add_argument('--batch_size', type=int, default=32,  # 64
                            help='minibatch size')
        parser.add_argument('--save_checkpoint_every', type=int, default=2000,  # 2000
                            help='how often to save a model checkpoint (in iterations)?')
    else:
        parser.add_argument('--sentence_length', type=int, default=54,  # for goodnews 54
                            help='max sentences num for an article')
        parser.add_argument('--word_length', type=int, default=str(retr_num * 20),  # 20*6 20*8, 20*10, 20*12, 20*14
                            help='max word num for an article')
        parser.add_argument('--batch_size', type=int, default=32,  # 32
                            help='minibatch size')
        parser.add_argument('--save_checkpoint_every', type=int, default=12000,  # 12000
                            help='how often to save a model checkpoint (in iterations)?')

    # used for continue train a checkpoint
    parser.add_argument('--start_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)

    # the following parameters are same for all models
    # model to extract image features
    parser.add_argument('--cnn_model', type=str, default='resnet152',
                        help='resnet')
    parser.add_argument('--cnn_weight', type=str, default='data/resnet152-b121ed2d.pth',
                        help='path to CNN tf model. Note this MUST be a resnet right now.')

    # Model settings
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='rnn, gru, or lstm')

    parser.add_argument('--att_hid_size', type=int, default=512,
                        help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                        help='2048 for resnet, 512 for vgg')

    parser.add_argument('--max_epochs', type=int, default=25,  # 25
                        help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=5.0,  # 5.,
                        help='clip gradients at this value')
    parser.add_argument('--num_thread', type=int, default=4,
                        help='Number of threads to be used for retrieving the data')
    parser.add_argument('--drop_prob_lm', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--finetune_cnn_after', type=int, default=-1,
                        help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=1,
                        help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001,  # 0.002
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=50,  # 50 for non-pretrained embedding
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,  # 3
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')

    # Optimization: for the CNN
    parser.add_argument('--cnn_optim', type=str, default='adam',
                        help='optimization to use for CNN')
    parser.add_argument('--cnn_optim_alpha', type=float, default=0.8,
                        help='alpha for momentum of CNN')
    parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                        help='beta for momentum of CNN')
    parser.add_argument('--cnn_learning_rate', type=float, default=1e-5,
                        help='learning rate for the CNN')
    parser.add_argument('--cnn_weight_decay', type=float, default=0,
                        help='L2 weight decay just for the CNN')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,  # 5000 # if use flair, set 500
                        help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--checkpoint_path', type=str, default=dataset + '_save/',
                        help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=300,  # 100
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                        help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                        help='if true then use 80k, else use 110k')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args

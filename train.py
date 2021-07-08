# Use tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
from six.moves import cPickle
import pickle
import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils

try:
    import tensorflow as tf

except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


# def unwrap_self(arg, **kwarg):
#     return DataLoader.get_batch_one(*arg, **kwarg)

def train(opt):
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    print('batch size: ', opt.batch_size)
    print('max epochs:', opt.max_epochs)
    print('sent init:', opt.sen_init)
    print('img init', opt.img_init)
    print('similar sentence init', opt.sen_sim_init)
    print('sentence embed att:', opt.sentence_embed_att)
    print('word embed att:', opt.word_embed_att)
    print('word lstm use', opt.word_lstm_use)
    print('index size', opt.index_size)
    print('pointer match', opt.pointer_matching)
    print('pointer match weight', opt.pointer_matching_weight)
    print('word mask', opt.word_mask)
    print('caption model', opt.caption_model)
    print('related sequence length', opt.word_length)
    print('max words length', opt.word_length)
    print('decode layers num', opt.num_layers)

    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    if 'breakingnews' in opt.dataset:
        log_step = 200
    else:
        log_step = 1000
    # for debug purposes
    # a=get_batch_one(opt, [loader.split_ix, loader.shuffle, loader.iterators, loader.label_start_ix, loader.label_end_ix])
    # loader.get_batch('train')
    if not os.path.exists(opt.checkpoint_path + 'tensorboard/'):
        os.makedirs(opt.checkpoint_path + 'tensorboard/')

    else:
        for path in os.listdir(opt.checkpoint_path + 'tensorboard/'):
            os.remove(opt.checkpoint_path + 'tensorboard/' + path)
    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path + 'tensorboard/')
    np.random.seed(42)
    infos = {}
    histories = {}

    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            # infos = cPickle.load(f) # python2
            infos = pickle.load(f)  # python3
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[
                    checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                # histories = cPickle.load(f)
                histories = pickle.load(f)

    iteration = infos.get('iter', 0)
    # iteration = 26540
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    cnn_model = utils.build_cnn(opt)
    cnn_model.cuda()
    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    if opt.pointer_matching:
        crit = utils.LanguageModelMatchCriterion(opt)
    else:
        crit = utils.LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    if opt.finetune_cnn_after != -1:
        # only finetune the layer2 to layer4
        cnn_optimizer = optim.Adam([ \
            {'params': module.parameters()} for module in cnn_model._modules.values()[5:] \
            ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            if os.path.isfile(os.path.join(opt.start_from, 'optimizer-cnn.pth')):
                cnn_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-cnn.pth')))

    best_epoch = 0
    # anwen hu 2019/11/05
    """opt.current_lr = opt.learning_rate
    platform = []"""
    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                # anwen hu 2020/5/15  avoid too small lr
                opt.current_lr = max(opt.learning_rate/10, opt.learning_rate * decay_factor)
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate

            print('current lr', opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            # Update the training stage of cnn
            if opt.finetune_cnn_after == -1 or epoch < opt.finetune_cnn_after:
                for p in cnn_model.parameters():
                    p.requires_grad = False
                cnn_model.eval()
            else:
                for p in cnn_model.parameters():
                    p.requires_grad = True
                # Fix the first few layers:
                for module in cnn_model._modules.values()[:5]:
                    for p in module.parameters():
                        p.requires_grad = False
                cnn_model.train()
            update_lr_flag = False
        # torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        # for validation training change the split to 'val'
        # data = loader.get_batch('val')
        data = loader.get_batch('train')

        data['images'] = utils.prepro_images(data['images'], True)
        # torch.cuda.synchronize()
        if iteration % log_step == 0:
            print('Read data:', time.time() - start)

        # torch.cuda.synchronize()
        start = time.time()
        tmp = [data['images'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats = cnn_model(images).permute(0, 2, 3, 1) # batch * 7 * 7 * 2048
        fc_feats = att_feats.mean(2).mean(1)

        if not opt.use_att:
            att_feats = Variable(torch.FloatTensor(1, 1, 1, 1).cuda())

        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), opt.seq_per_img,) +
                                                    att_feats.size()[1:])).contiguous().view(
            *((att_feats.size(0) * opt.seq_per_img,)
              + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), opt.seq_per_img,) +
                                                  fc_feats.size()[1:])).contiguous().view(
            *((fc_feats.size(0) * opt.seq_per_img,) +
              fc_feats.size()[1:]))
        model.zero_grad()
        optimizer.zero_grad()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            cnn_optimizer.zero_grad()

        if opt.sentence_embed:
            sen_embed = Variable(torch.from_numpy(np.array(data['sen_embed'])).cuda())
            if opt.word_embed_att:
                similar_words = Variable(torch.from_numpy(np.array(data['sim_words']))).cuda()
                if opt.word_mask:
                    word_masks = Variable(torch.from_numpy(np.array(data['word_masks']))).cuda()
                else:
                    word_masks = None
                if opt.pointer_matching:
                    match_labels = Variable(torch.from_numpy(np.array(data['match_labels']))).cuda()
                    match_masks = Variable(torch.from_numpy(np.array(data['match_masks']))).cuda()
                if opt.index_size != -1:
                    similar_words_index = Variable(torch.from_numpy(np.array(data['sim_words_index']))).cuda()
                else:
                    similar_words_index = None
            else:
                similar_words = None
                word_masks = None
                similar_words_index = None
            if opt.sen_sim_init:
                sim_sen = Variable(torch.from_numpy(np.array(data['sim']))).cuda()
            else:
                sim_sen = None
            if opt.pointer_matching:
                out, match_output = model(fc_feats, att_feats, labels, sen_embed, similar_words, word_masks,
                                          sim_sen, opt.sen_init, opt.img_init, None, similar_words_index)
                cap_loss, match_loss = crit(out, labels[:, 1:], masks[:, 1:], match_output, match_labels, match_masks)
                loss = cap_loss + match_loss
            else:
                out = model(fc_feats, att_feats, labels, sen_embed, similar_words, word_masks, sim_sen, opt.sen_init,
                            opt.img_init, None, similar_words_index)
                loss = crit(out, labels[:, 1:], masks[:, 1:])
            # loss += cov
        else:
            if 'show_attend_tell' in opt.caption_model:
                loss = crit(model(fc_feats, att_feats, labels, img_init=opt.img_init), labels[:, 1:], masks[:, 1:])
            else:
                loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
            # - 0.001 * crit(model(torch.zeros(fc_feats.size()).cuda(), torch.zeros(att_feats.size()).cuda(), labels), labels[:,1:], masks[:,1:])
        loss.backward()
        # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            utils.clip_gradient(cnn_optimizer, opt.grad_clip)
            cnn_optimizer.step()
        # train_loss = loss.data[0]
        train_loss = loss.item()
        # torch.cuda.synchronize()

        end = time.time()

        if iteration % log_step == 0:
            if opt.pointer_matching:
                train_loss_cap = cap_loss.item()
                train_loss_match = match_loss.item()
                print(
                    "Step [{}/{}], Epoch [{}/{}],  train_loss(cap) = {:.3f}, train_loss(match) = {:.3f}, time/batch = {:.3f}" \
                    .format((iteration + 1) % int(len(loader) / vars(opt)['batch_size']),
                            int(len(loader) / vars(opt)['batch_size']),
                            epoch, vars(opt)['max_epochs'], train_loss_cap, train_loss_match, end - start))
            else:
                print("Step [{}/{}], Epoch [{}/{}],  train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format((iteration + 1) % int(len(loader) / vars(opt)['batch_size']),
                              int(len(loader) / vars(opt)['batch_size']),
                              epoch, vars(opt)['max_epochs'], train_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:  # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                    best_epoch = epoch
                print('best model in epoch: ', best_epoch)
                if not os.path.exists(opt.checkpoint_path + opt.caption_model):
                    os.makedirs(opt.checkpoint_path + opt.caption_model)
                checkpoint_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'model.pth')
                cnn_checkpoint_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'model-cnn.pth')
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                print("cnn model saved to {}".format(cnn_checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    cnn_optimizer_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'optimizer-cnn.pth')
                    torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path + opt.caption_model, 'infos_' + opt.id + '.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path + opt.caption_model, 'histories_' + opt.id + '.pkl'),
                          'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'model-best.pth')
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'model-cnn-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    print("cnn model saved to {}".format(cnn_checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path + opt.caption_model, 'infos_' + opt.id + '-best.pkl'),
                              'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
train(opt)

# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
import numpy as np


class OldModel(CaptionModel):
    def __init__(self, opt):
        super(OldModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.sen_emb_size = opt.sentence_embed_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.emb_path = opt.emb_npy
        self.index_size = opt.index_size
        self.pointer_matching = opt.pointer_matching
        self.sen_init_type = opt.sen_init_type
        self.word_length = opt.word_length
        self.word_lstm_use = opt.word_lstm_use
        self.word_lstm_input_dim = self.input_encoding_size
        if 'sentence_embed_att' in opt:
            self.sentence_embed_att = opt.sentence_embed_att
        else:
            self.sentence_embed_att = False

        # Anwen Hu 2019/10/29
        self.word_embed_att = opt.word_embed_att

        self.ss_prob = 0.0  # Schedule sampling probability

        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size)  # feature to rnn_size
        self.sen_linear = nn.Linear(self.sen_emb_size, self.num_layers * self.rnn_size)
        self.img_sen_linear = nn.Linear(self.sen_emb_size + self.fc_feat_size, self.num_layers * self.rnn_size)

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        if self.word_embed_att:
            if not self.word_lstm_use:
                self.pretrained_emb_linear = nn.Linear(self.pretrained_emb_size, self.rnn_size)
            else:

                self.word_rnn = nn.LSTM(self.word_lstm_input_dim, int(self.rnn_size / 2), 1, bias=False,
                                        batch_first=True,
                                        dropout=self.drop_prob_lm, bidirectional=True)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        if self.index_size != -1:
            if self.pointer_matching:
                self.index_embed = nn.Embedding(self.index_size, int(self.input_encoding_size))  # 300
            else:
                if self.sentence_embed_att: # TODO:REMOVE THIS PART
                    self.index_embed = nn.Embedding(self.index_size, int(self.input_encoding_size/2))  # 300
                else:
                    self.index_embed = nn.Embedding(self.index_size, int(self.input_encoding_size))  # 300
                self.index_logit = nn.Linear(self.rnn_size, self.index_size)
                self.index_dropout = nn.Dropout(self.drop_prob_lm)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.emb_path != '':  # load glove embedding
            print('init embedding from', self.emb_path)
            self.embed.weight.data.copy_(torch.from_numpy(np.load(self.emb_path)))
        else:
            self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

        if self.index_size != -1:
            if self.pointer_matching:
                index_embedding = np.random.uniform(-initrange, initrange,
                                                    size=[self.index_size, int(self.input_encoding_size)])
                index_embedding[0] = 0
                self.index_embed.weight.data.copy_(torch.from_numpy(index_embedding))
            else:
                if self.sentence_embed_att: # TODO:REMOVE THIS PART
                    index_embedding = np.random.uniform(-initrange, initrange,
                                                        size=[self.index_size, int(self.input_encoding_size/2)])
                else:
                    index_embedding = np.random.uniform(-initrange, initrange,
                                                        size=[self.index_size, int(self.input_encoding_size)])
                index_embedding[0] = 0
                # self.index_embed.weight.data.copy_(torch.from_numpy(index_embedding))
                self.index_embed.weight.data.uniform_(-initrange, initrange)
                self.index_logit.bias.data.fill_(0)
                self.index_logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return (image_map.contiguous(), image_map.contiguous())
        else:
            return image_map

    def init_hidden2(self, sen_embed, sim_sen):
        # print(sen_embed.size())
        # print(sim_sen.size())
        if sim_sen is None:
            # print('init with all sentences')
            init_sen_embed = sen_embed
        else:
            # print('init with top-k most similar sentences')
            sim_sen = sim_sen.type(torch.cuda.DoubleTensor)
            sim_sen = sim_sen.unsqueeze(2).expand_as(sen_embed)
            init_sen_embed = sim_sen * sen_embed
        init_sen_embed = init_sen_embed.type(torch.cuda.FloatTensor)
        init_sen_embed = self.sen_linear(init_sen_embed.sum(dim=1)).view(-1, self.num_layers, self.rnn_size).transpose(
            0, 1)
        if self.rnn_type == 'lstm':
            return (init_sen_embed, init_sen_embed)
        else:
            return init_sen_embed

    def init_hidden3(self, fc_feats, sen_embed, sim_sen):
        if sim_sen is None:
            # print('init with all sentences')
            init_sen_embed = sen_embed
        else:
            # print('init with top-k most similar sentences')
            sim_sen = sim_sen.type(torch.cuda.DoubleTensor)
            sim_sen = sim_sen.unsqueeze(2).expand_as(sen_embed)
            init_sen_embed = sim_sen * sen_embed
        init_sen_embed = init_sen_embed.type(torch.cuda.FloatTensor)
        if self.sen_init_type == 'sum':
            init_fuse = torch.cat((fc_feats, init_sen_embed.sum(dim=1)), 1)
        elif self.sen_init_type == 'avg':
            init_fuse = torch.cat((fc_feats, init_sen_embed.mean(dim=1)), 1)
        else:
            print('invalid sent init type', self.sen_init_type)
            exit(0)
        # print(init_fuse.size())
        init_fuse = self.img_sen_linear(init_fuse).view(-1, self.num_layers, self.rnn_size).transpose(0, 1).contiguous()
        if self.rnn_type == 'lstm':
            return (init_fuse, init_fuse)
        else:
            return init_fuse

    def forward(self, fc_feats, att_feats, seq, sen_embed=None,
                similar_words=None, word_masks=None,
                sim_sen=None, sen_init=False, img_init=False,
                index_seq=None, similar_words_index=None,
                return_attention=False, return_w_attention=False, att_supervise=False):
        batch_size = fc_feats.size(0)
        # print(fc_feats.size())
        # print(img_init, sen_init)
        if img_init and not sen_init:
            # print('init by image')
            state = self.init_hidden(fc_feats)
        elif sen_init and not img_init:
            # print('init by sentences')
            state = self.init_hidden2(sen_embed, sim_sen)
        elif img_init and sen_init:
            # print('init by sentence and image')
            state = self.init_hidden3(fc_feats, sen_embed, sim_sen)
        else:
            print(img_init, sen_init)
            print('no init for decoder')
            exit(0)

        if similar_words is not None:
            word_embed = self.embed(similar_words)  # batch * word_length * input_encoding_size
            nametype_embed = None

            if self.index_size != -1:
                # used for pointer matching model
                # word_index_embed is the index for '1,2,....', used for differ PERSON-1 and PERSON-2
                word_index_embed = self.index_embed(similar_words_index)  # batch * word_length * input_encoding_size/2
            else:
                word_index_embed = None
        else:
            word_embed = None
            nametype_embed = None
            word_index_embed = None

        outputs = []
        if att_supervise:
            att_outputs = []
        if index_seq is not None:
            index_outputs = []
        if self.pointer_matching:
            match_outputs = []
        if return_attention or return_w_attention:
            coverage, cov_loss = torch.Tensor([]).cuda(), torch.zeros(batch_size).cuda()
        if self.word_embed_att:
            if self.word_lstm_use:
                if self.index_size != -1:
                    w_out, _ = self.word_rnn(word_embed + word_index_embed)  #
                else:
                    w_out, _ = self.word_rnn(word_embed)  # batch * word_length * emb_size > batch * word_length * rnn_size
            else:
                w_out = self.pretrained_emb_linear(word_embed.reshape([-1, self.pretrained_emb_size])).reshape([-1, self.word_length, self.rnn_size])  # batch * word_length * 1024 > batch * word_length * 512
        else:
            w_out = None

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                    if index_seq is not None:
                        it_index = index_seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
                    if index_seq is not None:
                        it_index = index_seq[:, i].data.clone()
                        prob_prev = torch.exp(index_outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        it_index.index_copy_(0, sample_ind,
                                             torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        it_index = Variable(it_index, requires_grad=False)
            else:
                it = seq[:, i].clone()
                if index_seq is not None:
                    it_index = index_seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            # anwen hu 2019/11/4
            nametype_xt = None
            xt = self.embed(it)

            # if self.index_size != -1:
            if index_seq is not None:
                # print(it_index)
                xt_index = self.index_embed(it_index)
            else:
                xt_index = None

            if return_attention or return_w_attention:
                output, state, atts = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out, word_masks,
                                                nametype_xt, nametype_embed, xt_index, word_index_embed,
                                                return_attention, return_w_attention)
                atts = torch.from_numpy(atts[1].squeeze(2)).cuda()  # 0:image_attention, 1:sent/word attention
                if i != 0:
                    cov_loss += torch.sum(torch.min(atts, coverage), 1)
                    coverage += atts
                else:
                    coverage = torch.cat((coverage, atts), 0)
            elif att_supervise:
                output, state, word_att = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out, word_masks,
                                                    nametype_xt, nametype_embed, xt_index, word_index_embed,
                                                    att_supervise=True)
                if len(att_outputs) < self.seq_length:
                    att_outputs.append(F.log_softmax(word_att, dim=1))
            elif self.pointer_matching:
                output, state, match_output = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out,
                                                        word_masks,
                                                        nametype_xt, nametype_embed, xt_index, word_index_embed)
                if len(match_outputs) < self.seq_length:
                    match_outputs.append(match_output)
            else:
                output, state = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out, word_masks,
                                          nametype_xt, nametype_embed, xt_index, word_index_embed)

            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)

        if return_attention or return_w_attention:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1), torch.sum(cov_loss) / batch_size
        elif self.pointer_matching:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1), torch.cat([_.unsqueeze(1) for _ in match_outputs], 1)
        else:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}, sen_embed=None, return_attention=False):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()

            state = self.init_hidden(tmp_fc_feats)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
            done_beams = []
            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                if sen_embed is not None:
                    if return_attention:
                        output, state, att = self.core(xt, fc_feats, att_feats, state, sen_embed, return_attention)
                    else:
                        output, state = self.core(xt, fc_feats, att_feats, state, sen_embed)
                else:
                    output, state = self.core(xt, fc_feats, att_feats, state)
                # output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}, sen_embed=None, similar_words=None, word_masks=None, sim_sen=None,
               similar_words_index=None,
               return_attention=False, return_w_attention=False):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sen_init = opt.get('sen_init', False)
        img_init = opt.get('img_init', False)
        # word_embed_att = opt.get('word_embed_att', False)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt, sen_embed, return_attention)

        batch_size = fc_feats.size(0)

        if img_init and not sen_init:
            state = self.init_hidden(fc_feats)
        elif sen_init and not img_init:
            # print('init by similar sentences')
            state = self.init_hidden2(sen_embed, sim_sen)
        elif img_init and sen_init:
            state = self.init_hidden3(fc_feats, sen_embed, sim_sen)
        else:
            print('no init for decoder')
            exit(0)

        if similar_words is not None:
            word_embed = self.embed(similar_words)  # batch * word_length * input_encoding_size
            nametype_embed = None
            if self.index_size != -1:
                word_index_embed = self.index_embed(similar_words_index)  # batch * word_length * input_encoding_size/2
            else:
                word_index_embed = None
        else:
            word_embed = None
            nametype_embed = None
            word_index_embed = None
        if self.word_embed_att:
            if self.word_lstm_use:
                if self.index_size != -1:
                    w_out, _ = self.word_rnn(word_embed + word_index_embed)  #
                else:
                    w_out, _ = self.word_rnn(word_embed)  # batch * word_length * emb_size > batch * word_length * rnn_size
            else:
                w_out = self.pretrained_emb_linear(word_embed.reshape([-1, self.pretrained_emb_size])).reshape([-1, self.word_length, self.rnn_size])
        else:
            w_out=None

        if return_attention or return_w_attention: atts = []

        seq = []
        seqLogprobs = []
        if self.pointer_matching:
            match_seqprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
                it_index = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it,
                                                             requires_grad=False))  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            nametype_xt = None
            xt = self.embed(Variable(it, requires_grad=False))
            xt_index = None
            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                it_index = it_index * unfinished.type_as(it_index)
                seq.append(it)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            if sen_embed is not None or word_embed is not None:
                # sen_embed = self.lda(lda)
                #     output, state = self.core(xt, fc_feats, att_feats, state, sen_embed)
                # elif self.sentence_embed_att:
                #     sen_embed = self.lda(lda)
                if return_attention or return_w_attention:
                    output, state, att = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out, word_masks,
                                                   nametype_xt, nametype_embed, xt_index, word_index_embed,
                                                   return_attention, return_w_attention)
                    atts.append(att)
                elif self.pointer_matching:
                    output, state, match_output = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out,
                                                            word_masks,
                                                            nametype_xt, nametype_embed, xt_index, word_index_embed)
                    match_seqprobs.append(match_output.data.cpu().numpy())
                else:
                    output, state = self.core(xt, fc_feats, att_feats, state, sen_embed, w_out, word_masks,
                                              nametype_xt, nametype_embed, xt_index, word_index_embed)
            else:
                output, state = self.core(xt, fc_feats, att_feats, state)
            # output, state = self.core(xt, fc_feats, att_feats, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))
        if return_attention or return_w_attention:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), atts
        else:
            if self.pointer_matching:
                return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs],
                                                                              1), match_seqprobs
            else:
                return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1),


class BreakingNewsCore(nn.Module):
    def __init__(self, opt):
        super(BreakingNewsCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        # rnn decoder layer
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size , self.rnn_size,
                                                      self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, fc_feats, att_feats, state, sen_embed, w_out, word_masks,
                                          nametype_xt, nametype_embed, xt_index, word_index_embed):
        output, state = self.rnn(xt.unsqueeze(0), state)
        return output.squeeze(0), state


class ShowAttendTellCore(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.index_size = opt.index_size
        self.pointer_matching = opt.pointer_matching
        self.word_mask = opt.word_mask

        # sentence embedding parameters
        self.sentence_embed_method = vars(opt).get('sentence_embed_method', '')
        self.sentence_embed_att = vars(opt).get('sentence_embed_att', False)
        self.sentence_length = vars(opt).get('sentence_length', None)  # max sentence num in an article
        self.sentence_embed_size = vars(opt).get('sentence_embed_size', None)
        self.sentence_embed = vars(opt).get('sentence_embed', False)

        # anwenhu 2019/10/29
        self.word_embed_att = vars(opt).get('word_embed_att', False)
        self.word_length = vars(opt).get('word_length', None)  # max number of words

        if self.sentence_embed_method == 'conv':
            self.sen_conv_ch = 32
            self.ctx2att_sen = []
            self.ctx2att_sen += [utils.LeakyReLUConv2d(1, self.sen_conv_ch, [self.sentence_embed_size, 5], 1, [0, 2])]
            self.ctx2att_sen += [nn.Dropout(self.drop_prob_lm)]
            self.ctx2att_sen = nn.Sequential(*self.ctx2att_sen)
            self.h2att_sen = nn.Linear(self.rnn_size, self.sentence_embed_size)
            self.ch_embed = nn.Sequential(nn.Linear(self.sen_conv_ch, 1),
                                          # nn.ReLU(),
                                          nn.Dropout(self.drop_prob_lm))

        elif self.sentence_embed_method == 'bnews':
            self.sen_conv_ch = 256
            self.ctx2att_sen = []
            self.ctx2att_sen += [nn.Conv2d(1, self.sen_conv_ch, [self.sentence_embed_size, 5], 1, [0, 0])]
            self.ctx2att_sen += [nn.MaxPool2d((1, self.sentence_length - 4), 1)]

            self.ctx2att_sen_lin = []
            self.ctx2att_sen_lin += [nn.Linear(self.sen_conv_ch, 64)]
            self.ctx2att_sen_lin += [nn.ReLU(inplace=True)]
            self.ctx2att_sen_lin += [nn.Dropout(p=0.1)]

            self.ctx2att_sen = nn.Sequential(*self.ctx2att_sen)
            self.ctx2att_sen_lin = nn.Sequential(*self.ctx2att_sen_lin)

        elif self.sentence_embed_method == 'conv_deep':
            self.sen_conv_ch = 128
            self.ctx2att_sen = []
            self.ctx2att_sen += [utils.LeakyReLUConv2d(1, self.sen_conv_ch, [self.sentence_embed_size, 5], 1, [0, 2])]
            self.ctx2att_sen += [utils.INSResBlock(self.sen_conv_ch, self.sen_conv_ch, [1, 5], 1, [0, 2])]
            self.ctx2att_sen += [utils.INSResBlock(self.sen_conv_ch, self.sen_conv_ch, [1, 5], 1, [0, 2])]

            self.ctx2att_sen += [nn.Dropout(self.drop_prob_lm)]
            self.ctx2att_sen = nn.Sequential(*self.ctx2att_sen)
            self.h2att_sen = nn.Linear(self.rnn_size, self.sentence_length)
            self.ch_embed = nn.Sequential(nn.Linear(self.sen_conv_ch, 1),
                                          # nn.ReLU(),
                                          nn.Dropout(self.drop_prob_lm))

        elif self.sentence_embed_method == 'fc' or self.sentence_embed_method == 'fc_max':
            self.sentence_att = nn.Linear(self.sentence_embed_size, self.att_hid_size)
            self.h2att_sen = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net_sen = nn.Linear(self.att_hid_size, 1)

        if self.word_embed_att:
            if self.pointer_matching:
                self.word_match = nn.Linear(self.rnn_size, self.att_hid_size)
                self.h2match_word = nn.Linear(self.rnn_size, self.att_hid_size)
                self.alpha_net_word_match = nn.Linear(self.att_hid_size, 1)

                self.word_att = nn.Linear(self.rnn_size, self.att_hid_size)
                self.h2att_word = nn.Linear(self.rnn_size, self.att_hid_size)
            else:
                self.word_att = nn.Linear(self.rnn_size, self.att_hid_size)
                self.h2att_word = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net_word = nn.Linear(self.att_hid_size, 1)

        # rnn decoder layer
        if self.word_embed_att:
            # input: decoded_word + image attention + word attention
            self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size
                                                          + self.rnn_size, self.rnn_size,
                                                          self.num_layers, bias=False, dropout=self.drop_prob_lm)
        else:
            if self.sentence_embed_att and (self.sentence_embed_method == 'fc' or self.sentence_embed_method == 'fc_max'
                                            or self.sentence_embed_method == 'conv'):
                self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size
                                                              + self.sentence_embed_size, self.rnn_size,
                                                              self.num_layers, bias=False, dropout=self.drop_prob_lm)
            elif self.sentence_embed_method == 'bnews':
                self.rnn = getattr(nn, self.rnn_type.upper())(
                    self.input_encoding_size + self.att_feat_size + 64,
                    self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

            elif self.sentence_embed_method == 'conv_deep':
                self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size
                                                              + self.sen_conv_ch, self.rnn_size,
                                                              self.num_layers, bias=False, dropout=self.drop_prob_lm)
            elif self.sentence_embed and not self.sentence_embed_att:
                self.rnn = getattr(nn, self.rnn_type.upper())(
                    self.input_encoding_size + self.att_feat_size + self.att_hid_size,
                    self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
            else:
                self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size,
                                                              self.rnn_size, self.num_layers, bias=False,
                                                              dropout=self.drop_prob_lm)

        # image attention layer
        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        # else:
        # self.ctx2att = nn.Linear(self.att_feat_size, 1)
        # self.h2att = nn.Linear(self.rnn_size, 1)

    def forward(self, xt, fc_feats, att_feats, state, sen_embed=None,
                w_out=None, word_masks=None, nametype_xt=None, nametype_embed=None,
                xt_index=None, word_index_embed=None,
                return_attention=False, return_w_attention=False, att_supervise=False):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = att_feats.contiguous().view(-1, self.att_feat_size)
        if self.att_hid_size > 0:
            att = self.ctx2att(att)  # (batch * att_size) * att_hid_size
            att = att.view(-1, att_size, self.att_hid_size)  # batch * att_size * att_hid_size
            att_h = self.h2att(state[0][-1])  # batch * att_hid_size
            att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
            dot = att + att_h  # batch * att_size * att_hid_size
            dot = F.tanh(dot)  # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
            dot = self.alpha_net(dot)  # (batch * att_size) * 1
            dot = dot.view(-1, att_size)  # batch * att_size
            weight = F.softmax(dot)
            att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
            att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        else:
            att_res = fc_feats


        if self.sentence_embed_att:
            if self.sentence_embed_method == 'fc' or self.sentence_embed_method == 'fc_max':
                att_size_sen = self.sentence_length + 1  # an extra number for other sentences
                att_sen = sen_embed.view(-1, self.sentence_embed_size).float()
                att_sen = self.sentence_att(att_sen)  # (batch * att_size) * att_hid_size
                att_sen = att_sen.view(-1, att_size_sen, self.att_hid_size)  # batch * att_size * att_hid_size
                att_h_sen = self.h2att_sen(state[0][-1])  # batch * att_hid_size
                att_h_sen = att_h_sen.unsqueeze(1).expand_as(
                    att_sen)  # batch * 1 * att_hide_size > batch * att_size * att_hid_size
                dot = att_sen + att_h_sen  # batch * att_size * att_hid_size
                dot = F.tanh(dot)  # batch * att_size * att_hid_size
                # dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
                dot = self.alpha_net_sen(dot)  # (batch * att_size) * 1
                # dot = dot.view(-1, att_size)  # batch * att_size
                # anwen hu 2019/11/2 : add squeeze and dim
                weight_sen = F.softmax(dot.squeeze(2), dim=1)  # batch * att_size
                # att_feats_sen = att_feats.view(-1, att_size_sen, self.sentence_embed_size)  # batch * att_size * att_feat_size
                if self.sentence_embed_method == 'fc':
                    # permute: batch * att_size * embed_size > batch * embed_size * att_size
                    # bmm > batch * embed_size * 1
                    # squeeze > batch * embed_size
                    att_res_sen = torch.bmm(sen_embed.permute(0, 2, 1).float(), weight_sen.unsqueeze(2)).squeeze(
                        2)  # batch * embed_size
                elif self.sentence_embed_method == 'fc_max':
                    # fancy indexing, we are taking the max of the attention values and choosing the sen_embed index accordingly.
                    att_res_sen = sen_embed[torch.arange(0, sen_embed.size()[0]).long(),
                                  weight_sen.argmax(1).squeeze(1), :]

            elif self.sentence_embed_method == 'conv':
                att_h_sen = self.h2att_sen(state[0][-1])
                sen = sen_embed + att_h_sen.unsqueeze(1)
                sen = sen.permute(0, 2, 1).unsqueeze(1)
                att_sen = self.ctx2att_sen(sen)
                dot = F.tanh(att_sen)
                dot = dot.squeeze(2).permute(0, 2, 1)
                weight_sen = F.softmax(self.ch_embed(dot).squeeze(2))
                att_res_sen = torch.bmm(sen_embed.permute(0, 2, 1), weight_sen.unsqueeze(2))
                att_res_sen = att_res_sen.squeeze(2)

            elif self.sentence_embed_method == 'conv_deep':
                att_h_sen = self.h2att_sen(state[0][-1])
                # sen = sen_embed + att_h_sen.unsqueeze(1)
                # sen = sen.permute(0,2,1).unsqueeze(1)
                att_sen = self.ctx2att_sen(sen_embed.permute(0, 2, 1).unsqueeze(1))
                att_sen_combined = att_h_sen.unsqueeze(1) + att_sen.squeeze(2)
                dot = F.tanh(self.ch_embed(att_sen_combined.permute(0, 2, 1)))
                # dot = dot.squeeze(2).permute(0, 2, 1)
                weight_sen = F.softmax(dot.squeeze(2))
                att_res_sen = torch.bmm(att_sen.squeeze(2), weight_sen.unsqueeze(2))
                att_res_sen = att_res_sen.squeeze(2)
        if self.sentence_embed_method == 'bnews':
            intermediate = self.ctx2att_sen(sen_embed.permute(0, 2, 1).unsqueeze(1))
            final = self.ctx2att_sen_lin(intermediate.squeeze(2).squeeze(2))

        if self.word_embed_att:
            att_size_word = self.word_length
            att_word = self.word_att(w_out.reshape([-1, self.rnn_size])).view(-1, att_size_word, self.att_hid_size)
            att_h_word = self.h2att_word(state[0][-1])  # batch * att_hid_size

            att_h_word = att_h_word.unsqueeze(1).expand_as(att_word)
            dot = self.alpha_net_word(F.tanh(att_word + att_h_word))  # batch * word_length * 1
            # print(dot)

            if self.word_mask:  # Anwen Hu 2019/11/13 add word-level attention mask
                dot = dot.squeeze(2) * word_masks
                # print(word_masks)
                # print(dot)
                # exit(0)
                dot = dot.unsqueeze(2)
            weight_word = F.softmax(dot.squeeze(2), dim=1)  # batch * word_length
            # print(weight_word.squeeze(2).sum(0), weight_word.squeeze(2).sum(0).shape)
            att_res_word = torch.bmm(w_out.permute(0, 2, 1).float(), weight_word.unsqueeze(2)).squeeze(
                2)  # if not roberta_word_use, batch * rnn_size else, batch * 1024
        # RNN decoder
        if self.word_embed_att:
            # print(state[0].is_contiguous())
            # print(state[1].is_contiguous())
            # print(xt.is_contiguous())
            output, state = self.rnn(torch.cat([xt, att_res, att_res_word.float()], 1).unsqueeze(0), state)
        elif self.sentence_embed_method == 'bnews':
            output, state = self.rnn(torch.cat([xt, final, att_res], 1).unsqueeze(0), state)
        elif self.sentence_embed_att and (self.sentence_embed_method == 'conv' or self.sentence_embed_method == 'fc'
                                          or self.sentence_embed_method == 'fc_max'):
            # print(state[0].is_contiguous())
            # print(state[1].is_contiguous())
            # print(xt.is_contiguous())
            output, state = self.rnn(torch.cat([xt, att_res, att_res_sen.float()], 1).unsqueeze(0), state)

        elif sen_embed is not None:
            output, state = self.rnn(torch.cat([xt, sen_embed, att_res], 1).unsqueeze(0), state)
        else:
            output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)

        if return_attention:  # return sentence-level attention weight
            return output.squeeze(0), state, [weight.data.cpu().numpy(), weight_sen.data.cpu().numpy()]
        elif return_w_attention:  # return word-level attention weight
            return output.squeeze(0), state, [weight.data.cpu().numpy(), weight_word.data.cpu().numpy()]
        elif self.pointer_matching:
            match_word = self.word_match(w_out.reshape([-1, self.rnn_size])).view(-1, att_size_word, self.att_hid_size)
            match_h_word = self.h2match_word(state[0][-1])
            match_h_word = match_h_word.unsqueeze(1).expand_as(match_word)
            match_dot = self.alpha_net_word_match(F.tanh(match_word - match_h_word))  # batch * word_length * 1
            match_score = F.log_softmax(match_dot.squeeze(2), dim=1)  # batch * word_length
            return output.squeeze(0), state, match_score
        else:
            return output.squeeze(0), state


class AllImgCore(nn.Module):
    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size

        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size,
                                                      self.rnn_size, self.num_layers, bias=False,
                                                      dropout=self.drop_prob_lm)

    def forward(self, xt, fc_feats, att_feats, state):
        output, state = self.rnn(torch.cat([xt, fc_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state


class ShowAttendTellModel(OldModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)


class BreakingNews(OldModel):
    def __init__(self, opt):
        super(BreakingNews, self).__init__(opt)
        self.core = BreakingNewsCore(opt)


class AllImgModel(OldModel):
    def __init__(self, opt):
        super(AllImgModel, self).__init__(opt)
        self.core = AllImgCore(opt)

import ujson as json
import tensorflow as tf
import numpy as np
import pickle
import logging

from mrc.module.model import Model
from mrc.mrc_call_config import setting
from mrc.mrc_call_util import prepro, convert_tokens, get_logger

logger = get_logger(drop_file=False)

class MRC():

    def __init__(self, max_passage):

        self.batch_size = max_passage

        logger.info('config load')
        self.config = setting()

        logger.info('embedding, dic_file load')
        with open(self.config.word_emb_file, "rb") as fh:
            word_mat = np.array(pickle.load(fh), dtype=np.float32)
        with open(self.config.char_emb_file, "rb") as fh:
            char_mat = np.array(pickle.load(fh), dtype=np.float32)
        with open(self.config.word_dic_file, 'r') as f:
            self.word2idx_dict = json.load(f)
        with open(self.config.char_dic_file, 'r') as f:
            self.char2idx_dict = json.load(f)

        logger.info('model setting')
        self.model = Model(config=self.config, word_mat=word_mat, char_mat=char_mat, batch_size=self.batch_size, call=True,trainable=False)

        logger.info('model load')
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.sess, self.config.api_model)

        self.sess.run(tf.assign(self.model.is_train, tf.constant(False, dtype=tf.bool)))

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    def confidenceScore(self, l1, l2, start):
        start_confidence = l1[start]
        l2 = l2[start:start + self.config.max_token]
        l2_softmax = self.softmax(l2)
        end_confidence = np.max(l2_softmax)
        mul_confidence = start_confidence * end_confidence

        return start_confidence, end_confidence, mul_confidence

    def search(self, question, passage_list, passage_num):
        ### question, and passage_list should be string and NOT Null
        if isinstance(question, str) \
                and question is not None \
                and all([isinstance(passage, str) for passage in passage_list]) \
                and None not in passage_list:

            try :
                pair = [[passage, question] for passage in passage_list]

                # _ = passage_tokens
                eval_, record, _ = prepro(self.config, pair, self.word2idx_dict,
                                                       self.char2idx_dict, self.batch_size) ## context, spans, question / ci, qi, cci, qci

                c, q, ch, qh = record[0], record[1], record[2], record[3]

                yp1, yp2,l1, l2, cm = self.sess.run([self.model.yp1, self.model.yp2,self.model.logits1, self.model.logits2,self.model.c_mask],
                                                           feed_dict={self.model.cp: c, self.model.qp: q, self.model.chp: ch,
                                                                     self.model.qhp: qh, self.model.passage_num : passage_num})

                items = []
                for idx, start in enumerate(yp1):
                    _, _, sec = self.confidenceScore(l1[idx], l2[idx], start)

                    answer, start_idx, end_idx = convert_tokens(
                        eval_[idx], yp1[idx].tolist(), yp2[idx].tolist())

                    item = {'start_idx': start_idx, 'end_idx': end_idx, 'best_span_str': answer,'confidence': sec}
                    items.append(item)

                return {'result_code': 100, 'items': items}

            except Exception as e:
                return {'result_code' : 500, 'errorMessage' : str(e).replace('\n', '=-=')}

        else:
            return {'result_code': 400, 'errorMessage' : "wrong input parameter"}





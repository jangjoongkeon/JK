# -*- coding: utf-8 -*- 

import numpy as np
import mecab
import logging
import datetime

from logging.handlers import RotatingFileHandler
from service_locator import config

Me = mecab.MeCab()

logPath = config.get('log.path')

def get_logger(drop_file):

    stream_handler = logging.StreamHandler()

    if drop_file:
        logger = logging.getLogger('file_logger')

        now_date = datetime.datetime.today().strftime('%Y-%m-%d')
        file_path = logPath + 'mrc_error-' + now_date + '.log'
        file_handler = RotatingFileHandler(file_path, encoding='utf8', delay=True)

        formatter = logging.Formatter(fmt='[%(asctime)s]-[%(name)s]-%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter((formatter))

        logger.addHandler(file_handler)
    else:
        logger = logging.getLogger('info_logger')

    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger

def convert_tokens(eval_, pp1, pp2):

    context = eval_["context"]
    spans = eval_["spans"]

    start_idx = spans[pp1][0]
    end_idx = spans[pp2][1]

    answer = context[start_idx: end_idx+1]

    return answer,start_idx, end_idx ## q_id : 정답 / uuid : 정답

def TestToknizer(r):

    q_split =Me.morphs(r)
    q_pos =Me.pos(r)

    row =[]
    for i in range(len(q_pos)):
        row.append(q_pos[i][0]+'/'+q_pos[i][1])

    return q_split,row


def convert_idx(con, ct):
    current = 0
    spans =[]
    for idx, token in enumerate(ct):
        current = con.find(token,current)
        if current<0:
            raise ("Token cannot be found")
        if idx ==len(ct)-1 :
            spans.append((current, len(con)))
        else : 
            if con[current+len(token)] ==' ':
                spans.append((current,current+len(token)))
            else :
                spans.append((current, current+len(token)-1))

        current +=len(token)
    return spans

def process_file(pair):
    examples = []
    eval_examples = []
    
    for context, question in pair:
        context = context.replace(
            "''", '" ').replace("``", '" ')
        context_tokens, token_pos = TestToknizer(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        
        ques = question.replace(
            "''", '" ').replace("``", '" ')
        ques_tokens,ques_pos = TestToknizer(ques)
        ques_chars = [list(token) for token in ques_tokens]

        example = {"context_tokens": context_tokens, "token_pos": token_pos,"context_chars": context_chars, "ques_tokens": ques_tokens,
                   "ques_pos":ques_pos,"ques_chars": ques_chars,'question' : ques}

        eval_example = {"context": context, "spans": spans,'question' : ques}
        
        examples.append(example)
        eval_examples.append(eval_example)

    return examples, eval_examples,context_tokens


def build_features(config, examples,word2idx_dict, char2idx_dict, max_batch_size, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit #최대 단어 수, test면 1000개 아니면 400개
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit # 최대 글자수 16
    
    context_idxs = np.zeros([max_batch_size, para_limit], dtype=np.int32) ## para 최대길이인 400개의 0 
    context_char_idxs = np.zeros([max_batch_size, para_limit, char_limit], dtype=np.int32) # 400 / 16
    ques_idxs = np.zeros([max_batch_size, ques_limit], dtype=np.int32) # 50
    ques_char_idxs = np.zeros([max_batch_size, ques_limit, char_limit], dtype=np.int32) # 50 / 16    

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for line, example in enumerate(examples):
        for i, token in enumerate(example["context_tokens"]):
            context_idxs[line,i] = _get_word(token) ## Context에 등장하는 모든 단어를 idx 로 대체함

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[line,i] = _get_word(token) ## Q에 등장하는 모든 단어를 idx로 대체함

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[line,i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[line,i, j] = _get_char(char)

        record = (context_idxs,
                  ques_idxs,
                  context_char_idxs,
                  ques_char_idxs)

    return record


def prepro(config,pair, word2idx_dict, char2idx_dict, max_batch_size):
    
    examples, eval, passage_tokens = process_file(pair)
    record = build_features(config, examples, word2idx_dict, char2idx_dict, max_batch_size, is_test=True)

    return eval, record, passage_tokens  ## con, span, qa / ci qi cci qci

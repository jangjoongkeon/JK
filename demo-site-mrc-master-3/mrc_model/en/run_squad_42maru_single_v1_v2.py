# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from en.func2 import softmax_mask, dense, dropout, summ, ptr_net,\
    native_gru, dot_attention, dotsoft, content_prob
import pickle
import collections
import json
import math
import os
import random
import bert.modeling as modeling
import bert.optimization as optimization
import bert.tokenization as tokenization
import six
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "has_train_file", False,
    "If true, use the train file if exists.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "thresh", -5.224823812643687,
    "empirical threshold value")


class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += ", orig_answer_text: [%s]"\
            % (tokenization.printable_text(self.orig_answer_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %d" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 s_mask,
                 e_mask,
                 q_mask,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.s_mask = s_mask
        self.e_mask = e_mask
        self.q_mask = q_mask
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = None
                if is_training:
                    # 트레이닝의 경우, 정답은 반드시 하나!
                    if FLAGS.version_2_with_negative and len(qa["answers"]) != 1:
                        is_impossible = 1
                        answer = qa["plausible_answers"][0]
                    else:
                        answer = qa["answers"][0]
                        is_impossible = 0
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        tokenization.whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                        continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""
    """examples 데이타를 최종 InputFeatures 변환하여 FeatureWriter 이용하여 최종 tf_record 파일로 저장"""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []  # word piece 처리된 토큰에 대해 원래 doc_tokens 에서의 인덱스 매핑 정보
        orig_to_tok_index = []  # doc_tokens에 대해 word piece 처리된 이후 all_doc_tokens 에서 해당 인덱스 매핑 정보
        all_doc_tokens = []  # word piece 처리된 이후 모든 토큰 리스트
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            # tok_end_position 의 경우, 여러 word piece 분리된 경우를 고려하여 해당 위치까지 계산
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            s_mask = []
            e_mask = []
            q_mask = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            s_mask.append(0)
            e_mask.append(0)
            q_mask.append(0)

            for i, token in enumerate(query_tokens):
                tokens.append(token)
                segment_ids.append(0)
                s_mask.append(0)
                e_mask.append(0)
                q_mask.append(1)

            tokens.append("[SEP]")
            segment_ids.append(0)
            s_mask.append(0)
            e_mask.append(0)
            q_mask.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                if is_max_context == False:
                    s_mask.append(0)
                else:
                    s_mask.append(1)
                e_mask.append(1)
                q_mask.append(0)
                token_is_max_context[len(tokens)] = is_max_context

                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            s_mask.append(0)
            e_mask.append(0)
            q_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                s_mask.append(0)
                e_mask.append(0)
                q_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(s_mask) == max_seq_length
            assert len(e_mask) == max_seq_length

            assert len(q_mask) == max_seq_length

            start_position = None
            end_position = None
            is_impossible = None
            if is_training:
                is_impossible = example.is_impossible
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and
                                tok_end_position <= doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                # tok_start_position : example.start_position에 해당되는 all_doc_tokens의 위치
                # tok_start_position - doc_start : doc span 내에서의 정답 상대 위치
                # start_position : tokens 내에서 정답 위치
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

            if example_index < 3:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                tf.logging.info(
                    "s_mask: %s" % " ".join([str(x) for x in s_mask]))
                tf.logging.info(
                    "e_mask: %s" % " ".join([str(x) for x in e_mask]))
                tf.logging.info(
                    "q_mask: %s" % " ".join([str(x) for x in q_mask]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                s_mask=s_mask,
                e_mask=e_mask,
                q_mask=q_mask,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)

            # Run callback
            output_fn(feature)
            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the largest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    # word piece 처리된 경우에 좀 더 정확한 정답 부분 있는지 체크(예. (1895-1943)에서 1895인 경우)
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 s_mask, e_mask, q_mask, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]
    
    # Embedding Question and Passage
    expanded_e_mask = tf.expand_dims(tf.cast(e_mask, tf.float32), axis=-1)
    expanded_e_mask = tf.tile(expanded_e_mask, [1, 1, hidden_size])
    c = final_hidden * expanded_e_mask # shape=[batch, length, hidden_size]
    expanded_q_mask = tf.expand_dims(tf.cast(q_mask, tf.float32), axis=-1)
    expanded_q_mask = tf.tile(expanded_q_mask, [1, 1, hidden_size])
    q = final_hidden * expanded_q_mask # shape=[batch, length, hidden_size]
    d = 75
    
    with tf.variable_scope("pointer_for_softmax"):
        # Make a attended question verctor for initial input vector of decoder.
        init = summ(q, d, mask=q_mask, keep_prob=0.7, is_train=is_training) # shape=[batch, hidden_size]
        pointer = ptr_net(batch=batch_size,
                          hidden=init.get_shape().as_list()[-1],
                          keep_prob=0.7,
                          is_train=is_training)
        start_logits, end_logits = pointer(init, c, d, s_mask, e_mask)
        mult_sof = tf.matmul(tf.expand_dims(tf.nn.softmax(start_logits), axis=2),
                             tf.expand_dims(tf.nn.softmax(end_logits), axis=1))

    with tf.variable_scope("pointer_for_sigmoid"):
        init = summ(q, d, mask=q_mask, is_train=is_training)
        start_logits_sig, end_logits_sig = pointer(init, c, d, s_mask, e_mask, sig=True)
        mult_sig = tf.matmul(tf.expand_dims(tf.sigmoid(start_logits_sig), axis=2),
                             tf.expand_dims(tf.sigmoid(end_logits_sig), axis=1))
        
    # Tokens in is_ans will have the probability that the answer include that token or not.
    is_ans_1 = tf.nn.tanh(dense(c, d, use_bias=False, scope="is_ans_1"))
    is_ans = dense(is_ans_1, 1, use_bias=False, scope="is_ans")
    # Convert the value of tokens in the Question part to -INF
    # because the start point couldn't be at the Question part.
    is_ans = softmax_mask(tf.squeeze(is_ans, [-1]), e_mask) # shape = [batch, leangth]
    
    if not FLAGS.version_2_with_negative:
        return (start_logits, end_logits, start_logits_sig, end_logits_sig,
                mult_sof, mult_sig, is_ans, tf.reduce_max(is_ans, axis=-1))
    else:
        # Tokens in has_ans will have the probability that the question is answerable or not.
        has_ans_1 = tf.nn.tanh(dense(final_hidden, d, use_bias=False, scope="has_ans_1"))
        has_ans = tf.squeeze(dense(has_ans_1, 1, use_bias=False, scope="has_ans"), [-1])
        return (start_logits, end_logits, start_logits_sig, end_logits_sig,
                mult_sof, mult_sig, is_ans, tf.reduce_max(is_ans, axis=-1),
                tf.reduce_max(has_ans, axis=-1))


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        s_mask = features["s_mask"]
        e_mask = features["e_mask"]
        q_mask = features["q_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        all_results = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            s_mask=s_mask,
            e_mask=e_mask,
            q_mask=q_mask,
            use_one_hot_embeddings=use_one_hot_embeddings
         )
        if not FLAGS.version_2_with_negative:
            (start_logits, end_logits, start_logits_sig, end_logits_sig,
             mult_sof, mult_sig,is_ans, max_is_ans) = all_results
        else:
            (start_logits, end_logits, start_logits_sig, end_logits_sig,
             mult_sof, mult_sig,is_ans, max_is_ans, max_has_ans) = all_results
            
        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def _compute_loss(logits, positions):
                '''
                The loss become lower
                if the value of answer token(start position or end postion)
                in logits converges to +INF
                '''
                
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            def _compute_sig_loss(logits, positions, is_answerable):
                '''
                The loss become lower
                Answerable case:
                The value of answer token(start position or end postion) in logits
                converges to +INF and values of non-answer tokens converge to -INF.
                Not answerable case:
                All values of tokens in logits converges to -INF.
                '''
                
                exanded_is_answerable = tf.tile(tf.expand_dims(is_answerable, axis=-1),
                                                [1, seq_length])
                # Answerable case:
                # the value of answer token(start position or end postion) would be 1 and
                # all values of non-answer tokens would be 0.
                # Not answerable case:
                # all values of tokens would be 0.
                one_hot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32) \
                    * exanded_is_answerable # shape = [batch, seq_length]
                   
                loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.stop_gradient(one_hot_positions),
                            logits=logits),
                        axis=-1)
                        )
                return loss

            def _compute_mult_loss(mult_sof, mult_sig,
                                  start_positions, end_positions,
                                  is_answerable):
                one_hot_start_positions = tf.one_hot(
                    start_positions, depth=seq_length, dtype=tf.float32)
                one_hot_end_positions = tf.one_hot(
                    end_positions, depth=seq_length, dtype=tf.float32)
                # The label matrix which has only 1 on [start_position, end_position]
                # and the other values are all 0.
                mult_sof_label = tf.matmul(
                    tf.expand_dims(one_hot_start_positions, axis=2),
                    tf.expand_dims(one_hot_end_positions, axis=1)
                    )
                # The tf.losses.log_loss is almost same with cross-entropy,
                # but it receive a 2-dimensional matrix as an input,
                # whether cross-entropy receive an array.
                # The loss_sof became lower if both of the values,
                # one on start position in start_logits and
                # the other on end postion in end_logits,
                # converge to 1 simultaneously.
                # Also the values on the other postion in logits have to converge to 0
                # to make loss lower.
                loss_sof = tf.losses.log_loss(labels=tf.stop_gradient(mult_sof_label),
                                              predictions=mult_sof)
                
                # Answerable case    : Same with mult_sof_label
                # Not answerable case: All values are 0.
                expanded_is_answerable = tf.tile(tf.expand_dims(
                    tf.expand_dims(is_answerable, axis=-1),
                    axis=-1),
                    [1, seq_length, seq_length])
                mult_sig_label = mult_sof_label * expanded_is_answerable
                # The loss_sig became lower
                # Answerable case:
                # Same with loss_sof case.
                # Not answerable case:
                # All values of tokens in both of logits converge to 0.
                loss_sig = tf.losses.log_loss(labels=tf.stop_gradient(mult_sig_label),
                                              predictions=mult_sig)
                return (loss_sof*0.5) + (loss_sig*1.5)

            def _compute_score_loss(is_ans, start_positions, end_positions, is_answerable):
                '''
                The loss become lower
                Answerable case:
                The values of answer tokens(between start position and end postion) in logits
                converge to +INF and values of non-answer tokens converge to -INF.
                Not answerable case:
                All values of tokens in logits converges to -INF.
                '''
                
                # Answerable case     = [0,...0,1,,...1,0,...0], maksing 1 only for answer tokens.
                # Not answerable case = [0,.................,0], all is 0.
                expanded_is_answerable = tf.tile(tf.expand_dims(is_answerable, axis=-1),
                                                 [1, seq_length])
                answer_mask = tf.sequence_mask(end_positions + 1,
                                               seq_length,
                                               dtype=tf.float32) \
                              - tf.sequence_mask(start_positions,
                                                 seq_length,
                                                 dtype=tf.float32)
                answer_mask *= expanded_is_answerable # shape = [batch, seq_len]
                              
                loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.stop_gradient(answer_mask), logits=is_ans), axis=-1))              
                return loss
            
            def _compute_has_ans_loss(max_has_ans, is_answerable):
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.stop_gradient(is_answerable), logits=max_has_ans))
                return loss
            
            start_positions = features["start_positions"]
            end_positions = features["end_positions"]
            is_impossible = tf.cast(features["is_impossible"], tf.float32)
            is_answerable = 1 - is_impossible

            start_loss = _compute_loss(start_logits, start_positions)
            end_loss = _compute_loss(end_logits, end_positions)
            start_loss_sig = _compute_sig_loss(start_logits_sig,
                                               start_positions,
                                               is_answerable)
            end_loss_sig = _compute_sig_loss(end_logits_sig,
                                             end_positions,
                                             is_answerable)
            mult_loss = _compute_mult_loss(mult_sof, mult_sig,
                                           start_positions, end_positions,
                                           is_answerable)
            ans_loss = _compute_score_loss(is_ans, start_positions, end_positions,
                                           is_answerable)
            # The is_ans_loss become lower
            # Answerable case:
            # max_is_ans converges to +INF.
            # It seems that this loss contribute for making difference
            # between the values of answer tokens(to +INF)
            # and those of non-answer tokens(to -INF) bigger.
            # Not answerable case:
            # All values of tokens in is_ans converge to -INF.
            # (It has same role with compute_score_loss)
            '''
            Improvement Idea
            How about use average of several values of tokens(ex. min_ans_len)?
            I think it can make bigger difference
            between answer part values and non-answer part values.
            '''
            is_ans_loss = _compute_has_ans_loss(max_is_ans, is_answerable)
            if FLAGS.version_2_with_negative:
                # The has_ans_loss become lower
                # Answerable case:
                # max_has_ans converges to +INF.
                # It seems that this loss contribute for making the value of a token
                # related with an answer converge to +INF.
                # Not answerable case:
                # All values of tokens in has_ans converge to -INF.
                '''
                Improvement Idea
                Because we don't need many values,
                1. We can add one FC layer to get one value from has_ans
                rather than take the maximum value.
                2. Or we can use the average value of several large values(ex. min_ans_len)
                '''
                has_ans_loss = _compute_has_ans_loss(max_has_ans, is_answerable)

            '''
            Improvement Idea
            Because "has_ans_max" are only used for
            discrimination that a question is answerable or not,
            we can add some weight between 0 and 1 to is_ans_loss and has_ans_loss
            '''
            total_loss = (start_loss + end_loss) * 0.25 \
                         + (start_loss_sig + end_loss_sig) * 0.75 \
                         + ans_loss \
                         + mult_loss \
                         + is_ans_loss
            if FLAGS.version_2_with_negative:
                total_loss += has_ans_loss
            
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "start_logits_sig": start_logits_sig,
                "end_logits_sig": end_logits_sig,
                "is_ans": is_ans,
                "max_is_ans": max_is_ans,
            }
            if FLAGS.version_2_with_negative:
                predictions["max_has_ans"] = max_has_ans
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "s_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "e_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "q_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        tf.logging.info("input_fn batch_size:{}".format(batch_size))

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits",
                                    "is_ans", "max_is_ans", "max_has_ans",
                                    "start_logits_sig", "end_logits_sig"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_prediction_pb_file,
                      output_nbest_file):
    """Write final predictions to the json file."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing predictions pb to: %s" % (output_prediction_pb_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit",
         "ans_score", "max_is_ans", "max_has_ans"])

    all_predictions = collections.OrderedDict()
    all_predictions_pb = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits + result.start_logits_sig,
                                              n_best_size)
            end_indexes = _get_best_indexes(result.end_logits + result.end_logits_sig,
                                            n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    # if start_index or end_index is not in documnet part, continue.
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    avg_is_ans = sum(result.is_ans[start_index:end_index + 1]) \
                                / (end_index + 1 - start_index)
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            # The ans_score become larger if
                            # the values of answer tokens in is_ans converge to +INF
                            # and the values of answer tokens(start & end) in logits
                            # converge to +INF.
                            # Improvement Idea
                            # The value of xxx_logits_sig is between 0 and 1.
                            # However, All values in is_ans are in range of real number.
                            # I think we'd better take an average value of sigmoided is_ans.
                            # In this case, we might give more weight to avg_is_ans.
                            ans_score=avg_is_ans \
                                     + result.start_logits_sig[start_index] \
                                     + result.end_logits_sig[end_index],
                            max_is_ans=result.max_is_ans,
                            max_has_ans=result.max_has_ans \
                                if FLAGS.version_2_with_negative else -10.0
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit + x.ans_score),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit",
                                "ans_score", "max_is_ans", "max_has_ans"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    ans_score=pred.ans_score,
                    max_is_ans=pred.max_is_ans,
                    max_has_ans=pred.max_has_ans \
                        if FLAGS.version_2_with_negative else -10.0
                )
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0,
                                 ans_score=-10.0, max_is_ans=-10.0, max_has_ans=-10.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit + entry.ans_score)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["ans_score"] = entry.ans_score
            output["max_is_ans"] = entry.max_is_ans
            output["max_has_ans"] = entry.max_has_ans \
                if FLAGS.version_2_with_negative else -10.0
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        
        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            '''
            Improvement Idea
            We can experiment with various approaches
            to take a value from is_ans and has_ans vector.
            For example, FCN, average of values in an answer part, etc.
            But if we change the way, we need to find FLAGS.thresh newly.
            '''
            answerable_score = nbest_json[0]["ans_score"] + nbest_json[0]["max_is_ans"] \
                + nbest_json[0]["max_has_ans"]
            all_predictions[example.qas_id] = '' if -answerable_score > FLAGS.thresh \
                                                 else nbest_json[0]["text"]
            all_predictions_pb[example.qas_id] = -answerable_score
            
        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    if FLAGS.version_2_with_negative:
        with tf.gfile.GFile(output_prediction_pb_file, "w") as writer:
            writer.write(json.dumps(all_predictions_pb, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["s_mask"] = create_int_feature(feature.s_mask)
        features["e_mask"] = create_int_feature(feature.e_mask)
        features["q_mask"] = create_int_feature(feature.q_mask)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            features["is_impossible"] = create_int_feature([feature.is_impossible])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_squad_examples(
            input_file=FLAGS.train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        tf.logging.info("has_train_file:{}, train_file: {}".format(FLAGS.has_train_file, train_file))
        if not FLAGS.has_train_file:
            tf.logging.info("***** Create train.tf_record file *****")
            train_writer = FeatureWriter(
                filename=train_file,
                is_training=True)

            convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_training=True,
                output_fn=train_writer.process_feature)
            train_writer.close()

            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num orig examples = %d", len(train_examples))
            tf.logging.info("  Num split examples = %d", train_writer.num_features)
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            del train_examples

        else:
            tf.logging.info("***** Already have train.tf_record file *****")

        train_input_fn = input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        eval_examples = read_squad_examples(
            input_file=FLAGS.predict_file, is_training=False)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        eval_features_file = os.path.join(FLAGS.output_dir, "eval_features")
        if not os.path.isfile(eval_file):
            tf.logging.info("***** Create eval.tf_record file *****")
            eval_writer = FeatureWriter(
                filename=os.path.join(eval_file),
                is_training=False)
            eval_features = []

            def append_feature(feature):
                eval_features.append(feature)
                eval_writer.process_feature(feature)

            convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=FLAGS.max_seq_length,
                doc_stride=FLAGS.doc_stride,
                max_query_length=FLAGS.max_query_length,
                is_training=False,
                output_fn=append_feature)
            # with open(eval_features_file, 'wb') as fh:
            with tf.gfile.GFile(eval_features_file, "wb") as writer:
                pickle.dump(eval_features, writer)

            eval_writer.close()

            tf.logging.info("***** Running predictions *****")
            tf.logging.info("  Num orig examples = %d", len(eval_examples))
            tf.logging.info("  Num split examples = %d", len(eval_features))
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        else:
            tf.logging.info("***** Already have eval.tf_record file *****")
            # with open(eval_features_file, 'rb') as fh:
            with tf.gfile.GFile(eval_features_file, 'rb') as fh:
                eval_features = pickle.load(fh)
        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            is_ans = [float(x) for x in result["is_ans"].flat]
            max_is_ans = float(result["max_is_ans"])
            if FLAGS.version_2_with_negative:
                max_has_ans = float(result["max_has_ans"])
            else:
                max_has_ans = None
            start_logits_sig = [float(x) for x in result["start_logits_sig"].flat]
            end_logits_sig = [float(x) for x in result["end_logits_sig"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    is_ans=is_ans,
                    max_is_ans=max_is_ans,
                    max_has_ans=max_has_ans,
                    start_logits_sig=start_logits_sig,
                    end_logits_sig=end_logits_sig))

        output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
        output_prediction_pb_file = os.path.join(FLAGS.output_dir, "predictions_pb.json")
        output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_prediction_pb_file,
                          output_nbest_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

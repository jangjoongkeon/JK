import os
import tensorflow as tf

from service_locator import config


def setting():
    flags = tf.flags

    device = config.get('tensorflow.device')

    if device == 'GPU':
        pass
    elif device == 'CPU':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

    target_dir = config.get('tensorflow.target_dir')
    api_model = config.get('tensorflow.api_model')

    word_emb_file = config.get('tensorflow.word_emb_file')
    word_dic_file = config.get('tensorflow.word_dic_file')
    char_emb_file = config.get('tensorflow.char_emb_file')
    char_dic_file = config.get('tensorflow.char_dic_file')

    char_dim = config.get('tensorflow.char_dim')
    fast_dim = config.get('tensorflow.fast_dim')
    para_limit = config.get('tensorflow.para_limit')
    ques_limit = config.get('tensorflow.ques_limit')
    test_para_limit = config.get('tensorflow.test_para_limit')
    test_ques_limit = config.get('tensorflow.test_ques_limit')
    char_limit = config.get('tensorflow.char_limit')
    max_token = config.get('tensorflow.max_token')
    capacity = config.get('tensorflow.capacity')
    num_threads = config.get('tensorfow.num_threads')
    is_bucket = config.get('tensorflow.is_bucket')
    batch_size = config.get('tensorflow.batch_size')
    keep_prob = config.get('tensorflow.keep_prob')
    ptr_keep_prob = config.get('tensorflow.ptr_keep_prob')
    hidden = config.get('tensorflow.hidden')
    char_hidden = config.get('tensorflow.char_hidden')


    flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")

    flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
    flags.DEFINE_string("word_dic_file", word_dic_file, "word to dic")
    flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
    flags.DEFINE_string("char_dic_file", char_dic_file, "char to dic")
    flags.DEFINE_string("api_model", api_model, "model for API")

    flags.DEFINE_integer("char_dim", char_dim, "Embedding dimension for char")

    flags.DEFINE_integer("fast_dim", fast_dim, "Embedding dimension for Fasttext")

    flags.DEFINE_integer("para_limit", para_limit, "Limit length for paragraph")
    flags.DEFINE_integer("ques_limit", ques_limit, "Limit length for question")
    flags.DEFINE_integer("test_para_limit", test_para_limit, "Limit length for paragraph in test file")
    flags.DEFINE_integer("test_ques_limit", test_ques_limit, "Limit length for question in test file")
    flags.DEFINE_integer("char_limit", char_limit, "Limit length for character")
    flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
    flags.DEFINE_integer("char_count_limit", -1, "Min count for char")
    flags.DEFINE_integer("max_token", max_token, "max span token number")

    flags.DEFINE_integer("capacity", capacity, "Batch size of dataset shuffle")
    flags.DEFINE_integer("num_threads", num_threads, "Number of threads in input pipeline")

    if device == 'GPU':
        flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn rnn (should be False for CPU)")
    elif device == 'CPU':
        flags.DEFINE_boolean("use_cudnn", False, "Whether to use cudnn rnn (should be False for CPU)")
    else:
        raise Exception('determine which device you will use CPU or PUG')

    flags.DEFINE_boolean("is_bucket", is_bucket, "build bucket batch iterator or not")

    flags.DEFINE_integer("batch_size", batch_size, "Batch size")
    flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")
    flags.DEFINE_float("keep_prob", keep_prob, "Dropout keep prob in rnn")
    flags.DEFINE_float("ptr_keep_prob", ptr_keep_prob, "Dropout keep prob for pointer network")
    flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
    flags.DEFINE_integer("hidden", hidden, "Hidden size")
    flags.DEFINE_integer("char_hidden", char_hidden, "GRU dimention for char")
    flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")

    return flags.FLAGS

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

INF = 1e30


class cudnn_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(
                1, num_units, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(
                1, num_units, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(
                num_units, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            gru_bw = tf.contrib.rnn.GRUCell(
                num_units, kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            init_fw = gru_fw.zero_state(batch_size, dtype=tf.float32)
            init_bw = gru_bw.zero_state(batch_size, dtype=tf.float32)
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))
            self.keep_prob=keep_prob
            self.is_train=is_train

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]

        return res


'''class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask_a, mask_c, sig = False):
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask_a, mask_c, sig=sig)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask_a, mask_c, sig=sig)
            return logits1, logits2'''
class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask_s, mask_e, sig = False):
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dropout_mask, d, mask_s, sig=sig)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dropout_mask, d, mask_e, sig=sig)
            return logits1, logits2


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        if is_train:
            args = tf.nn.dropout(args, keep_prob, noise_shape=noise_shape) * scale
        else:
            args = args
        '''args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)'''
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val
def softmax_mask_a(val, mask_a, mask_c):
    W1 = tf.get_variable(
            'mask_attention_add',
            shape=(1, ),
            initializer=tf.ones_initializer,
            regularizer=None,
            trainable=True,
        )
    
    return -INF * (1 - tf.cast(mask_c, tf.float32)) -10 *W1* (1 - tf.cast(mask_a, tf.float32)) + val
def softmax_mask_(val, mask_a, mask_c):
    
    
    return -INF * (1 - tf.cast(mask_c, tf.float32)) -10 * (1 - tf.cast(mask_a, tf.float32)) + val
def softmax_mask_p(val, mask_a, mask_c):
    W1 = tf.get_variable(
            'mask_pointer_add',
            shape=(1, ),
            initializer=tf.ones_initializer,
            regularizer=None,
            trainable=True,
        )
    
    return -INF * (1 - tf.cast(mask_c, tf.float32)) -10 *W1* (1 - tf.cast(mask_a, tf.float32)) + val
'''def pointer(inputs, state, hidden, mask_a, mask_c, scope="pointer", sig = False):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask_(tf.squeeze(s, [2]), mask_a, mask_c)
        if sig == True :
            a = tf.expand_dims(tf.nn.sigmoid(s1), axis=2)
        else:
            a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1'''
def pointer(inputs, state, hidden, mask_c, scope="pointer", sig = False):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask_c)
        if sig == True :
            a = tf.expand_dims(tf.nn.sigmoid(s1), axis=2)
        else:
            a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1

def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ", sig = False):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        if sig == True :
            a = tf.expand_dims(tf.nn.sigmoid(s1), axis=2)
        else:
            a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res

def content_prob(inputs, inputs2, mask, hidden, keep_prob=1.0, is_train=None, scope="content_prob"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_inputs2 = dropout(inputs2, keep_prob=keep_prob, is_train=is_train)
        


        with tf.variable_scope("attention"):
            inputs_ = dense(d_inputs, hidden, use_bias=False, scope="inputs")
            inputs_2 = dense(d_inputs2, hidden, use_bias=False, scope="inputs2")
            #inputs_ = tf.squeeze(dense(inputs_, 1, use_bias=False, scope="inputs_dense"), [-1])
            inputs_ = tf.squeeze(dense(dropout(tf.nn.tanh(inputs_+inputs_2), keep_prob=keep_prob, is_train=is_train), 1, use_bias=False, scope="inputs_dense"), [-1])
            
            prob = softmax_mask(inputs_, mask)
            prob2 = tf.reduce_max(prob, axis=1)


            return prob, prob2

def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]
        
        with tf.variable_scope("attention"):
            inputs_ = tf.nn.leaky_relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.leaky_relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))

            outputs_ = tf.matmul(inputs_, tf.transpose(
                memory_, [0, 2, 1])) / (hidden ** 0.5)
            



            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])

            outputs_mask = softmax_mask(outputs_, mask)
            logits = tf.nn.softmax(outputs_mask)
            

            logits2=tf.nn.softmax(tf.reduce_max(outputs_mask, axis=1, keep_dims=True))
           
            
            outputs2= tf.matmul(logits2, memory)
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res*gate, outputs2
def dot_attention_new(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention_new"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]
        
        with tf.variable_scope("attention"):
            inputs_ = tf.nn.leaky_relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs"))
            memory_ = tf.nn.leaky_relu(
                dense(d_memory, hidden, use_bias=False, scope="memory"))


            
            inputs_tile = conv(tf.tile(tf.expand_dims(inputs_, axis=2), [1, 1, tf.shape(memory)[1], 1]), 1)
            memory_tile = conv(tf.tile(tf.expand_dims(memory_, axis=1), [1, JX, 1, 1]), 1)
            outputs_mul = inputs_tile*memory_tile
            outputs_sub = inputs_tile-memory_tile
        with tf.variable_scope("outputs"):
            outputs = tf.concat([inputs_tile, memory_tile, outputs_mul, outputs_sub], axis=-1)
            outputs_ = tf.squeeze(conv(outputs, 1), axis=-1)


            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])

            outputs_mask = softmax_mask(outputs_, mask)
            logits = tf.nn.softmax(outputs_mask)
            

            logits2=tf.nn.softmax(tf.reduce_max(outputs_mask, axis=1, keep_dims=True))
           
            
            outputs2= tf.matmul(logits2, memory)
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res*gate, outputs2
def dot_attention_additive(inputs, memory, past, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention_add"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        d_past = dropout(past, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]
        
        with tf.variable_scope("attention"):
            inputs_ = dense(d_inputs, hidden, use_bias=False, scope="inputs")
            past_ = dense(d_past, hidden, use_bias=False, scope="past")
            memory_ = dense(d_memory, hidden, use_bias=False, scope="memory")

            outputs_ = tf.matmul(tf.nn.tanh(inputs_+past_), tf.transpose(
                memory_, [0, 2, 1])) #/ (hidden ** 0.5)
           
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask(outputs_, mask))
           
            
            outputs = tf.matmul(logits, memory)
           
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res*gate
def dot_attention_(inputs, memory, past, mask_a, mask_c, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        d_past = dropout(past, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]
        
        with tf.variable_scope("attention"):
            inputs_ = dense(d_inputs, hidden, use_bias=False, scope="inputs")
            memory_ = dense(d_memory, hidden, use_bias=False, scope="memory")
            past_ = dense(d_past, hidden, use_bias=False, scope="past")
            outputs_ = tf.matmul(tf.nn.tanh(inputs_+past_), tf.transpose(
                memory_, [0, 2, 1])) #/ (hidden ** 0.5)
           
            mask_a = tf.tile(tf.expand_dims(mask_a, axis=1), [1, JX, 1])
            mask_c = tf.tile(tf.expand_dims(mask_c, axis=1), [1, JX, 1])
            logits = tf.nn.softmax(softmax_mask_(outputs_, mask_a, mask_c))
            
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res*gate

def dotsoft(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dotsoft"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.tanh(
                dense(d_inputs, memory.get_shape().as_list()[2], use_bias=False, scope="inputs"))
            '''memory_ = tf.nn.tanh(
                dense(d_memory, hidden, use_bias=False, scope="memory"))''' #fix
            outputs_ = tf.matmul(inputs_, tf.transpose(
                d_memory, [0, 2, 1])) #/ ((4*hidden) ** 0.5)  #fix
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])  #fix two lines
            logits = tf.squeeze(softmax_mask(outputs_, mask), axis=1)
            #return logits, outputs_
            #outputs = tf.matmul(logits, memory)
            '''logits2 = tf.nn.softmax(softmax_mask(outputs, mask))
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))'''
            return logits
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
initializer_x = tf.contrib.layers.xavier_initializer()
def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden], regularizer=regularizer, initializer=initializer_x)
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], regularizer=regularizer, initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res

initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "SAME")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs
def conv_JE(inputs, bias = None, activation = None, filter_shape = [1, 1, 1], kernel_size=1, name = "conv", reuse = None, valid="SAME"):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            #filter_shape = [kernel_size,shapes[-1],output_size]
            #bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, valid)
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs
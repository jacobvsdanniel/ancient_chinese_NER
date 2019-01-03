import sys

import numpy as np
import tensorflow as tf

class Config(object):
    """Store hyper parameters for the model."""
    
    def __init__(self):
        self.name = "YOLO"
        
        self.character_to_index = {"c": 0, "h": 1, "UNK": 2}
        self.alphabet_size = len(self.character_to_index)
        self.character_d = 300
        
        self.hidden_layers = 2
        self.hidden_d = 100
        
        self.output_d_list = [100,2]
        
        self.optimizer = "nadam"
        self.gradient_threshold = 1
        self.keep_rate = 0.65
        return
        
class Model(object):
    """A model with input, output, forward inference, backward update.
    
    Instantiating an object of this class only defines a Tensorflow computation graph
    under the name scope config.name. Weights of a model instance reside in a Tensorflow session.
    """
    
    def __init__(self, config):
        """Contruct Tensowflow graph."""
        
        self.create_hyper_parameter(config)
        self.create_input()
        self.create_hidden()
        self.create_output()
        self.create_update_op()
        return
        
    def create_hyper_parameter(self, config):
        """Add attributes of cofig to self."""
        
        for parameter in dir(config):
            if parameter[0] == "_": continue
            setattr(self, parameter, getattr(config, parameter))
        return
        
    def create_input(self):
        """Construct the input layer."""
        
        # Input placeholders
        #                                   [nodes, samples]
        self.y  = tf.placeholder(  tf.int32, [      None])
        self.sf = tf.placeholder(tf.float32, [      None, 1])
        self.cl = tf.placeholder(  tf.int32, [      None])
        self.ci = tf.placeholder(  tf.int32, [None, None])
        self.cf = tf.placeholder(tf.float32, [None, None, 2])
        
        # Compute input dimensions
        self.nodes = tf.shape(self.ci)[0]
        self.samples = tf.shape(self.ci)[1]
        
        # Create character embedding dictionary
        with tf.variable_scope(self.name, initializer=tf.random_normal_initializer(stddev=0.1)):
            self.c_table = tf.get_variable("c_table",
                [self.alphabet_size, self.character_d]
            )
        
        # Compute character features
        cv = tf.gather(self.c_table, self.ci) # [nodes, samples, character_d]
        cx = tf.concat([cv, self.cf], 2)      # [nodes, samples, character_d+2]
        self.cx = cx
        return
        
    def create_cell(self, hidden_d, input_d, is_top_cell):
        if not hasattr(self, "kr"):
            self.kr = tf.placeholder(tf.float32)
        
        cell = tf.nn.rnn_cell.LSTMCell(hidden_d)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob = self.kr,
            state_keep_prob = self.kr,
            output_keep_prob = self.kr if is_top_cell else 1.0,
            variational_recurrent = True,
            input_size = input_d,
            dtype = tf.float32,
        )
        return cell
        
    def create_bidrectional_multi_cell(self, dx):
        forward_cell_list = []
        backward_cell_list = []
        for layer in range(self.hidden_layers):
            input_d = dx if layer==0 else self.hidden_d
            is_top_cell = (layer == self.hidden_layers-1)
            forward_cell_list.append(
                self.create_cell(self.hidden_d, input_d, is_top_cell)
            )
            backward_cell_list.append(
                self.create_cell(self.hidden_d, input_d, is_top_cell)
            )
        forward_cell = tf.nn.rnn_cell.MultiRNNCell(forward_cell_list)
        backward_cell = tf.nn.rnn_cell.MultiRNNCell(backward_cell_list)
        return forward_cell, backward_cell
        
    def create_hidden(self):
        """Create a layer to encode inputs."""
        
        
        dx = self.character_d + 2
        forward_cell, backward_cell = self.create_bidrectional_multi_cell(dx)
        with tf.variable_scope(self.name):
            with tf.variable_scope("context_encoder"):
                # top_output: [2, nodes, samples, hidden_d], axis0: forward/backward
                # last_state: [2, layers, 2, samples, hidden_d], axis0: forward/backward, axis2: LSTM c/h
                top_output, last_state = tf.nn.bidirectional_dynamic_rnn(
                    forward_cell,
                    backward_cell,
                    self.cx, # [nodes, samples, dx]
                    sequence_length = self.cl, # [samples]
                    dtype = tf.float32,
                    time_major = True,
                )
        shf = last_state[0][self.hidden_layers-1].h[:,:] # [samples, hidden_d]
        shb = last_state[1][self.hidden_layers-1].h[:,:] # [samples, hidden_d]
        sh = tf.concat([shf, shb, self.sf], 1)    # [samples, hidden_d*2 + 1]
        self.sh = sh
        return
        
    def create_output(self):
        """Create an output layer computing 1/0 likelihood."""
        
        assert self.output_d_list[-1] == 2
        dh = self.hidden_d*2 + 1
        
        with tf.variable_scope(self.name):
            self.output_layer_list = []
            output_d_list = [dh] + self.output_d_list
            for i in range(1, len(output_d_list)):
                self.output_layer_list.append(
                    tf.layers.Dense(output_d_list[i], use_bias=True, name=f"output_{i}")
                )
                
        o = self.sh # [samples, dh]
        for output_layer in self.output_layer_list[:-1]:
            o = output_layer(o)
            o = tf.nn.relu(o)
        o = self.output_layer_list[-1](o) # [samples, 2]
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=o) # [samples]
        loss = tf.reduce_sum(loss)
        self.loss = loss
        
        # Output positive likelihood
        pl = tf.nn.softmax(o, 1) # [samples, 2]
        pl = pl[:,1]             # [samples]
        self.pl = pl
        return
        
    def create_update_op(self):
        """Create the computation of back-propagation."""
        
        if self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer()
        elif self.optimizer == "nadam":
            optimizer = tf.contrib.opt.NadamOptimizer()
        
        def clip_gradient(g):
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            if self.gradient_threshold == 0:
                return g # Nadam cant deal with IndexedSlices, so conversion is needed even if there is no clipping
            n = tf.norm(g)
            return tf.cond(n<self.gradient_threshold, lambda: g, lambda: g*self.gradient_threshold/n)
        
        gv_list = optimizer.compute_gradients(self.loss)
        gv_list = [(clip_gradient(g), v) for g, v in gv_list]
        self.update_op = optimizer.apply_gradients(gv_list)
        return
        
    def get_formatted_input(self, sample_list):
        """Preprocessing: extract numpy arrays for tensorflow input placeholders from input data.
        
        sample_list: [sample]
            sample: [Y/N, mention_string, prefix_string, suffix_string]
        """
        samples = len(sample_list)
        nodes = max(len(prefix)+len(suffix) for _, _, prefix, suffix in sample_list)
        
        #             [nodes, samples]
        y  = np.zeros([       samples   ], dtype=np.int32)
        sf = np.zeros([       samples, 1], dtype=np.float32)
        cl = np.zeros([       samples   ], dtype=np.int32)
        ci = np.zeros([nodes, samples   ], dtype=np.int32)
        cf = np.zeros([nodes, samples, 2], dtype=np.float32)
        
        unk_index = self.character_to_index["UNK"]
        for s, (label, mention, prefix, suffix) in enumerate(sample_list):
            y[s] = 1 if label=="Y" else 0
            sf[s][0] = len(mention)
            context = prefix + suffix
            cl[s] = len(context)
            for n, c in enumerate(context):
                ci[n][s] = self.character_to_index.get(c, unk_index)
                if n < len(prefix):
                    cf[n][s][0] = 1
                else:
                    cf[n][s][1] = 1
        
        return y, sf, cl, ci, cf
        
    def train(self, sample_list):
        """Update parameters for a batch of samples."""
        
        y, sf, cl, ci, cf = self.get_formatted_input(sample_list)
        
        loss, _ = self.sess.run(
            [self.loss, self.update_op],
            feed_dict = {
                self.y: y, self.sf: sf,
                self.cl: cl, self.ci: ci, self.cf:cf,
                self.kr: self.keep_rate,
            }
        )
        return loss
        
    def predict(self, sample_list):
        """Predict the positive likehood of each sample."""
        
        _, sf, cl, ci, cf = self.get_formatted_input(sample_list)
        
        pl = self.sess.run(self.pl,
            feed_dict = {
                self.sf: sf,
                self.cl: cl, self.ci: ci, self.cf:cf,
                self.kr: 1.0,
            }
        )
        return pl
        
def main():
    config = Config()
    model = Model(config)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        for v in tf.trainable_variables():
            print(v)
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    
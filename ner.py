import os
import random

import tensorflow as tf

from model import Model, Config

def get_model(character_list_file, model_directory, model_name, hidden="2-100", output="100-2"):
    """Load a pretrained Model."""
    
    config = Config()
    config.name = model_name
    config.hidden_layers, config.hidden_d = [int(i) for i in hidden.split("-")]
    config.output_d_list = [int(i) for i in output.split("-")]
    
    print(f"Reading {character_list_file}...", end="", flush=True)
    with open(character_list_file, "r") as f:
        line_list = f.read().splitlines()
    character_to_index = {line: index for index, line in enumerate(line_list)}
    character_to_index["UNK"] = len(character_to_index)
    print(f" {len(character_to_index)} characters")
    config.character_to_index = character_to_index
    config.alphabet_size = len(character_to_index)
    
    with tf.Graph().as_default():
        model = Model(config)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        model.sess = tf.Session(config=tf_config)
        model.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(model.sess, os.path.join(model_directory, model_name))
    return model
    
def make_batch_list(sample_list, batch_samples, batch_nodes):
    """Create batches of samples.
    
    batch_list: [(index, sample)]
        sample: [Y/N, mention_string, prefix_string, suffix_string]
    """
    
    index_sample_list = sorted(
        enumerate(sample_list),
        key=lambda index_sample: len(index_sample[1][2]) + len(index_sample[1][3])
    )
    
    batch_list = []
    batch = []
    for index, sample in index_sample_list:
        batch.append((index, sample))
        nodes = len(sample[2]) + len(sample[3])
        if len(batch)>=batch_samples or len(batch)*nodes>=batch_nodes:
            batch_list.append(batch)
            batch = []
    if batch:
        batch_list.append(batch)
    
    random.shuffle(batch_list)
    return batch_list

def predict(model, sample_list, batch_samples=256, batch_nodes=8000):
    """Predict the positive likelihood of each sample.
    
    sample_list: [sample]
        sample: [Y/N, mention_string, prefix_string, suffix_string]
    """
    
    batch_list = make_batch_list(sample_list, batch_samples, batch_nodes)
    pl_list = [None] * len(sample_list)
    
    for batch in batch_list:
        index_list, sub_sample_list = zip(*batch)
        for i, pl in enumerate(model.predict(sub_sample_list)):
            pl_list[index_list[i]] = pl
    return pl_list

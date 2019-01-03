import os
import sys
import time
import random
import argparse

import numpy as np
import tensorflow as tf

from model import Config, Model

def load_meta_data(config):
    """Read dataset-related configuration."""
    
    if config.dataset == "oldhan":
        import oldhan as data_utils
    
    config.character_to_index = data_utils.read_character_list(config.data_file)
    config.alphabet_size = len(config.character_to_index)
    return

def initialize_model(config):
    """Define Tensorflow graph and create model according to configuration."""
    
    model = Model(config)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    model.sess = tf.Session(config=tf_config)
    model.sess.run(tf.global_variables_initializer())
    return model
    
def load_data(config):
    """Read dataset.
    
    split_data: {split: sample_list}
        sample_list: [sample]
            sample: [Y/N, mention_string, prefix_string, suffix_string]
    """
    
    if config.dataset == "oldhan":
        import oldhan as data_utils
    
    split_data = data_utils.read_dataset(config.data_file)
    return split_data
    
def make_batch_list(sample_list, batch_samples=16, batch_nodes=500):
    """Create batches of samples.
    
    batch_list: [(index, sample)]
        sample: [Y/N, mention_string, prefix_string, suffix_string]
    """
    
    # order samples by prefix + suffix
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

def train_an_epoch(model, sample_list):
    """Update model parameters for every sample once."""
    
    batch_list = make_batch_list(sample_list, batch_samples=16, batch_nodes=500)
    
    total_samples = len(sample_list)
    samples = 0
    loss = 0.
    for batch in batch_list:
        _, sub_sample_list = zip(*batch)
        loss += model.train(sub_sample_list)
        samples += len(batch)
        
        print(" "*8 + "\r" + f"({samples}/{total_samples}) loss={loss/samples:.2e}", end="", flush=True)
    print("\r" + " "*64 + "\r", end="", flush=True)
    
    return loss/samples

def predict_dataset(model, sample_list):
    """Predict the positive likelihood of each sample."""
    # line_list = [
        # u"N\t050000847\t宗　人　府\t部　郞　中　　臣汪　桂原　任　\t　主　事臣徐　煥內閣中書今",
        # u"Y\t050000909\t山西道\t。皆於月之二十五日。兵部堂官會\t御史掣籤於　天安門外。若掣差官",
        # u"Y\t050000909\t山西道\t議之件。而督以例限。每月於兵科\t註銷。",
        # u"Y\t050000909\t山西道\t務府、奉宸苑、上駟院、武備院、\t御史、北城御史、鑲白旗、崇文門",
        # u"Y\t050000909\t山西道\t。監察御史。滿洲一人。漢一人。\t掌印監察御史。滿洲一人。漢一人",
        # u"Y\t050000909\t山西道\t十三倉。浙江道稽察禮部都察院。\t稽察兵部翰林院六科中書科總督倉",
        # u"Y\t050000909\t山西道\t蘇安徽刑名。浙江道掌浙江刑名。\t掌山西刑名。山東道掌山東刑名。",
        # u"Y\t050000909\t山西道\t。河南道御史監掣。武職月選籤。\t御史監掣。搭餉。局錢搭放兵餉。",
        # u"Y\t050000909\t山西道\t畿道江南道各三人。河南道浙江道\t山東道陝西道湖廣道江西道福建道",
        # u"Y\t050000909\t山西道\t西司廣西司雲南司。都察院河南道\t陝西道湖廣道江西道福建道廣西道",
    # ]
    # for index, line in enumerate(line_list):
        # print line
        # label, entity_id, mention, prefix, suffix = line.split("\t")
        # assert label == sample_list[index][0]
        # assert mention == sample_list[index][1]
        # assert prefix == sample_list[index][2]
        # assert suffix == sample_list[index][3]
    batch_list = make_batch_list(sample_list, batch_samples=256, batch_nodes=8000)
    pl_list = [None] * len(sample_list)
    total_samples = len(sample_list)
    samples = 0
    
    for batch in batch_list:
        index_list, sub_sample_list = zip(*batch)
        samples += len(sub_sample_list)
        for i, pl in enumerate(model.predict(sub_sample_list)):
            pl_list[index_list[i]] = pl
            
        print(" "*8 + "\r" + f"({samples}/{total_samples})", end="", flush=True)
    print("\r" + " "*64 + "\r", end="", flush=True)
            
    # for i, sample in enumerate(sample_list):
        # label, mention, prefix, suffix = sample
        # pl = pl_list[i]
        # print "%.2f%%\t%s\t%s\t%s\t%s\t%s" % (pl*100, label, "000000000", mention, prefix, suffix)
    return pl_list
    
def evaluate_prediction(sample_list, pl_list, positive_threshold=0.5):
    """Compute accuracies across samples."""
    
    samples = len(sample_list)
    correct = 0.
    for index, (label, _, _, _) in enumerate(sample_list):
        y = 1 if label=="Y" else 0
        y_hat = 1 if pl_list[index]>positive_threshold else 0
        if y == y_hat:
            correct += 1
    return correct/samples

def train_script(config):
    """Update model parameters until it converges or reaches maximum epochs."""
    
    load_meta_data(config)
    split_data = load_data(config)
    model = initialize_model(config)
    
    saver = tf.train.Saver()
    best_epoch = 0
    best_score = -1
    best_loss = float("inf")
    for epoch in range(1, config.max_epochs+1):
        print(f"\n<Epoch {epoch}>")
        
        start_time = time.time()
        loss = train_an_epoch(model, split_data["train"])
        elapsed = time.time() - start_time
        print(f"[train] loss={loss:.2e} elapsed={elapsed:.0f}s")
        
        start_time = time.time()
        pl_list = predict_dataset(model, split_data["test"])
        score = evaluate_prediction(split_data["test"], pl_list)
        elapsed = time.time() - start_time
        print(f"[test] acc={score:.2%} elapsed={elapsed:.0f}s", end="")
        
        if best_score < score:
            print(" best")
            best_epoch = epoch
            best_score = score
            best_loss = loss
            saver.save(model.sess, f"./model/{config.name}")
        else:
            print(f" worse #{epoch-best_epoch}")
        if epoch-best_epoch >= config.patience: break
    
    saver.restore(model.sess, f"./model/{config.name}")
    pl_list = predict_dataset(model, split_data["train"])
    score = evaluate_prediction(split_data["train"], pl_list)
    print(f"\n<Best Epoch {best_epoch}> loss={best_loss:.2e} train={score:.2%} test={best_score:.2%}")
    return

def evaluate_script(config):
    """Compute the accuracy of an existing model."""
    
    load_meta_data(config)
    split_data = load_data(config)
    model = initialize_model(config)
    
    saver = tf.train.Saver()
    saver.restore(model.sess, f"./model/{config.name}")
    
    start_time = time.time()
    split = config.split_list[0]
    pl_list = predict_dataset(model, split_data[split])
    score = evaluate_prediction(split_data[split], pl_list)
    elapsed = time.time() - start_time
    print(f"[{split}] acc={score:.2%} elapsed={elapsed:.0f}s")
    return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", dest="mode", default="train", choices=["train", "evaluate"])
    parser.add_argument("-dataset", dest="dataset", default="oldhan", choices=["oldhan"])
    parser.add_argument("-data_file", dest="data_file")
    parser.add_argument("-split", dest="split", default="test", choices=["train", "test"])
    parser.add_argument("-hidden", dest="hidden", default="2-100")
    parser.add_argument("-output", dest="output", default="100-2")
    parser.add_argument("-optimizer", dest="optimizer", default="nadam", choices=["adam", "nadam"])
    parser.add_argument("-gradient_threshold", dest="gradient_threshold", default="1")
    parser.add_argument("-epoch", dest="epoch", default="400-20")
    parser.add_argument("-suffix", dest="suffix", default="")
    parser.add_argument("-model_name", dest="model_name", default="")
    arg = parser.parse_args()
    
    config = Config()
    config.name = (
        "model"
        + "_" + arg.dataset
        + "_" + arg.data_file[arg.data_file.rfind("/")+1:]
        + "_BiLSTM-" + arg.hidden
        + "_output-" + arg.output)
    if arg.suffix: config.name += "_" + arg.suffix
    if arg.model_name: config.name = arg.model_name
    
    config.dataset = arg.dataset
    config.data_file = arg.data_file
    
    config.hidden_layers, config.hidden_d = [int(i) for i in arg.hidden.split("-")]
    config.output_d_list = [int(i) for i in arg.output.split("-")]
    
    config.optimizer = arg.optimizer
    config.gradient_threshold = float(arg.gradient_threshold)
    
    config.max_epochs, config.patience = [int(i) for i in arg.epoch.split("-")]
    
    if arg.mode == "train":
        config.split_list = ["train", "test"]
        train_script(config)        
    elif arg.mode == "evaluate":
        config.split_list = [arg.split]
        evaluate_script(config)
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    
import os
import sys
import time
import random
from collections import defaultdict

raw_data_path = "/share/home/jacobvsdanniel/NER/oldhan/unlabeled_oldhan"

dataset = "oldhan"
entity_type_to_data_file = {
    "person": os.path.join(dataset, "person.txt"),
    "location": os.path.join(dataset, "location.txt"),
    "organization": os.path.join(dataset, "organization.txt"),
    "officer": os.path.join(dataset, "officer.txt"),
}
entity_type_to_labeled_eid_list_file = {
    "person": os.path.join(dataset, "person_id.txt"),
    "organization": os.path.join(dataset, "organization_id.txt"),
    "officer": os.path.join(dataset, "officer_id.txt"),
}
corpus_id_to_labeled_eid_list_file = {
    "0211002": os.path.join(dataset, "location_id_1.txt"),
    "0206017": os.path.join(dataset, "location_id_1.txt"),
    "0202024": os.path.join(dataset, "location_id_1.txt"),
    "0211001": os.path.join(dataset, "location_id_2.txt"),
    "0206071": os.path.join(dataset, "location_id_4.txt"),
    "0206072": os.path.join(dataset, "location_id_4.txt"),
    "0206073": os.path.join(dataset, "location_id_4.txt"),
    "0206075": os.path.join(dataset, "location_id_4.txt"),
    "0206078": os.path.join(dataset, "location_id_4.txt"),
    "0206074": os.path.join(dataset, "location_id_4.txt"),
    "0206076": os.path.join(dataset, "location_id_4.txt"),
    "0206079": os.path.join(dataset, "location_id_3.txt"),
}
entity_id_prefix_to_entity_type = {
    "01": "person",
    "02": "location",
    "05": "organization",
    "06": "officer"
}

def extract_data():
    """Extract usable data from partially-checked corpora of person, location, organization, and officer."""
    
    # Read all data
    corpus_id_to_sample_list = defaultdict(lambda: [])
    mention_to_eid_set = defaultdict(lambda: set())
    for corpus_id in os.listdir(raw_data_path):
        corpus_directory = os.path.join(raw_data_path, corpus_id)
        print(f"Reading corpus {corpus_directory}...")
        for file_name in os.listdir(corpus_directory):
            entity_file = os.path.join(corpus_directory, file_name)
            with open(entity_file, "r") as f:
                line_list = f.read().splitlines()
            for line_index, line in enumerate(line_list[1:]):
                y, entity_id, mention, prefix, suffix, _, _ = line.split("\t")
                corpus_id_to_sample_list[corpus_id].append([y, entity_id, mention, prefix, suffix])
                mention_to_eid_set[mention].add(entity_id)
    
    # Read list of labeled IDs
    entity_type_to_labeled_eid_to_index = {}
    for entity_type, labeled_eid_list_file in entity_type_to_labeled_eid_list_file.items():
        with open(labeled_eid_list_file, "r") as f:
            labeled_eid_list = f.read().splitlines()
        entity_type_to_labeled_eid_to_index[entity_type] = {labeled_eid: index for index, labeled_eid in enumerate(labeled_eid_list)}
        
    corpus_id_to_labeled_eid_to_index = {}
    for corpus_id, labeled_eid_list_file in corpus_id_to_labeled_eid_list_file.items():
        with open(labeled_eid_list_file, "r") as f:
            labeled_eid_list = f.read().splitlines()
        corpus_id_to_labeled_eid_to_index[corpus_id] = {labeled_eid: index for index, labeled_eid in enumerate(labeled_eid_list)}
    
    # Collect usable data
    entity_type_to_sample_list = {entity_type: [] for entity_type in entity_type_to_data_file}
    for corpus_id, sample_list in corpus_id_to_sample_list.items():
        for y, entity_id, mention, prefix, suffix in sample_list:
            # Exclude the samples with unknown labels
            if y not in ["Y", "N"]: continue
            # Exclude the mentions that do not map to one single entity
            if len(mention_to_eid_set[mention]) > 1: continue
            # Exclude the entities that are not labeled
            entity_type = entity_id_prefix_to_entity_type[entity_id[:2]]
            if entity_type in entity_type_to_labeled_eid_to_index:
                if entity_id not in entity_type_to_labeled_eid_to_index[entity_type]:
                    continue
            elif corpus_id in corpus_id_to_labeled_eid_to_index:
                if entity_id not in corpus_id_to_labeled_eid_to_index[corpus_id]:
                    continue
            else:
                continue
            # Add usable data
            entity_type_to_sample_list[entity_type].append([y, entity_id, mention, prefix, suffix])
    
    # Write usable data
    for entity_type, sample_list in entity_type_to_sample_list.items():
        mention_to_samples = defaultdict(lambda: 0)
        positives = 0
        with open(entity_type_to_data_file[entity_type], "w") as f:
            for y, entity_id, mention, prefix, suffix in sample_list:
                mention_to_samples[mention] += 1
                if y=="Y": positives += 1
                f.write(f"{y}\t{entity_id}\t{mention}\t{prefix}\t{suffix}\n")
        unique_mentions = len(mention_to_samples)
        samples = len(sample_list)
        positive = positives/samples
        print(f"{entity_type}: {unique_mentions} unique_mentions, {samples} samples, {positive:.2%} positive")
    return
    
def split_data(data_file):
    """Split a dataset containing usable Y/N samples of one entity type into train-test."""
    
    print(f"Splitting {data_file}...")
    
    # Read data
    with open(data_file, "r") as f:
        sample_list = f.read().splitlines()
    
    # Group samples by mention
    samples = len(sample_list)
    mention_to_sample_list = defaultdict(lambda: [])
    mention_to_entity_id = {}
    for sample in sample_list:
        _, entity_id, mention, _, _ = sample.split("\t")
        mention_to_sample_list[mention].append(sample)
        if mention in mention_to_entity_id:
            assert mention_to_entity_id[mention] == entity_id
        else:
            mention_to_entity_id[mention] = entity_id
    
    # Shuffle group order
    mention_samplelist_list = list(mention_to_sample_list.items())
    random.shuffle(mention_samplelist_list)
    
    # Split data into 9:1 and write down
    split_samples = defaultdict(lambda: 0)
    split_mentions = defaultdict(lambda: 0)
    split_entity_id_set = defaultdict(lambda: set())
    split_sample_fp = {}
    split_mention_fp = {}
    with open(data_file+"_train_sample", "w") as split_sample_fp["train"],\
         open(data_file+"_train_mention_entity", "w") as split_mention_fp["train"],\
         open(data_file+"_test_sample", "w") as split_sample_fp["test"],\
         open(data_file+"_test_mention_entity", "w") as split_mention_fp["test"]:
         
        sub_samples = 0
        for mention, sample_list in mention_samplelist_list:
            split = "train" if sub_samples < samples*9/10 else "test"
            sub_samples += len(sample_list)
            for sample in sample_list:
                split_sample_fp[split].write(sample + "\n")
            entity_id = mention_to_entity_id[mention]
            split_mention_fp[split].write(f"{mention} {entity_id}\n") 
            
            split_mentions[split] += 1
            split_samples[split] += len(sample_list)
            split_entity_id_set[split].add(entity_id)
            
    for split in ["train", "test"]:
        entities = len(split_entity_id_set[split])
        print(f"  {entities} entities, {split_mentions[split]} mentions, {split_samples[split]} samples")
    return
    
def read_dataset(data_file, data_split_list=["train", "test"], write_category_list=False):
    """Read a dataset of Y/N samples of one entity type."""
    
    # Read samples for each split
    split_sample_list = defaultdict(lambda: [])
    
    split_character_count = defaultdict(lambda: defaultdict(lambda: 0))
    split_y_count = defaultdict(lambda: defaultdict(lambda: 0))
    split_eid_set = defaultdict(lambda: set())
    split_mention_set = defaultdict(lambda: set())
    mention_to_entity_id = {}
    
    print(f"Reading {data_file}...")
    for split in data_split_list:
        split_file = f"{data_file}_{split}_sample"
        with open(split_file, "r") as f:
            line_list = f.read().splitlines()
        for line in line_list:
            y, eid, mention, prefix, suffix = line.split("\t")
            split_sample_list[split].append([y, mention, prefix, suffix])
            
            for character in mention+prefix+suffix:
                split_character_count[split][character] += 1
            split_y_count[split][y] += 1
            split_eid_set[split].add(eid)
            split_mention_set[split].add(mention)
            
            assert y in ["Y", "N"]
            if mention in mention_to_entity_id:
                assert mention_to_entity_id[mention] == eid
            else:
                mention_to_entity_id[mention] = eid
    
    
    # Show statistics of each data split 
    print("-" * 80)
    print(f"{'split':>20s}{'entities':>12s}{'mentions':>12s}{'samples':>12s}{'characters':>12s}{'Y':>8s}")
    print("-" * 80)
    for split in data_split_list:
        entities = len(split_eid_set[split])
        mentions = len(split_mention_set[split])
        samples = len(split_sample_list[split])
        characters = sum(split_character_count[split].values())
        positive = split_y_count[split]["Y"] / (split_y_count[split]["Y"]+split_y_count[split]["N"])
        print(f"{split:>20s}{entities:12d}{mentions:12d}{samples:12d}{characters:12d}{positive:8.2%}")
    
    # Get character distribution
    if write_category_list and "train" in data_split_list:
        character_list_file = f"{data_file}_train_character"
        with open(character_list_file, "w") as f:
            for character, count in sorted(split_character_count["train"].items(), key=lambda x: x[1], reverse=True):
                if count < 10: break
                f.write(f"{character}\n")
    
    return split_sample_list
    
def read_character_list(data_file):
    """Read the list of trainable characters."""
    
    character_list_file = f"{data_file}_train_character"
    
    print(f"Reading {character_list_file}...", end="", flush=True)
    with open(character_list_file, "r") as f:
        line_list = f.read().splitlines()
    character_to_index = {line: index for index, line in enumerate(line_list)}
    character_to_index["UNK"] = len(character_to_index)
    
    print(f" {len(character_to_index)} characters")
    return character_to_index
    
def main():
    # extract_data()
    
    for entity_type, data_file in entity_type_to_data_file.items():
        # if entity_type not in ["officer"]: continue
        # split_data(data_file)
        print()
        read_character_list(data_file)
        read_dataset(data_file, write_category_list=True)
        pass
        
    return
    
if __name__ == "__main__":
    main()
    sys.exit
    
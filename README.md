# ancient_chinese_NER

This repository contains a model for training, evaluating, predicting entities for ancient Chinese documents.
Included are a model implemented with TensorFlow, scripts for train and evaluate, Python API for prediction.

```python
model.py # The computation graph
evaluate.py # The training and evaluation scripts
oldhan.py # The API for extracting and reading ancient Chinese data
```

## IHP Dataset
### 1. Get the dataset

The dataset is provided by Institute of History and Philology, Academia Sinica (中央研究院歷史語言研究所).

Corpora
* 明實錄
* 大明會典
* 明史
* 清實錄
* 大清會典
* 大清會典事例
* 四庫全書總目提要

Annotation
* 人名 (person)
* 地名 (location)
* 機關 (organization)
* 職官 (officer)

Contact IHP for the dataset.
Set its root directory in oldhan.py.
```python
raw_data_path = "/share/home/jacobvsdanniel/NER/oldhan/unlabeled_oldhan"
```

### 2. Preprocess Data

As the dataset is only partially annotated, contact IHP for label information.
Set their path in oldhan.py, where eid stands for entity ID.
```python
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
```

Set output path, extract usable data and split into train-test with functions in oldhan.py.
```python
entity_type_to_data_file = {
    "person": os.path.join(dataset, "person.txt"),
    "location": os.path.join(dataset, "location.txt"),
    "organization": os.path.join(dataset, "organization.txt"),
    "officer": os.path.join(dataset, "officer.txt"),
}
extract_data()
for entity_type, data_file in entity_type_to_data_file.items():
    split_data(data_file)
    read_dataset(data_file, write_category_list=True)
```

## Data Format

The preprocessed IHP dataset or any custom dataset should observe the following naming convention and format.
Take entity type person as example.

* person.txt_train_sample
* person.txt_test_sample

Samples are delimited by line breaks.
Every line is a tab delimited list of label, entity ID, mention, left context, right context.
```
Y       010001937       李衞    議具奏欽此遵旨議凖原任直隸總督  應於京師賢良祠崇祀並照例立傳○
```

* person.txt_train_character

Character list file can be created automatically by using the following function in oldhan.py.
```python
read_dataset("person.txt", write_category_list=True)
```

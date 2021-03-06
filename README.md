# ancient_chinese_NER

This repository contains a model for training, evaluating, predicting entities for ancient Chinese documents.

Included are a model implemented with TensorFlow, scripts for train and evaluate, Python API for prediction.

[Online Demo](http://sky.iis.sinica.edu.tw:9010)

```python
model.py # Computation graph
evaluate.py # Training and evaluation scripts
oldhan.py # Preprocessing functions for ancient Chinese data
ner.py # Prediction API
demo.py # Sample command line usage of prediction API
web_demo/ # Sample website usage of prediction API
```

## 1. IHP Dataset
### Get the dataset

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

### Preprocess Data

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

## 2. Data Format

The preprocessed IHP dataset or any custom dataset should observe the following naming convention and format.
Take entity type person as example.

* person.txt_train_sample
* person.txt_test_sample

Samples are delimited by line breaks.
Every line is a tab delimited list of label, entity ID, mention, left context, right context.
```
Y       010001937       李衞    議具奏欽此遵旨議凖原任直隸總督  應於京師賢良祠崇祀並照例立傳○
N       010002111       德光    漢家封駕六龍第十駕六龍御翠華帝  天下薄海内總一家四徼外正朔加者
```

* person.txt_train_character

Character list file can be created automatically by using the following function in oldhan.py.
```python
read_dataset("person.txt", write_category_list=True)
```

## 3. Train Model

Prerequisite
* TensorFlow 1.12.0

Train model for, say, entity type person with files under ./oldhan/. The output will be saved to ./model/.
```
python evaluate.py -mode train -dataset oldhan -data_file oldhan/person.txt -hidden 2-100 -output 100-2 -suffix 20190101
```

Evaluate performance on, say, testing split.
```
python evaluate.py -mode evaluate -split test -dataset oldhan -data_file oldhan/person.txt -hidden 2-100 -output 100-2 -suffix 20190101
```

## 4. Prediction API

Prerequisite
* TensorFlow 1.12.0

Two Python API are provided in ner.py.
```python
get_model(character_list_file, model_directory, model_name, hidden="2-100", output="100-2")
predict(model, sample_list, batch_samples=256, batch_nodes=8000)
```

Their sample usage is provided in demo.py.
```python
entity_type_to_model[entity_type] = ner.get_model(
    f"oldhan/{entity_type}.txt_train_character",
    "model",
    f"model_oldhan_{entity_type}.txt_BiLSTM-2-100_output-100-2_run1",
    hidden = "2-100",
    output = "100-2",
)
pl_list = ner.predict(
    entity_type_to_model[entity_type],
    sample_list,
    batch_samples = 256,
    batch_nodes = 8000,
)
for index, line in enumerate(line_list):
    print(f"{pl_list[index]:.2%}\t{line}")
```

A demo website is provided in demo_website/. To start the server, run
```
$ python server.py -host 0.0.0.0 -port 9002
```

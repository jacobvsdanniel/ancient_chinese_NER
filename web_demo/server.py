import os
import sys
import time
import json
import argparse

from flask import Flask, render_template, request

from ner import get_model, predict

app = Flask(__name__)

entity_type_to_model = {}

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/load_model", methods=["POST"])
def load_model():
    # Read entity type from frontend
    data = json.loads(request.data)
    entity_type = data["entity_type"].strip()
    
    global entity_type_to_model
    if entity_type not in entity_type_to_model:
        entity_type_to_model[entity_type] = get_model(
            f"oldhan/{entity_type}.txt_train_character",
            "model",
            f"model_oldhan_{entity_type}.txt_BiLSTM-2-100_output-100-2_run1",
            hidden = "2-100",
            output = "100-2",
        )
    
    return "\n".join(sorted(entity_type_to_model))
    
@app.route("/predict_positive_likelihood", methods=["POST"])
def predict_positive_likelihood():
    
    # Read entity type, sample list from frontend
    data = json.loads(request.data)
    entity_type = data["entity_type"].strip()
    sample_list = data["sample_list"].strip().split("\n")
    
    pl_list = predict(
        entity_type_to_model[entity_type],
        [line.split("\t") for line in sample_list],
        batch_samples = 256,
        batch_nodes = 8000,
    )
    
    annotated_text = ""
    for i, pl in enumerate(pl_list):
        annotated_text += f"Y信心 {pl:>4.0%}\n"
    
    return annotated_text
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host", default="0.0.0.0")
    parser.add_argument("-port", default="9002")
    arg = parser.parse_args()
    
    app.run(host=arg.host, port=arg.port)
    return
    
if __name__ == '__main__':
    main()
    sys.exit()
    
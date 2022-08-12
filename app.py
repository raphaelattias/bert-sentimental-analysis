from distutils import text_file
from flask import Flask, jsonify, request
import io
import json
from src.datamodules.text_datamodule import TextDataModule, TextDataSet
from src.models.bert_module import BertModule
from service_streamer import ThreadedStreamer


from pytorch_lightning import Trainer

app = Flask(__name__)

model = BertModule().load_from_checkpoint(checkpoint_path='logs/experiments/runs/default/2022-08-10_11-46-45/checkpoints/epoch_000.ckpt').net
text_dm = TextDataModule(texts_list = ["text"], labels_list= [1])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.json['file']
        # text = file.read().decode('UTF-8')
        text_ds = TextDataSet(texts = [text], labels= [1])
        text_dm.data_test = text_ds

        x,y = next(iter(text_dm.test_dataloader()))
        pred = pred = model(x["input_ids"])['logits'].argmax().item()

        return jsonify({"positive": pred})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
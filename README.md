# Bert for Sentimental Analysis

## Introduction
This repository is meant to be a personal exercise on MLOPS for the NLP task. The exercise covers many crucial steps of MLOPS. Here are the key points from this project.
1. Use a SOTA model from **huggingface**, here I used BERT.
2. Train on a NLP datasets using the **dataset** library, here we used iMDB.
2. Make configuration and reproducibility easier with **hydra**.
3. Use a modern ML library, I used **Pytorch Lightning**.
3. Make the model trainable on Apple Silicon with **Torch 1.12**.
4. Finetune model with logging and tracking using **wandb**.
5. Find best hyper parameters by performing a sweep using **wandb sweep**.
6. Deploy model on a REST API with **Flask** to perform HTML POST request on a server.
7. Make the model sharable using **Docker** to build an image.
8. Benchmark the Flask API using **wrk**.

## Installation
```
git clone bert-sentimental-analysis
pip install -r requirements.txt
```

## Usage

The model can be trained on CPU, GPU or Apple Silicon by running the command `python train.py`. The pre-trained BERT model and iMDB dataset should be downloaded. The best hyperparameters are already set in the `config` folder. 

Once the `ckpt` checkpoint is obtained, simply put the path in `app.py` and start the flask server: 
````
FLASK_APP=app.py flask run
````
One can then perform HTML request to the REST API, an example is given in `test_app.py`.

## Docker 

To build a docker image, simply run:
````
docker build -f Dockerfile -t bert:train .  
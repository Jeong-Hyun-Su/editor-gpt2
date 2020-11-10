import time 
import json
import random
import requests
import threading
from queue import Queue, Empty

import torch
from torch.nn import functional as F
import numpy as np
from transformers import AutoTokenizer

from flask import Flask, request, Response, jsonify, render_template


app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1


# url 수정 필요
models = {
    "gpt2-large": "http://localhost:80/",
    "gpt2-cover-letter": "http://localhost:80/",
    "gpt2-reddit": "http://localhost:80/",
    "gpt2-story": "http://localhost:80/",
    "gpt2-ads": "http://localhost:80/",
    "gpt2-business": "http://localhost:80/",
    "gpt2-film": "http://localhost:80/",
    "gpt2-trump": "http://localhost:80/",
    "gpt2-debate": "http://localhost:80/"
}

tokenizer_url = {
    "gpt2-reddit": "mrm8488/gpt2-finetuned-reddit-tifu",
    "gpt2-ads": "gpt2-adstext/gpt2-adstext",
    "gpt2-business": "laxya007/gpt2_business",
    "gpt2-trump": "huggingtweets/realdonaldtrump",
    "gpt2-story": "pranavpsv/gpt2-genre-story-generator",
    "gpt2-film": "cpierse/gpt2_film_scripts",
    "gpt2-large": "gpt2-large",
    "gpt2-cover-letter": "jonasmue/cover-letter-gpt2",
    "gpt2-debate": "zanderbush/DebateWriting"
}

def load_tokenizers():
    tokenizers = {}
    
    for name, url in tokenizer_url.items():
        tokenizers[name] = AutoTokenizer.from_pretrained(url)

    return tokenizers

tokenizers = load_tokenizers()

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:    
                requests['output'] = run_generation(requests['input'][0], requests['input'][1], requests['input'][2])

threading.Thread(target=handle_requests_by_batch).start()

def run_generation(model, text, mode):
    tokenizer = tokenizers[model]
    input_ids = tokenizer.encode(text)

    min_length = len(input_ids)
    if mode == "long":
        length = min_length + 20
    else: 
        length = min_length + random.randrange(2, 6)

    url = models[model] + model
    header = {"content-type":"application/json"}
    data = json.dumps({
        "text": input_ids,
        "num_samples": 5,
        "length": length
    })

   
    count = 0
    while True: 
        response = requests.post(url, data=data, headers=header)

        if response.status_code == 200:    
            outputs = json.loads(response.text)
            result = {}

            for idx, output in enumerate(outputs):
                result[idx] = tokenizer.decode(output[min_length:], skip_special_tokens=True)
                result[idx] = result[idx].strip()

            return result
        
        # 응답이 429이면 재시도
        elif response.status_code == 429:
            count += 1
            time.sleep(0.2)
        
        elif response.status_code not in [429, 200] or count == 15:
            return {'error': 'failed'}

@app.route("/gpt2", methods=["POST"])
def gpt2():    
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'TooManyReqeusts'}), 429
    
    try:
        args = []

        model = request.form['model']
        context = request.form['context']
        length = request.form['length']

        args = [model, context, length]
    except Exception:
        return jsonify({'error':'Invalid Inputs'}), 400

    # Queue
    req = {
        'input': args
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)
    
    result = req['output']

    if 'error' in result:
        return Response("fail", status=500)

    return result

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/")
def main():
    return render_template("index.html")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
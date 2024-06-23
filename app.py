import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_next_word(input_text, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs[1]
    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode(next_token_id)
    return next_token

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['text']
    next_word = generate_next_word(input_text, model, tokenizer)
    return jsonify({'next_word': next_word})

if __name__ == '__main__':
    app.run(debug=True)

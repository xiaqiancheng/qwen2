from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_name = "./Qwen2-1.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def process_input(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除输入提示部分，仅保留生成的助理回复
    response = response.split('\nassistant\n', 1)[-1]
    return response


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data['text']
    processed_input = process_input(user_input)
    bot_response = generate_response(processed_input)

    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


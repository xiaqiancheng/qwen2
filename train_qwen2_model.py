import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# 参数配置
MODEL_NAME = "./Qwen2-1.5B-Instruct"
FINE_TUNED_MODEL_PATH = "./models/fine_tuned"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 3

# 预设问题和回答
preset_data = {
    'train': [
        {'question': "你好", 'answer': "你好，有什么可以帮您的吗？"},
        {'question': "你是谁", 'answer': "我是智能客服助手。"},
        {'question': "今天天气怎么样", 'answer': "今天的天气很好，阳光明媚。"}
    ],
    'validation': [
        {'question': "你叫什么名字", 'answer': "我叫智能客服助手。"},
        {'question': "能帮我订票吗", 'answer': "当然可以，请告诉我您的出行日期和目的地。"}
    ]
}

# 加载预训练模型和分词器
def load_pretrained_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# 保存模型和分词器
def save_model_and_tokenizer(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# 加载和预处理数据集
def load_and_preprocess_dataset(tokenizer):
    # 构建 DatasetDict
    dataset_dict = DatasetDict()

    # 转换预设数据格式
    train_examples = [{"question": q['question'], "answer": q['answer']} for q in preset_data['train']]
    validation_examples = [{"question": q['question'], "answer": q['answer']} for q in preset_data['validation']]

    # 创建数据集
    dataset_dict['train'] = Dataset.from_dict({"question": [ex["question"] for ex in train_examples],
                                               "answer": [ex["answer"] for ex in train_examples]})
    dataset_dict['validation'] = Dataset.from_dict({"question": [ex["question"] for ex in validation_examples],
                                                    "answer": [ex["answer"] for ex in validation_examples]})

    def preprocess_function(examples):
        inputs = [f"问: {q} 答: {a}" for q, a in zip(examples['question'], examples['answer'])]
        encodings = tokenizer(inputs, max_length=MAX_LENGTH, padding="max_length", truncation=True)

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': None,  # 如果有标签数据，根据需要进行调整
        }

    # 对数据集进行预处理
    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

    return tokenized_dataset

# 设置训练参数
def get_training_arguments():
    return TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
    )

# 微调模型
def fine_tune_model(model, tokenized_dataset):
    training_args = get_training_arguments()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()

# 主函数
def main():
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        os.makedirs(FINE_TUNED_MODEL_PATH)

    # 加载预训练模型和分词器
    tokenizer, model = load_pretrained_model()

    # 加载和预处理数据集
    tokenized_dataset = load_and_preprocess_dataset(tokenizer)

    # 微调模型
    fine_tune_model(model, tokenized_dataset)

    # 保存微调后的模型
    save_model_and_tokenizer(model, tokenizer, FINE_TUNED_MODEL_PATH)
    print("模型微调完成并保存")

if __name__ == "__main__":
    main()

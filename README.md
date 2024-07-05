# 千问2大模型

### 1.环境安装
本案例基于Python>=3.8，请在您的计算机上安装好Python；
国内镜像安装

```
pip install transformers datasets torch fastapi uvicorn flask -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.模型下载
[访问 Hugging Face 模型库](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)

在模型页面上，点击 "Files and versions" 标签，下载到./Qwen2-1.5B-Instruct 目录下

### 3.运行 Flask 应用程序
```
python3 app.py
```
测试部署

可以使用 curl 或 Postman 来测试您的部署。以下是使用 curl 的示例：
```
curl -X POST http://127.0.0.1:5000/generate -H "Content-Type: application/json" -d '{"text": "Your input text here"}'
```

训练脚本
```
python3 train_qwen2_model.py
```

# Qwen2大模型微调
[知乎](https://zhuanlan.zhihu.com/p/702491999)


#### Qwen 系列模型的不同版本主要在于参数量、用途和量化方式。以下是一些关键区别：

1. 参数量：

    0.5B、1.5B、7B、57B、72B：代表模型的参数量，参数量越大，模型越复杂和强大。

2. 用途：

    Instruct：经过指令微调，适用于执行各种指令任务。

    非 Instruct：通用语言模型。

3. 量化方式：

    AWQ、GPTQ-Int8、GPTQ-Int4、MLX、GGUF：不同的量化技术，影响模型的推理速度和内存使用。
    
具体型号如 Qwen2-72B-Instruct-GPTQ-Int4 代表一个经过指令微调、使用 GPTQ-Int4 量化技术的 720 亿参数模型。根据具体需求选择适合的版本。
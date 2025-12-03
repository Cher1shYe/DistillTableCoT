# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # 指定模型路径
# model_path = "./final_model"  # 例如: "./模型目录"

# # 加载模型和分词器
# print("正在加载模型...")
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,  # 半精度节省内存
#     device_map="auto"  # 自动选择GPU/CPU
# )
# print("模型加载完成!")

# # 使用模型生成文本
# def generate_text(prompt, max_length=100):
#     inputs = tokenizer(prompt, return_tensors="pt")
    
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=max_length,
#             num_return_sequences=1,
#             temperature=0.7,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # 测试模型
# prompt = "请解释一下机器学习的基本概念："
# result = generate_text(prompt)
# print("模型回复:", result)

from transformers import pipeline
import torch

# 创建文本生成pipeline
model_path = "./final_model"
generator = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    device=0 if torch.cuda.is_available() else -1  # 使用GPU如果可用
)

# 直接使用
results = generator(
    "请解释一下机器学习的基本概念：",
    max_length=50,
    temperature=0.7,
    do_sample=True,
    num_return_sequences=1
)

print(results[0]['generated_text'])
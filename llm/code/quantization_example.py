#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM量化示例程序
展示如何使用不同量化方法压缩大型语言模型，以降低内存占用和推理成本
"""

import os
import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
import numpy as np

# 支持的量化方法
QUANT_METHODS = ['fp16', 'int8', 'int4', 'gptq', 'awq']

def parse_args():
    parser = argparse.ArgumentParser(description="LLM量化演示程序")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Hugging Face模型ID")
    parser.add_argument("--quant_method", type=str, choices=QUANT_METHODS, default="fp16",
                        help=f"量化方法: {', '.join(QUANT_METHODS)}")
    parser.add_argument("--benchmark", action="store_true", 
                        help="是否进行性能基准测试")
    parser.add_argument("--output_dir", type=str, default="./quantized_models",
                        help="量化模型输出目录")
    return parser.parse_args()

def load_model_and_tokenizer(model_id, quant_method, device="cuda"):
    """
    加载模型并根据指定方法进行量化
    """
    print(f"正在加载模型 {model_id} 并应用 {quant_method} 量化...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 根据量化方法加载模型
    if quant_method == "fp16":
        # 半精度加载
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    elif quant_method == "int8":
        # 8位整数量化
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto"
        )
    
    elif quant_method == "int4":
        # 4位整数量化
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto"
        )
    
    elif quant_method == "gptq":
        # GPTQ量化 (需要预先量化的模型)
        try:
            from transformers import GPTQConfig
            quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
        except Exception as e:
            print(f"GPTQ量化失败，请确保已安装相关依赖: {e}")
            print("回退到4位量化...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto"
            )
    
    elif quant_method == "awq":
        # AWQ量化 (需要预先量化的模型或optimum库)
        try:
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_pretrained(
                model_id,
                device_map="auto"
            )
        except Exception as e:
            print(f"AWQ量化失败，请确保已安装相关依赖: {e}")
            print("回退到4位量化...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto"
            )
    
    return model, tokenizer

def save_quantized_model(model, tokenizer, quant_method, output_dir):
    """
    保存量化后的模型
    """
    save_dir = os.path.join(output_dir, f"{quant_method}_model")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在保存量化模型到 {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"模型已保存到: {save_dir}")

def run_inference_benchmark(model, tokenizer, n_samples=20):
    """
    执行推理性能基准测试
    """
    print("\n开始性能基准测试...")
    
    # 加载基准数据集
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
        prompts = dataset["instruction"][:n_samples]
    except Exception:
        # 如果无法加载数据集，使用预定义的提示
        prompts = [
            "请解释量子计算的基本原理",
            "写一个关于气候变化的短文",
            "解释人工智能如何影响就业市场",
            "给一个10岁的孩子解释什么是相对论",
            "写一个Python函数来检测回文"
        ] * 4  # 重复以达到样本数
        prompts = prompts[:n_samples]
    
    # 创建文本生成管道
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    
    # 测量推理延迟
    total_tokens = 0
    generation_times = []
    token_generation_speeds = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n测试提示 {i+1}/{len(prompts)}")
        print(f"提示: {prompt[:50]}...")
        
        # 编码输入以计算输入令牌数
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        input_tokens = input_ids.shape[1]
        
        # 生成文本并测量时间
        start_time = time.time()
        outputs = pipeline(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
        end_time = time.time()
        
        generated_text = outputs[0]["generated_text"]
        
        # 计算生成的总令牌数
        output_ids = tokenizer.encode(generated_text, return_tensors="pt").to(model.device)
        output_tokens = output_ids.shape[1]
        new_tokens = output_tokens - input_tokens
        
        generation_time = end_time - start_time
        generation_times.append(generation_time)
        
        tokens_per_second = new_tokens / generation_time
        token_generation_speeds.append(tokens_per_second)
        total_tokens += new_tokens
        
        print(f"生成时间: {generation_time:.2f}秒")
        print(f"生成速度: {tokens_per_second:.2f} tokens/秒")
        print(f"生成样例: {generated_text[:100]}...")
    
    # 计算并显示统计数据
    avg_time = np.mean(generation_times)
    avg_speed = np.mean(token_generation_speeds)
    p90_time = np.percentile(generation_times, 90)
    
    print("\n基准测试结果:")
    print(f"平均生成时间: {avg_time:.2f}秒")
    print(f"P90生成时间: {p90_time:.2f}秒")
    print(f"平均生成速度: {avg_speed:.2f} tokens/秒")
    print(f"总生成tokens: {total_tokens}")
    
    # 内存使用统计
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        print(f"GPU内存使用: {gpu_memory_used:.2f} GB")

def print_model_size_stats(model):
    """
    打印模型大小统计信息
    """
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    total_params_millions = total_params / (1000 ** 2)
    
    # 估计模型内存占用
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    print("\n模型统计信息:")
    print(f"参数数量: {total_params_millions:.2f}M")
    print(f"估计内存占用: {size_mb:.2f} MB")
    
    # 如果在GPU上，打印实际GPU内存使用情况
    if next(model.parameters()).is_cuda:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        print(f"GPU总内存: {gpu_memory:.2f} GB")
        print(f"已分配GPU内存: {allocated:.2f} GB")
        print(f"已保留GPU内存: {reserved:.2f} GB")

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device = "cuda"
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("GPU不可用，使用CPU")
    
    # 加载并量化模型
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.quant_method, device)
    
    # 打印模型统计信息
    print_model_size_stats(model)
    
    # 简单推理示例
    prompt = "人工智能的未来发展趋势是什么?"
    print(f"\n示例提示: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(
            input_ids, 
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        end_time = time.time()
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n生成文本: {generated_text}")
    print(f"生成时间: {end_time - start_time:.2f}秒")
    
    # 性能基准测试
    if args.benchmark:
        run_inference_benchmark(model, tokenizer)
    
    # 保存量化模型
    save_quantized_model(model, tokenizer, args.quant_method, args.output_dir)

if __name__ == "__main__":
    main() 
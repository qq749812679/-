#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM推理服务器示例
演示如何构建一个高性能的LLM推理服务器，支持批处理和并发请求
"""

import os
import time
import json
import asyncio
import argparse
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from queue import Queue, PriorityQueue
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 请求模型
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="生成文本的提示")
    max_tokens: int = Field(256, description="生成的最大token数量")
    temperature: float = Field(0.7, description="采样温度", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="nucleus采样的概率阈值", ge=0.0, le=1.0)
    stream: bool = Field(False, description="是否流式返回生成结果")
    priority: int = Field(1, description="请求优先级(1-5，1为最高)", ge=1, le=5)

class GenerationResponse(BaseModel):
    id: str = Field(..., description="请求ID")
    text: str = Field(..., description="生成的文本")
    prompt_tokens: int = Field(..., description="提示的token数量")
    completion_tokens: int = Field(..., description="生成的token数量")
    total_tokens: int = Field(..., description="总token数量")
    generation_time: float = Field(..., description="生成时间(秒)")

# 批处理请求管理器
class BatchManager:
    def __init__(
        self, 
        model, 
        tokenizer, 
        batch_size=4, 
        max_batch_wait_time=0.5,
        device="cuda",
        max_queue_size=100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_batch_wait_time = max_batch_wait_time
        self.device = device
        
        # 优先级队列用于保存等待的请求
        self.request_queue = PriorityQueue(maxsize=max_queue_size)
        
        # 用于存储流式响应的字典
        self.streamers = {}
        
        # 启动批处理循环
        self.batch_thread = Thread(target=self._batch_loop, daemon=True)
        self.batch_thread.start()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_tokens": 0,
            "avg_batch_size": 0,
            "avg_wait_time": 0,
        }
    
    async def add_request(self, request_id: str, request: GenerationRequest) -> Optional[TextIteratorStreamer]:
        """将请求添加到批处理队列"""
        try:
            self.stats["total_requests"] += 1
            
            # 处理流式请求
            streamer = None
            if request.stream:
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                self.streamers[request_id] = streamer
            
            # 对请求进行编码
            request_data = {
                "id": request_id,
                "request": request,
                "time": time.time(),
                "input_ids": None,  # 会在批处理线程中设置
                "attention_mask": None,
                "result_queue": asyncio.Queue(),
                "streamer": streamer
            }
            
            # 根据优先级将请求添加到队列
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.request_queue.put((request.priority, request_data))
            )
            
            logger.info(f"请求 {request_id} 添加到队列 (优先级: {request.priority})")
            return streamer
            
        except Exception as e:
            logger.error(f"添加请求到队列时出错: {e}")
            raise
    
    def _batch_loop(self):
        """批处理主循环，持续处理队列中的请求"""
        while True:
            try:
                batch = []
                start_time = time.time()
                
                # 收集批次请求
                while len(batch) < self.batch_size and (time.time() - start_time) < self.max_batch_wait_time:
                    try:
                        # 非阻塞方式获取请求，超时时间为剩余的最大等待时间
                        remaining_wait_time = max(0, self.max_batch_wait_time - (time.time() - start_time))
                        _, request_data = self.request_queue.get(timeout=remaining_wait_time)
                        
                        # 对输入进行编码
                        encoded_input = self.tokenizer(
                            request_data["request"].prompt,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                        
                        request_data["input_ids"] = encoded_input["input_ids"].to(self.device)
                        request_data["attention_mask"] = encoded_input["attention_mask"].to(self.device)
                        request_data["prompt_tokens"] = len(encoded_input["input_ids"][0])
                        
                        batch.append(request_data)
                        
                    except Exception as e:
                        if "queue.Empty" not in str(e.__class__):
                            logger.error(f"收集批次时出错: {e}")
                        break
                
                # 如果批次不为空，则处理它
                if batch:
                    wait_times = [time.time() - req["time"] for req in batch]
                    avg_wait_time = sum(wait_times) / len(wait_times)
                    self.stats["avg_wait_time"] = (self.stats["avg_wait_time"] * self.stats["total_batches"] + avg_wait_time) / (self.stats["total_batches"] + 1)
                    
                    logger.info(f"处理批次: {len(batch)} 请求, 平均等待时间: {avg_wait_time:.3f}秒")
                    self.stats["total_batches"] += 1
                    self.stats["avg_batch_size"] = (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + len(batch)) / self.stats["total_batches"]
                    
                    # 处理批次
                    self._process_batch(batch)
            
            except Exception as e:
                logger.error(f"批处理循环中出错: {e}")
                time.sleep(1)  # 出错时暂停，避免无限循环消耗资源
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """处理一个请求批次"""
        try:
            # 准备批次输入
            batch_inputs = []
            batch_attention_masks = []
            batch_max_tokens = []
            
            for req in batch:
                batch_inputs.append(req["input_ids"])
                batch_attention_masks.append(req["attention_mask"])
                batch_max_tokens.append(req["request"].max_tokens)
            
            # 将输入拼接成批次
            batch_input_ids = torch.cat(batch_inputs)
            batch_attention_mask = torch.cat(batch_attention_masks)
            
            # 对于流式请求，使用单独的线程
            stream_threads = []
            for req in batch:
                if req["streamer"]:
                    thread = Thread(
                        target=self._generate_stream,
                        args=(
                            req["id"],
                            req["input_ids"],
                            req["attention_mask"],
                            req["request"],
                            req["streamer"],
                            req["result_queue"],
                            req["prompt_tokens"]
                        )
                    )
                    stream_threads.append(thread)
                    thread.start()
            
            # 处理非流式请求作为一个批次
            non_stream_reqs = [req for req in batch if not req["streamer"]]
            if non_stream_reqs:
                non_stream_inputs = torch.cat([req["input_ids"] for req in non_stream_reqs])
                non_stream_masks = torch.cat([req["attention_mask"] for req in non_stream_reqs])
                
                # 执行生成
                with torch.no_grad():
                    gen_start_time = time.time()
                    
                    # 为批次中的每个请求使用其各自的生成参数
                    outputs = self.model.generate(
                        input_ids=non_stream_inputs,
                        attention_mask=non_stream_masks,
                        max_new_tokens=max(req["request"].max_tokens for req in non_stream_reqs),
                        do_sample=True,
                        temperature=non_stream_reqs[0]["request"].temperature,  # 简化：使用第一个请求的温度
                        top_p=non_stream_reqs[0]["request"].top_p,  # 简化：使用第一个请求的top_p
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
                    
                    gen_end_time = time.time()
                
                # 处理结果并将其发送到各自的队列
                offset = 0
                for req in non_stream_reqs:
                    input_length = req["input_ids"].shape[1]
                    output = outputs.sequences[offset]
                    completed_text = self.tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    completion_tokens = len(output) - input_length
                    
                    # 创建响应
                    response = GenerationResponse(
                        id=req["id"],
                        text=completed_text,
                        prompt_tokens=req["prompt_tokens"],
                        completion_tokens=completion_tokens,
                        total_tokens=req["prompt_tokens"] + completion_tokens,
                        generation_time=gen_end_time - gen_start_time
                    )
                    
                    # 更新统计信息
                    self.stats["total_tokens"] += response.total_tokens
                    
                    # 发送结果
                    asyncio.run_coroutine_threadsafe(
                        req["result_queue"].put(response),
                        asyncio.get_event_loop()
                    )
                    
                    offset += 1
                    logger.info(f"请求 {req['id']} 已完成: {completion_tokens} tokens")
            
            # 等待所有流式线程完成
            for thread in stream_threads:
                thread.join()
                
        except Exception as e:
            logger.error(f"处理批次时出错: {e}")
            # 向所有请求发送错误
            for req in batch:
                asyncio.run_coroutine_threadsafe(
                    req["result_queue"].put({"error": str(e)}),
                    asyncio.get_event_loop()
                )
    
    def _generate_stream(
        self, 
        req_id, 
        input_ids, 
        attention_mask, 
        request, 
        streamer, 
        result_queue,
        prompt_tokens
    ):
        """为单个请求生成流式响应"""
        try:
            gen_start_time = time.time()
            
            # 在单独的线程中启动生成
            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                streamer=streamer,
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 等待生成完成
            thread.join()
            
            # 计算生成的token数量
            generated_text = ""
            for text in streamer:
                generated_text += text
            
            completion_tokens = len(self.tokenizer.encode(generated_text))
            gen_end_time = time.time()
            
            # 创建响应
            response = GenerationResponse(
                id=req_id,
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                generation_time=gen_end_time - gen_start_time
            )
            
            # 更新统计信息
            self.stats["total_tokens"] += response.total_tokens
            
            # 发送完成信号
            asyncio.run_coroutine_threadsafe(
                result_queue.put(response),
                asyncio.get_event_loop()
            )
            
            logger.info(f"流式请求 {req_id} 已完成: {completion_tokens} tokens")
            
        except Exception as e:
            logger.error(f"流式生成时出错: {e}")
            asyncio.run_coroutine_threadsafe(
                result_queue.put({"error": str(e)}),
                asyncio.get_event_loop()
            )
    
    async def get_result(self, request_id: str) -> GenerationResponse:
        """获取请求的结果"""
        # 找到对应的请求
        for _, req_data in list(self.request_queue.queue):
            if req_data["id"] == request_id:
                return await req_data["result_queue"].get()
        
        raise ValueError(f"请求ID {request_id} 未找到")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取批处理管理器的统计信息"""
        stats = self.stats.copy()
        stats["queue_size"] = self.request_queue.qsize()
        return stats

# FastAPI应用
app = FastAPI(title="LLM推理服务")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
batch_manager = None

@app.on_event("startup")
async def startup_event():
    global batch_manager
    args = parse_args()
    
    # 加载模型和分词器
    logger.info(f"正在加载模型: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # 根据需要添加特殊令牌
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if args.int8:
        logger.info("使用INT8量化加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            load_in_8bit=True,
        )
    elif args.int4:
        logger.info("使用INT4量化加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            load_in_4bit=True,
        )
    else:
        logger.info("使用FP16加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    # 创建批处理管理器
    batch_manager = BatchManager(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_wait_time=args.max_wait_time,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_queue_size=args.max_queue_size
    )
    
    logger.info(f"服务器已启动，监听端口: {args.port}")

@app.post("/v1/completions", response_model=GenerationResponse)
async def create_completion(request: GenerationRequest):
    """创建文本补全"""
    request_id = str(uuid.uuid4())
    
    try:
        # 将请求添加到批处理管理器
        streamer = await batch_manager.add_request(request_id, request)
        
        if request.stream:
            # 返回流式响应
            async def generate_stream():
                try:
                    for text in streamer:
                        yield f"data: {json.dumps({'id': request_id, 'text': text})}\n\n"
                    
                    # 获取最终结果以获取token计数
                    final_result = await batch_manager.get_result(request_id)
                    yield f"data: {json.dumps({'id': request_id, 'finish_reason': 'stop', 'usage': {'prompt_tokens': final_result.prompt_tokens, 'completion_tokens': final_result.completion_tokens, 'total_tokens': final_result.total_tokens}})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"流式响应中出错: {e}")
                    yield f"data: {json.dumps({'id': request_id, 'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        else:
            # 等待结果并返回
            result = await batch_manager.get_result(request_id)
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return result
            
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/stats")
async def get_stats():
    """获取服务器统计信息"""
    if batch_manager:
        return batch_manager.get_stats()
    return {"error": "服务器尚未初始化"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

def parse_args():
    parser = argparse.ArgumentParser(description="LLM推理服务器")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="要加载的模型ID")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务器端口")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="最大批处理大小")
    parser.add_argument("--max_wait_time", type=float, default=0.5,
                        help="批处理最大等待时间(秒)")
    parser.add_argument("--max_queue_size", type=int, default=100,
                        help="请求队列最大大小")
    parser.add_argument("--int8", action="store_true",
                        help="使用INT8量化")
    parser.add_argument("--int4", action="store_true",
                        help="使用INT4量化")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run("inference_server:app", host="0.0.0.0", port=args.port, log_level="info") 
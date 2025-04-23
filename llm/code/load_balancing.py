#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM负载均衡示例
演示如何实现一个智能的LLM负载均衡器，可以根据多种策略路由请求到多个LLM服务实例
"""

import os
import time
import json
import asyncio
import random
import logging
import argparse
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    model: Optional[str] = Field(None, description="指定使用的模型")
    routing_strategy: str = Field("least_busy", description="路由策略: least_busy, round_robin, random, latency")

class ServerState(BaseModel):
    """表示LLM服务实例的状态"""
    id: str
    url: str
    model: str
    capacity: int = Field(1, description="相对处理能力")
    active_requests: int = 0
    total_requests: int = 0
    avg_latency: float = 0
    health_status: bool = True
    last_health_check: float = 0
    last_response_time: float = 0

# 负载均衡器
class LLMLoadBalancer:
    def __init__(self):
        # 服务器实例列表
        self.servers: Dict[str, ServerState] = {}
        
        # 用于轮询调度的计数器
        self.round_robin_counter = 0
        
        # 用于历史延迟统计的窗口
        self.latency_window_size = 10
        self.latency_history: Dict[str, List[float]] = {}
        
        # 用于健康检查的间隔(秒)
        self.health_check_interval = 30
        
        # 创建HTTP会话
        self.session = None
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
        }
    
    async def initialize(self):
        """初始化负载均衡器"""
        self.session = aiohttp.ClientSession()
        
        # 启动健康检查循环
        asyncio.create_task(self._health_check_loop())
    
    async def close(self):
        """关闭负载均衡器，释放资源"""
        if self.session:
            await self.session.close()
    
    def add_server(self, server_url: str, model: str, capacity: int = 1) -> str:
        """添加LLM服务实例"""
        server_id = str(uuid.uuid4())
        self.servers[server_id] = ServerState(
            id=server_id,
            url=server_url,
            model=model,
            capacity=capacity,
            last_health_check=time.time()
        )
        self.latency_history[server_id] = []
        logger.info(f"添加服务器: {server_url} (model: {model}, capacity: {capacity})")
        return server_id
    
    def remove_server(self, server_id: str) -> bool:
        """移除LLM服务实例"""
        if server_id in self.servers:
            del self.servers[server_id]
            if server_id in self.latency_history:
                del self.latency_history[server_id]
            logger.info(f"移除服务器: {server_id}")
            return True
        return False
    
    async def _health_check_loop(self):
        """定期检查所有服务器的健康状态"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_servers_health()
            except Exception as e:
                logger.error(f"健康检查循环中出错: {e}")
    
    async def _check_all_servers_health(self):
        """检查所有服务器的健康状态"""
        logger.info("执行健康检查...")
        
        for server_id, server in list(self.servers.items()):
            try:
                # 如果距离上次健康检查时间不足，则跳过
                if time.time() - server.last_health_check < self.health_check_interval * 0.9:
                    continue
                
                # 使用健康检查端点检查服务器状态
                health_url = f"{server.url}/health"
                start_time = time.time()
                
                async with self.session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        # 更新服务器状态
                        server.health_status = True
                        latency = time.time() - start_time
                        self._update_server_latency(server_id, latency)
                        logger.info(f"服务器 {server.url} 正常, 延迟: {latency:.3f}秒")
                    else:
                        logger.warning(f"服务器 {server.url} 返回非200状态码: {response.status}")
                        server.health_status = False
                
            except Exception as e:
                logger.error(f"检查服务器 {server.url} 健康状态时出错: {e}")
                server.health_status = False
            
            # 更新上次健康检查时间
            server.last_health_check = time.time()
    
    def _update_server_latency(self, server_id: str, latency: float):
        """更新服务器的延迟历史记录"""
        if server_id in self.latency_history:
            # 添加新延迟并保持窗口大小
            history = self.latency_history[server_id]
            history.append(latency)
            if len(history) > self.latency_window_size:
                history.pop(0)
            
            # 更新服务器的平均延迟
            if history:
                avg_latency = sum(history) / len(history)
                self.servers[server_id].avg_latency = avg_latency
    
    def _select_server_random(self, model: Optional[str] = None) -> Optional[ServerState]:
        """随机选择一个健康的服务器"""
        # 筛选健康的服务器
        healthy_servers = [s for s in self.servers.values() if s.health_status]
        if model:
            healthy_servers = [s for s in healthy_servers if s.model == model]
        
        if not healthy_servers:
            return None
        
        # 随机选择一个服务器
        return random.choice(healthy_servers)
    
    def _select_server_round_robin(self, model: Optional[str] = None) -> Optional[ServerState]:
        """使用轮询算法选择服务器"""
        # 筛选健康的服务器
        healthy_servers = [s for s in self.servers.values() if s.health_status]
        if model:
            healthy_servers = [s for s in healthy_servers if s.model == model]
        
        if not healthy_servers:
            return None
        
        # 使用轮询计数器选择服务器
        self.round_robin_counter += 1
        index = self.round_robin_counter % len(healthy_servers)
        return healthy_servers[index]
    
    def _select_server_least_busy(self, model: Optional[str] = None) -> Optional[ServerState]:
        """选择负载最轻的服务器"""
        # 筛选健康的服务器
        healthy_servers = [s for s in self.servers.values() if s.health_status]
        if model:
            healthy_servers = [s for s in healthy_servers if s.model == model]
        
        if not healthy_servers:
            return None
        
        # 根据活跃请求数和处理能力计算负载比率
        min_load_ratio = float('inf')
        selected_server = None
        
        for server in healthy_servers:
            # 计算负载比率(活跃请求/容量)
            load_ratio = server.active_requests / server.capacity
            
            if load_ratio < min_load_ratio:
                min_load_ratio = load_ratio
                selected_server = server
        
        return selected_server
    
    def _select_server_lowest_latency(self, model: Optional[str] = None) -> Optional[ServerState]:
        """选择延迟最低的服务器"""
        # 筛选健康的服务器
        healthy_servers = [s for s in self.servers.values() if s.health_status]
        if model:
            healthy_servers = [s for s in healthy_servers if s.model == model]
        
        if not healthy_servers:
            return None
        
        # 按平均延迟排序
        sorted_servers = sorted(healthy_servers, key=lambda s: s.avg_latency)
        return sorted_servers[0] if sorted_servers else None
    
    def select_server(self, request: GenerationRequest) -> Optional[ServerState]:
        """根据请求选择合适的服务器"""
        strategy = request.routing_strategy.lower()
        model = request.model
        
        # 根据策略选择服务器
        if strategy == "random":
            return self._select_server_random(model)
        elif strategy == "round_robin":
            return self._select_server_round_robin(model)
        elif strategy == "latency":
            return self._select_server_lowest_latency(model)
        else:  # 默认使用最小负载策略
            return self._select_server_least_busy(model)
    
    async def forward_request(self, request: GenerationRequest) -> Any:
        """转发请求到选定的服务器并返回结果"""
        self.stats["total_requests"] += 1
        
        # 选择服务器
        server = self.select_server(request)
        if not server:
            self.stats["failed_requests"] += 1
            logger.error("未找到可用的服务器")
            raise HTTPException(status_code=503, detail="所有服务器都不可用")
        
        # 标记服务器上的活跃请求增加
        server.active_requests += 1
        server.total_requests += 1
        
        try:
            # 准备转发请求
            forward_url = f"{server.url}/v1/completions"
            request_data = request.dict(exclude={"routing_strategy"})
            
            if not request.stream:
                # 非流式请求
                start_time = time.time()
                async with self.session.post(forward_url, json=request_data, timeout=60) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"服务器返回错误: {response.status}, {error_text}")
                        self.stats["failed_requests"] += 1
                        raise HTTPException(status_code=response.status, detail=error_text)
                    
                    result = await response.json()
                    end_time = time.time()
                    
                # 更新统计信息
                latency = end_time - start_time
                self._update_server_latency(server.id, latency)
                server.last_response_time = end_time
                
                # 更新全局统计信息
                self.stats["successful_requests"] += 1
                n = self.stats["successful_requests"]
                self.stats["avg_response_time"] = ((n-1) * self.stats["avg_response_time"] + latency) / n
                
                return result
                
            else:
                # 流式请求，需要直接流式返回
                return await self._forward_stream_request(server, forward_url, request_data)
                
        except aiohttp.ClientError as e:
            logger.error(f"请求服务器 {server.url} 时出错: {e}")
            self.stats["failed_requests"] += 1
            raise HTTPException(status_code=502, detail=f"服务器连接错误: {str(e)}")
            
        finally:
            # 标记服务器上的活跃请求减少
            server.active_requests -= 1
    
    async def _forward_stream_request(self, server: ServerState, url: str, request_data: Dict[str, Any]):
        """转发流式请求并返回流式响应"""
        
        async def stream_generator():
            try:
                start_time = time.time()
                async with self.session.post(url, json=request_data, timeout=60) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"流式请求错误: {response.status}, {error_text}")
                        yield f"data: {json.dumps({'error': error_text})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # 直接转发SSE流
                    async for chunk in response.content:
                        yield chunk.decode('utf-8')
                
                end_time = time.time()
                latency = end_time - start_time
                
                # 更新统计信息
                self._update_server_latency(server.id, latency)
                server.last_response_time = end_time
                
                # 更新全局统计信息
                self.stats["successful_requests"] += 1
                n = self.stats["successful_requests"]
                self.stats["avg_response_time"] = ((n-1) * self.stats["avg_response_time"] + latency) / n
                
            except Exception as e:
                logger.error(f"流式转发出错: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取负载均衡器的统计信息"""
        # 基本统计信息
        stats = self.stats.copy()
        
        # 服务器统计信息
        server_stats = []
        for server_id, server in self.servers.items():
            server_stats.append({
                "id": server.id,
                "url": server.url,
                "model": server.model,
                "health": server.health_status,
                "active_requests": server.active_requests,
                "total_requests": server.total_requests,
                "avg_latency": server.avg_latency,
                "capacity": server.capacity,
            })
        
        stats["servers"] = server_stats
        return stats

# FastAPI应用
app = FastAPI(title="LLM负载均衡器")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局负载均衡器
load_balancer = LLMLoadBalancer()

@app.on_event("startup")
async def startup_event():
    """启动时初始化负载均衡器"""
    args = parse_args()
    
    # 初始化负载均衡器
    await load_balancer.initialize()
    
    # 添加服务器
    for server_config in args.servers:
        parts = server_config.split(",")
        url = parts[0].strip()
        model = parts[1].strip() if len(parts) > 1 else "default"
        capacity = int(parts[2]) if len(parts) > 2 else 1
        
        load_balancer.add_server(url, model, capacity)
    
    logger.info(f"负载均衡器已启动，监听端口: {args.port}")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    await load_balancer.close()
    logger.info("负载均衡器已关闭")

@app.post("/v1/completions")
async def create_completion(request: GenerationRequest):
    """处理补全请求"""
    try:
        return await load_balancer.forward_request(request)
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/stats")
async def get_stats():
    """获取负载均衡器的统计信息"""
    return load_balancer.get_stats()

@app.post("/v1/servers")
async def add_server(url: str, model: str = "default", capacity: int = 1):
    """添加新的服务器"""
    server_id = load_balancer.add_server(url, model, capacity)
    return {"server_id": server_id}

@app.delete("/v1/servers/{server_id}")
async def remove_server(server_id: str):
    """移除服务器"""
    success = load_balancer.remove_server(server_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"服务器 {server_id} 未找到")
    return {"success": True}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    # 检查是否有可用的服务器
    has_healthy_server = any(s.health_status for s in load_balancer.servers.values())
    
    if not has_healthy_server and load_balancer.servers:
        return {"status": "degraded", "message": "无可用服务器"}
    
    return {"status": "healthy"}

def parse_args():
    parser = argparse.ArgumentParser(description="LLM负载均衡器")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务器端口")
    parser.add_argument("--servers", type=str, nargs="+",
                        help="服务器列表，格式: URL,MODEL,CAPACITY", default=[])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run("load_balancing:app", host="0.0.0.0", port=args.port, log_level="info") 
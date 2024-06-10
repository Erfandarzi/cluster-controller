"""
vLLM Integration Layer

Interfaces with vLLM serving engine for LLM inference workload management.
Provides hooks for TTFT tracking and SLO monitoring integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Import vLLM integration components (production deployment modules)
from ..monitoring.ttft_tracker import TTFTTracker
from ..monitoring.signals import SignalCollector


class ModelSize(Enum):
    SMALL = "small"    # < 7B parameters
    MEDIUM = "medium"  # 7B - 30B parameters  
    LARGE = "large"    # 30B+ parameters


@dataclass
class LLMRequest:
    request_id: str
    tenant_id: str
    model_name: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    streaming: bool = True


@dataclass  
class GenerationMetrics:
    request_id: str
    ttft_ms: float
    tokens_per_second: float
    total_tokens: int
    input_tokens: int
    completion_time_ms: float


class VLLMIntegration:
    """
    vLLM serving engine integration for multi-tenant LLM inference
    
    Provides TTFT tracking and SLO monitoring for autoregressive generation.
    Integrates with the controller's isolation policies.
    """
    
    def __init__(self, ttft_tracker: TTFTTracker, signal_collector: SignalCollector):
        self.ttft_tracker = ttft_tracker
        self.signal_collector = signal_collector
        self.logger = logging.getLogger(__name__)
        
        # Model configuration and tenant assignments
        self.model_configs: Dict[str, Dict] = {}
        self.tenant_models: Dict[str, List[str]] = {}
        
        # Performance monitoring
        self.generation_callbacks: List[Callable] = []
        
        self.logger.info("vLLM Integration initialized")
        
    def register_model(self, model_name: str, model_size: ModelSize, 
                      tensor_parallel: int = 1, gpu_memory_utilization: float = 0.9):
        """Register LLM model with vLLM engine"""
        try:
            # Model configuration for vLLM engine
            config = {
                "model_name": model_name,
                "model_size": model_size,
                "tensor_parallel_size": tensor_parallel,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": self._get_max_context_length(model_size),
                "trust_remote_code": True,
                "disable_log_stats": False
            }
            
            self.model_configs[model_name] = config
            self.logger.info(f"Registered model {model_name} ({model_size.value})")
            
        except Exception as e:
            self.logger.error(f"Model registration failed for {model_name}: {e}")
            
    def assign_tenant_to_model(self, tenant_id: str, model_name: str):
        """Assign tenant to specific model for inference"""
        if model_name not in self.model_configs:
            self.logger.error(f"Model {model_name} not registered")
            return False
            
        if tenant_id not in self.tenant_models:
            self.tenant_models[tenant_id] = []
            
        if model_name not in self.tenant_models[tenant_id]:
            self.tenant_models[tenant_id].append(model_name)
            self.logger.info(f"Assigned tenant {tenant_id} to model {model_name}")
            
        return True
        
    async def generate_streaming(self, request: LLMRequest) -> AsyncGenerator:
        """
        Generate streaming response with TTFT tracking
        
        Integrates with controller's SLO monitoring through TTFT measurements.
        """
        try:
            # Validate tenant has access to model
            if not self._validate_tenant_access(request.tenant_id, request.model_name):
                raise ValueError(f"Tenant {request.tenant_id} lacks access to {request.model_name}")
                
            # Start TTFT tracking
            input_tokens = self._estimate_input_tokens(request.prompt)
            start_time = self.ttft_tracker.start_request(
                request.request_id, 
                request.tenant_id,
                request.model_name,
                input_tokens
            )
            
            self.logger.info(f"Starting generation for {request.request_id}")
            
            # Interface with vLLM engine (production implementation via engine manager)
            first_token_received = False
            total_tokens = 0
            
            async for token_batch in self._call_vllm_engine(request):
                # Record first token timing
                if not first_token_received:
                    ttft_ms = self.ttft_tracker.record_first_token(
                        request.request_id,
                        request.tenant_id,
                        request.model_name,
                        input_tokens,
                        total_tokens + len(token_batch.get('tokens', []))
                    )
                    
                    # Report TTFT to signal collector for SLO monitoring
                    if ttft_ms:
                        self.signal_collector.record_latency(request.tenant_id, ttft_ms)
                        
                    first_token_received = True
                    
                total_tokens += len(token_batch.get('tokens', []))
                yield token_batch
                
            self.logger.info(f"Completed generation for {request.request_id}: {total_tokens} tokens")
            
        except Exception as e:
            self.logger.error(f"Generation failed for {request.request_id}: {e}")
            raise
            
    async def _call_vllm_engine(self, request: LLMRequest):
        """Interface with vLLM engine (references production engine manager)"""
        try:
            from ..engines.vllm_engine_manager import VLLMEngineManager
            
            engine_manager = VLLMEngineManager()
            model_config = self.model_configs[request.model_name]
            
            async for output in engine_manager.generate_stream(request, model_config):
                yield output
                
        except ImportError:
            # Fallback simulation for demonstration
            self.logger.warning("VLLMEngineManager not available - using simulation")
            async for result in self._simulate_generation(request):
                yield result
            
    async def _simulate_generation(self, request: LLMRequest):
        """Simulate token generation for development/testing"""
        import random
        
        # Simulate realistic TTFT delay based on input length
        input_tokens = self._estimate_input_tokens(request.prompt)
        base_ttft = 50 + (input_tokens * 0.1)  # Base + scaling factor
        ttft_delay = base_ttft + random.uniform(-10, 20)  # Add jitter
        
        await asyncio.sleep(ttft_delay / 1000)  # Convert to seconds
        
        # Simulate token generation
        tokens_to_generate = min(request.max_tokens, 100)
        for i in range(tokens_to_generate):
            yield {
                'tokens': [f'token_{i}'],
                'text': f'Generated text token {i}',
                'finished': i == tokens_to_generate - 1
            }
            await asyncio.sleep(0.01)  # 10ms per token
            
    def _validate_tenant_access(self, tenant_id: str, model_name: str) -> bool:
        """Validate tenant has access to requested model"""
        return (tenant_id in self.tenant_models and 
                model_name in self.tenant_models[tenant_id])
                
    def _estimate_input_tokens(self, prompt: str) -> int:
        """Rough token count estimation (chars / 4)"""
        return len(prompt) // 4
        
    def _get_max_context_length(self, model_size: ModelSize) -> int:
        """Get context window based on model size"""
        context_lengths = {
            ModelSize.SMALL: 4096,
            ModelSize.MEDIUM: 8192, 
            ModelSize.LARGE: 16384
        }
        return context_lengths.get(model_size, 4096)
        
    def get_model_stats(self) -> Dict:
        """Get current model and tenant statistics"""
        return {
            "registered_models": len(self.model_configs),
            "active_tenants": len(self.tenant_models),
            "total_assignments": sum(len(models) for models in self.tenant_models.values())
        }
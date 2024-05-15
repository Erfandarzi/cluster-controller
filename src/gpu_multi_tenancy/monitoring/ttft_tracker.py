"""
Time-To-First-Token (TTFT) Monitoring

Specialized latency tracking for autoregressive generation workloads.
Tracks streaming inference patterns and first-token response times.
"""

import time
import logging
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class TTFTMeasurement:
    request_id: str
    tenant_id: str  
    model_name: str
    input_tokens: int
    ttft_ms: float
    total_tokens: int
    timestamp: float


class TTFTTracker:
    """
    Time-To-First-Token latency tracking for LLM serving
    
    Specialized for autoregressive generation where TTFT is critical
    for user experience in streaming applications.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # Per-tenant TTFT measurements
        self.tenant_measurements: Dict[str, Deque[TTFTMeasurement]] = {}
        
        # Request tracking for streaming inference
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        
        self.logger.info(f"TTFT Tracker initialized with window size {window_size}")
        
    def start_request(self, request_id: str, tenant_id: str, model_name: str, 
                     input_tokens: int) -> float:
        """Record start of inference request"""
        start_time = time.time()
        self.active_requests[request_id] = start_time
        
        self.logger.debug(f"Started request {request_id} for {tenant_id}, "
                         f"model={model_name}, input_tokens={input_tokens}")
        
        return start_time
        
    def record_first_token(self, request_id: str, tenant_id: str, model_name: str,
                          input_tokens: int, total_tokens: int) -> Optional[float]:
        """Record first token generation - calculates TTFT"""
        if request_id not in self.active_requests:
            self.logger.warning(f"Unknown request {request_id} for TTFT measurement")
            return None
            
        start_time = self.active_requests[request_id]
        current_time = time.time()
        ttft_ms = (current_time - start_time) * 1000
        
        # Create measurement record
        measurement = TTFTMeasurement(
            request_id=request_id,
            tenant_id=tenant_id,
            model_name=model_name,
            input_tokens=input_tokens,
            ttft_ms=ttft_ms,
            total_tokens=total_tokens,
            timestamp=current_time
        )
        
        # Add to tenant's measurement history
        if tenant_id not in self.tenant_measurements:
            self.tenant_measurements[tenant_id] = deque(maxlen=self.window_size)
            
        self.tenant_measurements[tenant_id].append(measurement)
        
        # Clean up active request
        del self.active_requests[request_id]
        
        self.logger.info(f"TTFT recorded: {tenant_id} request {request_id}: {ttft_ms:.1f}ms")
        
        return ttft_ms
        
    def get_ttft_p99(self, tenant_id: str) -> float:
        """Get p99 TTFT for tenant - primary SLO metric"""
        if tenant_id not in self.tenant_measurements:
            return 0.0
            
        measurements = self.tenant_measurements[tenant_id]
        if not measurements:
            return 0.0
            
        ttfts = [m.ttft_ms for m in measurements]
        return float(np.percentile(ttfts, 99))
        
    def get_ttft_p95(self, tenant_id: str) -> float:
        """Get p95 TTFT for tenant"""
        if tenant_id not in self.tenant_measurements:
            return 0.0
            
        measurements = self.tenant_measurements[tenant_id]
        if not measurements:
            return 0.0
            
        ttfts = [m.ttft_ms for m in measurements]
        return float(np.percentile(ttfts, 95))
        
    def get_ttft_p50(self, tenant_id: str) -> float:
        """Get median TTFT for tenant"""
        if tenant_id not in self.tenant_measurements:
            return 0.0
            
        measurements = self.tenant_measurements[tenant_id]
        if not measurements:
            return 0.0
            
        ttfts = [m.ttft_ms for m in measurements]
        return float(np.percentile(ttfts, 50))
        
    def get_slo_miss_rate(self, tenant_id: str, ttft_threshold_ms: float = 200.0) -> float:
        """Get TTFT SLO violation rate"""
        if tenant_id not in self.tenant_measurements:
            return 0.0
            
        measurements = self.tenant_measurements[tenant_id]
        if not measurements:
            return 0.0
            
        violations = sum(1 for m in measurements if m.ttft_ms > ttft_threshold_ms)
        return violations / len(measurements)
        
    def get_throughput_estimate(self, tenant_id: str, window_seconds: int = 60) -> float:
        """Estimate requests per second for tenant"""
        if tenant_id not in self.tenant_measurements:
            return 0.0
            
        measurements = self.tenant_measurements[tenant_id]
        if not measurements:
            return 0.0
            
        # Count recent requests within window
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_count = sum(1 for m in measurements if m.timestamp >= cutoff_time)
        return recent_count / window_seconds
        
    def get_tracking_summary(self) -> Dict:
        """Get overall tracking statistics"""
        total_measurements = sum(len(measurements) for measurements in self.tenant_measurements.values())
        
        return {
            "tracked_tenants": len(self.tenant_measurements),
            "total_measurements": total_measurements,
            "active_requests": len(self.active_requests),
            "window_size": self.window_size
        }
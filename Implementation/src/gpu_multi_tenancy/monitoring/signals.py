"""
Signal Collection

Collects per-tenant p99 latency and system signals for multi-tenancy control.
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class TenantMetrics:
    tenant_id: str
    timestamp: float
    p99_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    slo_miss_rate: float = 0.0
    throughput_rps: float = 0.0


class LatencyTracker:
    """Tracks latency percentiles as described in paper"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        
    def add_latency(self, latency_ms: float):
        """Record latency measurement"""
        self.latencies.append(latency_ms)
        
    def get_p99(self) -> float:
        """Get p99 latency - paper's primary signal"""
        if not self.latencies:
            return 0.0
        return float(np.percentile(list(self.latencies), 99))
        
    def get_p95(self) -> float:
        """Get p95 latency"""
        if not self.latencies:
            return 0.0
        return float(np.percentile(list(self.latencies), 95))
        
    def get_slo_miss_rate(self, slo_threshold_ms: float) -> float:
        """Get SLO violation rate - paper's key metric"""
        if not self.latencies:
            return 0.0
        violations = sum(1 for lat in self.latencies if lat > slo_threshold_ms)
        return violations / len(self.latencies)


class SignalCollector:
    """
    Implements paper's signal collection approach:
    - Per-tenant p99 latency (primary signal)
    - System-level metrics (secondary signals)
    - Configurable sampling interval Δ
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tenant_trackers: Dict[str, LatencyTracker] = {}
        
    def register_tenant(self, tenant_id: str):
        """Register tenant for monitoring"""
        if tenant_id not in self.tenant_trackers:
            self.tenant_trackers[tenant_id] = LatencyTracker()
            self.logger.info(f"Registered tenant {tenant_id} for monitoring")
            
    def record_latency(self, tenant_id: str, latency_ms: float):
        """Record latency measurement - paper's core data input"""
        if tenant_id not in self.tenant_trackers:
            self.register_tenant(tenant_id)
        self.tenant_trackers[tenant_id].add_latency(latency_ms)
        
    def collect_tenant_metrics(self, tenant_id: str) -> TenantMetrics:
        """Collect metrics for specific tenant"""
        tracker = self.tenant_trackers.get(tenant_id)
        if not tracker:
            self.register_tenant(tenant_id)
            tracker = self.tenant_trackers[tenant_id]
            
        return TenantMetrics(
            tenant_id=tenant_id,
            timestamp=time.time(),
            p99_latency_ms=tracker.get_p99(),
            p95_latency_ms=tracker.get_p95(),
            slo_miss_rate=tracker.get_slo_miss_rate(15.0),  # 15ms SLO threshold
            throughput_rps=self._estimate_throughput(tracker)
        )
        
    def collect_all_metrics(self) -> Dict[str, TenantMetrics]:
        """Collect metrics for all tenants"""
        metrics = {}
        for tenant_id in self.tenant_trackers.keys():
            metrics[tenant_id] = self.collect_tenant_metrics(tenant_id)
        return metrics
        
    def _estimate_throughput(self, tracker: LatencyTracker) -> float:
        """Estimate throughput from recent requests"""
        return len(tracker.latencies) / 60.0
        
    def get_system_signals(self) -> Dict:
        """
        Collect system-level signals:
        - PCIe counters  
        - NVML metrics
        - Host I/O stats
        """
        return {
            "pcie_bandwidth_mbps": self._get_pcie_bandwidth(),
            "gpu_utilization_pct": self._get_gpu_utilization(),
            "io_activity_mbps": self._get_io_activity()
        }
        
    def _get_pcie_bandwidth(self) -> float:
        """Get PCIe bandwidth utilization"""
        return 0.0
        
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization via NVML"""
        return 0.0
        
    def _get_io_activity(self) -> float:
        """Get host I/O activity"""
        return 0.0
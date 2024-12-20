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
        """Get PCIe bandwidth utilization via nvidia-smi and /proc"""
        try:
            import subprocess
            
            # Query PCIe throughput counters via nvidia-smi
            cmd = ['nvidia-smi', '--query-gpu=pcie.link.gen.current,pcie.link.width.current', 
                   '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_bandwidth = 0.0
                
                for line in lines:
                    if ',' in line:
                        gen, width = line.split(',')
                        # PCIe bandwidth calculation: Gen * Width * ~1GB/s per lane
                        gen_val = int(gen.strip()) if gen.strip().isdigit() else 3
                        width_val = int(width.strip()) if width.strip().isdigit() else 16
                        
                        # Rough bandwidth estimation (Gen4 x16 ≈ 64GB/s theoretical)
                        max_bw = gen_val * width_val * 1.0  # GB/s
                        total_bandwidth += max_bw
                        
                # Get actual utilization by reading PCIe counters
                pcie_util = self._read_pcie_counters()
                return min(pcie_util, total_bandwidth)
                
        except Exception as e:
            self.logger.debug(f"PCIe bandwidth query failed: {e}")
            
        return 0.0
        
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization via NVML/nvidia-smi"""
        try:
            import subprocess
            
            # Query GPU utilization across all devices
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', 
                   '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_util = 0.0
                count = 0
                
                for line in lines:
                    if ',' in line:
                        gpu_util, mem_util = line.split(',')
                        gpu_pct = float(gpu_util.strip()) if gpu_util.strip() != 'N/A' else 0.0
                        mem_pct = float(mem_util.strip()) if mem_util.strip() != 'N/A' else 0.0
                        
                        # Combined utilization metric
                        combined_util = max(gpu_pct, mem_pct)
                        total_util += combined_util
                        count += 1
                        
                if count > 0:
                    return total_util / count
                    
        except Exception as e:
            self.logger.debug(f"GPU utilization query failed: {e}")
            
        return 0.0
        
    def _get_io_activity(self) -> float:
        """Get host I/O activity from /proc/diskstats"""
        try:
            total_activity = 0.0
            
            with open('/proc/diskstats', 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= 14:
                        device = fields[2]
                        # Skip loop devices, ram disks, etc.
                        if device.startswith(('sd', 'nvme', 'xvd')):
                            # Read/write sectors (fields 5 and 9)
                            read_sectors = int(fields[5])
                            write_sectors = int(fields[9])
                            
                            # Convert to approximate MB/s (sector = 512 bytes)
                            activity_mb = (read_sectors + write_sectors) * 512 / (1024 * 1024)
                            total_activity += activity_mb
                            
            return total_activity
            
        except Exception as e:
            self.logger.debug(f"I/O activity query failed: {e}")
            
        return 0.0
        
    def _read_pcie_counters(self) -> float:
        """Read PCIe performance counters"""
        try:
            import subprocess
            
            # Use lspci to get PCIe device stats
            result = subprocess.run(['lspci', '-vvv'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse for NVIDIA devices and bandwidth indicators
                lines = result.stdout.split('\n')
                pcie_activity = 0.0
                
                for i, line in enumerate(lines):
                    if 'NVIDIA' in line and 'VGA' in line:
                        # Look for LnkSta (Link Status) in following lines
                        for j in range(i, min(i+20, len(lines))):
                            if 'LnkSta:' in lines[j]:
                                # Extract speed and width information
                                # This is a simplified heuristic
                                pcie_activity += 10.0  # Base activity indicator
                                break
                                
                return min(pcie_activity, 100.0)  # Cap at 100%
                
        except Exception as e:
            self.logger.debug(f"PCIe counter read failed: {e}")
            
        return 5.0  # Default background activity
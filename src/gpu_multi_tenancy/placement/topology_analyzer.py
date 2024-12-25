"""
PCIe Topology Analysis

Core placement scoring algorithm for topology-aware tenant placement.
Implements the weighted scoring heuristic from the controller's decision logic.
"""

import subprocess
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PCIeDevice:
    bus_id: str
    numa_node: int
    root_complex: str
    bandwidth_gbps: float
    current_load: float


class PCIeTopologyAnalyzer:
    """
    Core algorithm: PCIe topology-aware placement scoring
    
    Scores candidate GPU devices based on:
    1. PCIe root complex contention  
    2. Current bandwidth utilization
    3. NUMA domain locality
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device_cache: Dict[str, PCIeDevice] = {}
        self.topology_map: Dict[str, List[str]] = {}
        
    def score_device_placement(self, device_id: str, tenant_id: str) -> float:
        """
        Core placement scoring algorithm
        
        Returns normalized score (0.0 = best, 1.0 = worst)
        Lower scores indicate better placement candidates
        """
        try:
            device = self._get_device_info(device_id)
            if not device:
                return 1.0  # Worst score for unknown devices
                
            # Component 1: Root complex contention penalty
            rc_penalty = self._calculate_root_complex_penalty(device.root_complex)
            
            # Component 2: Current bandwidth utilization penalty  
            bw_penalty = min(device.current_load / 100.0, 1.0)
            
            # Component 3: NUMA locality bonus (inverse penalty)
            numa_penalty = self._calculate_numa_penalty(device.numa_node, tenant_id)
            
            # Weighted combination of topology factors
            total_score = (rc_penalty * 0.4 +  
                          bw_penalty * 0.4 + 
                          numa_penalty * 0.2)
                          
            self.logger.debug(f"Device {device_id} score: RC={rc_penalty:.2f}, "
                            f"BW={bw_penalty:.2f}, NUMA={numa_penalty:.2f}, "
                            f"Total={total_score:.2f}")
            
            return min(total_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Placement scoring failed for {device_id}: {e}")
            return 0.8  # Conservative fallback
            
    def _get_device_info(self, device_id: str) -> Optional[PCIeDevice]:
        """Get PCIe device topology information"""
        if device_id in self.device_cache:
            return self.device_cache[device_id]
            
        try:
            # Parse lspci output for device topology
            lspci_cmd = ['lspci', '-tv', '-s', device_id]
            result = subprocess.run(lspci_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
                
            # Parse topology information
            numa_node = self._parse_numa_node(device_id)
            root_complex = self._parse_root_complex(result.stdout)
            bandwidth = self._get_link_bandwidth(device_id)
            current_load = self._get_current_utilization(device_id)
            
            device = PCIeDevice(
                bus_id=device_id,
                numa_node=numa_node,
                root_complex=root_complex,
                bandwidth_gbps=bandwidth,
                current_load=current_load
            )
            
            self.device_cache[device_id] = device
            return device
            
        except Exception as e:
            self.logger.error(f"Failed to get device info for {device_id}: {e}")
            return None
            
    def _calculate_root_complex_penalty(self, root_complex: str) -> float:
        """
        Calculate penalty for root complex contention
        
        Higher penalty when multiple high-bandwidth devices 
        share the same PCIe root complex
        """
        if root_complex not in self.topology_map:
            return 0.1  # Low penalty for uncrowded complex
            
        # Count active devices on this root complex
        active_devices = len(self.topology_map[root_complex])
        
        if active_devices <= 2:
            return 0.1  # Low contention
        elif active_devices <= 4:
            return 0.5  # Moderate contention  
        else:
            return 0.9  # High contention
            
    def _calculate_numa_penalty(self, numa_node: int, tenant_id: str) -> float:
        """
        Calculate NUMA locality penalty
        
        Lower penalty when tenant processes are on same NUMA node
        """
        try:
            # Get tenant's current NUMA affinity (simplified)
            tenant_numa = self._get_tenant_numa_node(tenant_id)
            
            if tenant_numa == numa_node:
                return 0.0  # Perfect locality
            elif abs(tenant_numa - numa_node) == 1:
                return 0.3  # Adjacent NUMA node
            else:
                return 0.7  # Remote NUMA node
                
        except:
            return 0.4  # Default moderate penalty
            
    def _parse_numa_node(self, device_id: str) -> int:
        """Parse NUMA node for PCIe device"""
        try:
            numa_path = f"/sys/bus/pci/devices/{device_id.replace(':', '_')}/numa_node"
            with open(numa_path, 'r') as f:
                return int(f.read().strip())
        except:
            return 0  # Default to node 0
            
    def _parse_root_complex(self, lspci_output: str) -> str:
        """Extract root complex identifier from lspci"""
        lines = lspci_output.split('\n')
        for line in lines:
            if 'Root Port' in line or 'Root Complex' in line:
                # Extract root complex identifier
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        return "unknown"
        
    def _get_link_bandwidth(self, device_id: str) -> float:
        """Get PCIe link bandwidth capacity"""
        try:
            cmd = ['nvidia-smi', '--query-gpu=pcie.link.width.current,pcie.link.gen.current',
                   '--format=csv,noheader,nounits', '--id', device_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                width, gen = result.stdout.strip().split(',')
                # PCIe bandwidth estimation: Gen * Width * ~1GB/s per lane
                return float(gen) * float(width) * 1.0
                
        except:
            pass
        return 64.0  # Default PCIe 4.0 x16
        
    def _get_current_utilization(self, device_id: str) -> float:
        """Get current GPU utilization percentage"""
        try:
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu',
                   '--format=csv,noheader,nounits', '--id', device_id]  
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
                
        except:
            pass
        return 25.0  # Conservative default
        
    def _get_tenant_numa_node(self, tenant_id: str) -> int:
        """Get tenant's current NUMA node affinity"""
        # Simplified: return node 0 for demo
        # Real implementation would track tenant process locations
        return 0
        
    def get_topology_summary(self) -> Dict:
        """Get current topology state for debugging"""
        return {
            'devices': len(self.device_cache),
            'root_complexes': len(self.topology_map),
            'cache_hits': sum(1 for _ in self.device_cache.values())
        }
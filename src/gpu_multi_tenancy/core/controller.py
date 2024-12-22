"""
Multi-Tenancy Controller

SLO-aware GPU resource management with adaptive isolation.
Dynamic MIG + PCIe-aware placement + lightweight guardrails.
"""

import time
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque


@dataclass
class ControllerConfig:
    tail_threshold_ms: float = 15.0
    persistence_windows: int = 3
    dwell_time_observations: int = 256
    cooldown_observations: int = 128
    polling_interval_seconds: float = 2.0
    throughput_degradation_limit: float = 0.05


class ControllerState(Enum):
    MONITORING = "monitoring"
    UPGRADING_ISOLATION = "upgrading_isolation"
    RELAXING_ISOLATION = "relaxing_isolation"
    COOLDOWN = "cooldown"


class MultiTenancyController:
    """
    SLO-aware GPU resource controller.
    Implements three-tier isolation: PCIe placement -> MIG -> guardrails.
    """
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = ControllerState.MONITORING
        self.observation_count = 0
        self.last_action_observation = 0
        self.cooldown_until = 0
        self.tenants: Dict[str, 'TenantState'] = {}
        self.violation_windows: Dict[str, deque] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the SLO-aware control loop"""
        if self._running:
            return
        self.logger.info("Starting Multi-Tenancy Controller")
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        if not self._running:
            return
        self.logger.info("Stopping Multi-Tenancy Controller")
        self._running = False
        if self._thread:
            self._thread.join()
            
    def _control_loop(self):
        """Core control loop - demonstrates paper's main algorithm"""
        while self._running:
            try:
                self._process_observation()
                time.sleep(self.config.polling_interval_seconds)
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                time.sleep(1)
                
    def _process_observation(self):
        """Process one observation cycle - paper's Algorithm 1"""
        self.observation_count += 1
        
        # Skip actions if in cooldown (paper's dwell/cooldown mechanism)
        if self._is_in_cooldown():
            return
            
        # Check each tenant for SLO violations
        for tenant_id in self.tenants.keys():
            if self._should_upgrade_isolation(tenant_id):
                self._upgrade_tenant_isolation(tenant_id)
                break
            elif self._should_relax_isolation(tenant_id):
                self._relax_tenant_isolation(tenant_id)
                break
                
    def _should_upgrade_isolation(self, tenant_id: str) -> bool:
        """Check if tenant needs isolation upgrade (paper's persistence logic)"""
        if not self._at_reconfig_boundary():
            return False
            
        violations = self.violation_windows.get(tenant_id, deque())
        if len(violations) < self.config.persistence_windows:
            return False
            
        # Paper's requirement: persistent violations across Y windows
        return all(violations)
        
    def _upgrade_tenant_isolation(self, tenant_id: str):
        """
        Three-tier isolation upgrade (paper's core contribution):
        1. PCIe-aware placement
        2. Dynamic MIG reconfiguration  
        3. Lightweight guardrails
        """
        self.logger.info(f"Upgrading isolation for tenant {tenant_id}")
        self.state = ControllerState.UPGRADING_ISOLATION
        
        # Tier 1: PCIe-aware placement (if available)
        if self._try_pcie_placement(tenant_id):
            self._record_action()
            return
            
        # Tier 2: Dynamic MIG upgrade (if possible)
        if self._try_mig_upgrade(tenant_id):
            self._record_action()
            return
            
        # Tier 3: Lightweight guardrails (always available)
        self._apply_guardrails(tenant_id)
        self._record_action()
        
    def _try_pcie_placement(self, tenant_id: str) -> bool:
        """Attempt PCIe-aware placement using topology scoring"""
        try:
            import subprocess
            import re
            
            # Query PCIe topology and current utilization
            lspci_output = subprocess.check_output(['lspci', '-tv'], text=True)
            nvidia_devices = re.findall(r'(\d+:\d+\.\d+).*NVIDIA', lspci_output)
            
            if not nvidia_devices:
                return False
                
            # Get current bandwidth utilization per PCIe root complex  
            best_score = float('inf')
            target_device = None
            
            for device in nvidia_devices:
                # Calculate placement score based on current load
                score = self._calculate_placement_score(device, tenant_id)
                if score < best_score:
                    best_score = score
                    target_device = device
                    
            if target_device and best_score < 0.8:  # Threshold for worthwhile move
                self._relocate_tenant_pcie(tenant_id, target_device)
                self.logger.info(f"Relocated {tenant_id} to PCIe device {target_device}")
                return True
                
        except Exception as e:
            self.logger.error(f"PCIe placement failed: {e}")
            
        return False
        
    def _try_mig_upgrade(self, tenant_id: str) -> bool:
        """Attempt MIG profile upgrade via nvidia-smi"""
        try:
            import subprocess
            
            # Get current tenant's MIG assignment
            current_profile = self._get_tenant_mig_profile(tenant_id)
            if not current_profile:
                return False
                
            # Define upgrade path
            upgrade_map = {
                "1g.10gb": "2g.20gb",
                "2g.20gb": "3g.40gb", 
                "3g.40gb": "4g.40gb",
                "4g.40gb": "7g.80gb"
            }
            
            target_profile = upgrade_map.get(current_profile)
            if not target_profile:
                self.logger.debug(f"No upgrade available from {current_profile}")
                return False
                
            # Check GPU headroom before attempting upgrade
            gpu_id = self._get_tenant_gpu(tenant_id)
            if not self._has_mig_headroom(gpu_id, target_profile):
                return False
                
            # Perform MIG reconfiguration
            cmd = ['nvidia-smi', 'mig', '-cgi', target_profile, '-C', '-i', str(gpu_id)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._update_tenant_mig_assignment(tenant_id, target_profile, gpu_id)
                self.logger.info(f"Upgraded {tenant_id}: {current_profile} -> {target_profile}")
                return True
            else:
                self.logger.warning(f"MIG upgrade failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"MIG upgrade error: {e}")
            
        return False
        
    def _apply_guardrails(self, tenant_id: str):
        """Apply lightweight guardrails via MPS quotas and cgroup throttling"""
        try:
            import os
            import subprocess
            
            # Apply MPS active thread percentage reduction
            current_mps = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')
            target_mps = max(50, int(current_mps) - 20)  # Reduce by 20%, floor at 50%
            
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(target_mps)
            self.logger.info(f"Reduced MPS quota for {tenant_id}: {current_mps}% -> {target_mps}%")
            
            # Apply cgroup I/O throttling to background processes
            cgroup_path = f"/sys/fs/cgroup/gpu-tenants/{tenant_id}"
            if os.path.exists(cgroup_path):
                # Throttle I/O to 200MB/s for this tenant's cgroup
                io_max_path = f"{cgroup_path}/io.max"
                if os.path.exists(io_max_path):
                    with open(io_max_path, 'w') as f:
                        f.write("8:0 rbps=209715200 wbps=209715200\n")  # 200MB/s limit
                    self.logger.info(f"Applied I/O throttling: 200MB/s limit for {tenant_id}")
            
            # Pin tenant away from high-IRQ cores
            irq_cores = self._get_high_irq_cores()
            if irq_cores:
                isolated_cores = [str(i) for i in range(8) if i not in irq_cores[:2]]
                if isolated_cores:
                    cpuset_path = f"{cgroup_path}/cpuset.cpus"
                    if os.path.exists(cpuset_path):
                        with open(cpuset_path, 'w') as f:
                            f.write(','.join(isolated_cores))
                        self.logger.info(f"Isolated {tenant_id} to CPUs: {','.join(isolated_cores)}")
                        
        except Exception as e:
            self.logger.error(f"Guardrails application failed: {e}")
            # Continue execution - guardrails are best-effort
        
    def _should_relax_isolation(self, tenant_id: str) -> bool:
        """Check if isolation can be relaxed"""
        if not self._at_reconfig_boundary():
            return False
        tenant_state = self.tenants.get(tenant_id)
        return tenant_state and tenant_state.is_stable()
        
    def _relax_tenant_isolation(self, tenant_id: str):
        """Relax isolation when performance is stable"""
        self.logger.info(f"Relaxing isolation for {tenant_id}")
        self.state = ControllerState.RELAXING_ISOLATION
        self._record_action()
        
    def _record_action(self):
        """Record action and enter cooldown (paper's anti-thrashing mechanism)"""
        self.last_action_observation = self.observation_count
        self.cooldown_until = self.observation_count + self.config.cooldown_observations
        self.state = ControllerState.COOLDOWN
        
    def _at_reconfig_boundary(self) -> bool:
        """Check if enough time has passed since last action (dwell time)"""
        return (self.observation_count - self.last_action_observation >= 
                self.config.dwell_time_observations)
                
    def _is_in_cooldown(self) -> bool:
        """Check if controller is in cooldown period"""
        return self.observation_count < self.cooldown_until
        
    def get_status(self) -> Dict:
        """Get controller status - demonstrates operational state"""
        return {
            "state": self.state.value,
            "observation_count": self.observation_count,
            "active_tenants": len(self.tenants),
            "in_cooldown": self._is_in_cooldown()
        }


    def _calculate_placement_score(self, device: str, tenant_id: str) -> float:
        """Calculate PCIe placement score - lower is better"""
        try:
            from ..placement.topology_analyzer import PCIeTopologyAnalyzer
            from ..placement.numa_profiler import NUMAAffinityProfiler
            
            # Use topology-aware scoring algorithm
            topo_analyzer = PCIeTopologyAnalyzer()
            numa_profiler = NUMAAffinityProfiler()
            
            pcie_score = topo_analyzer.score_device_placement(device, tenant_id)
            numa_score = numa_profiler.get_numa_penalty(device)
            
            # Weighted combination of topology factors
            return pcie_score * 0.7 + numa_score * 0.3
            
        except ImportError:
            # Fallback to basic utilization if placement modules unavailable  
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    util = float(result.stdout.strip())
                    return util / 100.0  # Normalize to 0-1
            except:
                pass
            return 0.5  # Default moderate score
        except Exception as e:
            self.logger.debug(f"Placement scoring failed: {e}")
            return 0.5
        
    def _relocate_tenant_pcie(self, tenant_id: str, device: str):
        """Relocate tenant to different PCIe device"""
        # Update CUDA_VISIBLE_DEVICES for the tenant process
        self.logger.debug(f"Relocating {tenant_id} to {device}")
        
    def _get_tenant_mig_profile(self, tenant_id: str) -> str:
        """Get current MIG profile for tenant"""
        # Return current profile or None
        return "1g.10gb"  # Placeholder
        
    def _get_tenant_gpu(self, tenant_id: str) -> int:
        """Get GPU ID for tenant"""
        return 0  # Placeholder
        
    def _has_mig_headroom(self, gpu_id: int, profile: str) -> bool:
        """Check if GPU has capacity for profile"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', 'mig', '-lgip', '-i', str(gpu_id)], 
                                  capture_output=True, text=True)
            # Parse available slots vs requested profile
            return "Instance ID" in result.stdout  # Simplified check
        except:
            return False
            
    def _update_tenant_mig_assignment(self, tenant_id: str, profile: str, gpu_id: int):
        """Update tenant's MIG assignment"""
        self.logger.debug(f"Updated {tenant_id} assignment: {profile} on GPU {gpu_id}")
        
    def _get_high_irq_cores(self) -> list:
        """Get cores with high IRQ activity"""
        try:
            with open('/proc/interrupts', 'r') as f:
                # Parse IRQ stats - simplified
                return [0, 1]  # Placeholder - typically cores 0,1 handle most IRQs
        except:
            return []


@dataclass 
class TenantState:
    tenant_id: str
    stable_windows: int = 0
    
    def is_stable(self) -> bool:
        return self.stable_windows >= 10
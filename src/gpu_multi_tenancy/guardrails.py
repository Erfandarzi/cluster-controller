"""
Guardrails Module  

MPS quota management and cgroup-based resource controls.
"""

import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass

# Import tenant process management utilities  
from .runtime.process_manager import TenantProcessTracker
from .runtime.mps_controller import MPSQuotaManager
from .utils.cgroup_utils import CgroupManager


@dataclass
class GuardrailConfig:
    mps_min_quota: int = 50
    mps_step_size: int = 20  
    io_throttle_mbps: int = 200
    cpu_isolation_enabled: bool = True


class GuardrailManager:
    """Lightweight resource isolation via MPS and cgroups"""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.process_tracker = TenantProcessTracker()
        self.mps_controller = MPSQuotaManager()
        self.cgroup_manager = CgroupManager()
        self.tenant_states: Dict[str, Dict] = {}
        
    def apply_mps_quota(self, tenant_id: str, quota_pct: int) -> bool:
        """Set MPS active thread percentage for tenant"""
        try:
            pids = self.process_tracker.get_tenant_pids(tenant_id)
            if not pids:
                self.logger.warning(f"No processes found for tenant {tenant_id}")
                return False
                
            success = self.mps_controller.set_quota(tenant_id, quota_pct)
            if success:
                self.tenant_states[tenant_id] = {'mps_quota': quota_pct}
                self.logger.info(f"Applied MPS quota {quota_pct}% to {tenant_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"MPS quota setup failed: {e}")
            return False
        
    def reduce_mps_quota(self, tenant_id: str) -> bool:
        """Reduce MPS quota by configured step size"""
        current_quota = self.tenant_states.get(tenant_id, {}).get('mps_quota', 100)
        new_quota = max(self.config.mps_min_quota, current_quota - self.config.mps_step_size)
        return self.apply_mps_quota(tenant_id, new_quota)
        
    def apply_io_throttling(self, tenant_id: str, limit_mbps: int) -> bool:
        """Apply cgroup I/O bandwidth throttling"""
        try:
            success = self.cgroup_manager.create_tenant_cgroup(tenant_id)
            if not success:
                return False
                
            success = self.cgroup_manager.set_io_limits(tenant_id, limit_mbps)
            if success:
                pids = self.process_tracker.get_tenant_pids(tenant_id)
                self.cgroup_manager.assign_processes(tenant_id, pids)
                self.logger.info(f"Applied I/O throttling: {limit_mbps}MB/s for {tenant_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"I/O throttling failed: {e}")
            return False
            
    def isolate_cpu_cores(self, tenant_id: str, excluded_cores: list) -> bool:
        """Isolate tenant to specific CPU cores"""
        if not self.config.cpu_isolation_enabled:
            return False
            
        try:
            return self.cgroup_manager.set_cpu_affinity(tenant_id, excluded_cores)
        except Exception as e:
            self.logger.error(f"CPU isolation failed: {e}")
            return False
            
    def remove_guardrails(self, tenant_id: str):
        """Remove all guardrails for tenant"""
        try:
            self.mps_controller.reset_quota(tenant_id)  
            self.cgroup_manager.remove_limits(tenant_id)
            self.tenant_states.pop(tenant_id, None)
            self.logger.info(f"Removed guardrails for {tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Guardrail removal failed: {e}")

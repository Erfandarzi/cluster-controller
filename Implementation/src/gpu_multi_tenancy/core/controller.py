"""
Multi-Tenancy Controller

Central control loop that demonstrates the paper's core argument:
Dynamic MIG + PCIe-aware placement + guardrails for SLO compliance
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
    Main controller implementing the paper's core argument:
    SLO-aware dynamic resource management with three-tier approach
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
        """Attempt PCIe-aware placement - core algorithm details omitted"""
        # Implementation exists but topology scoring details withheld
        self.logger.debug(f"Attempting PCIe placement for {tenant_id}")
        return False  # Simplified for protection
        
    def _try_mig_upgrade(self, tenant_id: str) -> bool:
        """Attempt MIG profile upgrade - demonstrates concept"""
        self.logger.debug(f"Attempting MIG upgrade for {tenant_id}")
        # Basic MIG logic present but resource management details omitted
        return False  # Simplified for protection
        
    def _apply_guardrails(self, tenant_id: str):
        """Apply lightweight guardrails - shows third tier"""
        self.logger.info(f"Applying guardrails for {tenant_id}")
        # MPS quota and cgroup throttling logic exists but parameters withheld
        pass
        
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


@dataclass 
class TenantState:
    tenant_id: str
    stable_windows: int = 0
    
    def is_stable(self) -> bool:
        return self.stable_windows >= 10
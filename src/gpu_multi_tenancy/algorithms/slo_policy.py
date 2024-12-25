"""
SLO Policy Engine

Core algorithm implementing the paper's persistence and dwell-time logic.
Handles the decision FSM for when to trigger isolation changes.
"""

import time
import logging
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass
from collections import deque
from enum import Enum


class PolicyDecision(Enum):
    NO_ACTION = "no_action"
    UPGRADE_ISOLATION = "upgrade_isolation"  
    RELAX_ISOLATION = "relax_isolation"


@dataclass
class SLOMetrics:
    tenant_id: str
    timestamp: float
    p99_latency_ms: float
    slo_threshold_ms: float
    is_violation: bool
    
    
class SLOPolicyEngine:
    """
    Core Algorithm: SLO violation detection with persistence logic
    
    Implements the paper's key policy decisions:
    1. Persistent violation detection across Y windows
    2. Dwell time enforcement to prevent thrashing  
    3. Cooldown periods after actions
    4. Stability detection for relaxation
    """
    
    def __init__(self, persistence_windows: int = 3, 
                 dwell_observations: int = 256,
                 cooldown_observations: int = 128):
        self.persistence_windows = persistence_windows
        self.dwell_observations = dwell_observations
        self.cooldown_observations = cooldown_observations
        
        self.logger = logging.getLogger(__name__)
        
        # State tracking per tenant
        self.violation_history: Dict[str, Deque[bool]] = {}
        self.last_action_time: Dict[str, int] = {}
        self.cooldown_until: Dict[str, int] = {}
        self.observation_count = 0
        
        self.logger.info(f"SLO Policy: persistence={persistence_windows}, "
                        f"dwell={dwell_observations}, cooldown={cooldown_observations}")
        
    def evaluate_policy(self, metrics: SLOMetrics) -> PolicyDecision:
        """
        Core policy decision algorithm
        
        Returns the action to take based on SLO metrics and internal state.
        Implements Algorithm 1 from the paper.
        """
        self.observation_count += 1
        tenant_id = metrics.tenant_id
        
        # Initialize tenant tracking if needed
        if tenant_id not in self.violation_history:
            self.violation_history[tenant_id] = deque(maxlen=self.persistence_windows)
            self.last_action_time[tenant_id] = 0
            self.cooldown_until[tenant_id] = 0
            
        # Record current violation state  
        self.violation_history[tenant_id].append(metrics.is_violation)
        
        self.logger.debug(f"Tenant {tenant_id}: p99={metrics.p99_latency_ms:.1f}ms, "
                         f"violation={metrics.is_violation}, "
                         f"history={list(self.violation_history[tenant_id])}")
        
        # Check if we're in cooldown period
        if self._is_in_cooldown(tenant_id):
            self.logger.debug(f"Tenant {tenant_id} in cooldown until observation {self.cooldown_until[tenant_id]}")
            return PolicyDecision.NO_ACTION
            
        # Check if we're within dwell time from last action
        if not self._at_reconfig_boundary(tenant_id):
            self.logger.debug(f"Tenant {tenant_id} within dwell time")
            return PolicyDecision.NO_ACTION
            
        # Decision logic: Check for persistent violations
        if self._has_persistent_violations(tenant_id):
            self._record_action(tenant_id)
            self.logger.info(f"Policy decision: UPGRADE isolation for {tenant_id}")
            return PolicyDecision.UPGRADE_ISOLATION
            
        # Decision logic: Check for stability (relaxation candidate)  
        elif self._is_stable_for_relaxation(tenant_id):
            self._record_action(tenant_id)
            self.logger.info(f"Policy decision: RELAX isolation for {tenant_id}")
            return PolicyDecision.RELAX_ISOLATION
            
        return PolicyDecision.NO_ACTION
        
    def _has_persistent_violations(self, tenant_id: str) -> bool:
        """
        Core persistence algorithm
        
        Returns True if violations persist across Y consecutive windows.
        This is the key algorithm preventing spurious reactions.
        """
        history = self.violation_history[tenant_id]
        
        # Need full window history
        if len(history) < self.persistence_windows:
            return False
            
        # Check if ALL recent windows show violations
        persistent = all(history)
        
        if persistent:
            self.logger.info(f"Persistent violations detected for {tenant_id}: "
                           f"{list(history)}")
        
        return persistent
        
    def _is_stable_for_relaxation(self, tenant_id: str) -> bool:
        """
        Stability detection for isolation relaxation
        
        Returns True if tenant has been consistently under SLO
        for sufficient time to warrant resource reduction.
        """
        history = self.violation_history[tenant_id]
        
        # Need full window history
        if len(history) < self.persistence_windows:
            return False
            
        # Check if NO recent windows show violations
        stable = not any(history)
        
        # Additional stability requirement: longer observation period
        stable_observations = self.observation_count - self.last_action_time[tenant_id]
        sufficient_stability = stable_observations >= (self.dwell_observations * 2)
        
        if stable and sufficient_stability:
            self.logger.info(f"Stability detected for {tenant_id}: "
                           f"no violations for {stable_observations} observations")
            return True
            
        return False
        
    def _at_reconfig_boundary(self, tenant_id: str) -> bool:
        """
        Dwell time enforcement
        
        Prevents actions until sufficient observations have passed
        since the last reconfiguration. Critical anti-thrashing mechanism.
        """
        time_since_action = self.observation_count - self.last_action_time[tenant_id]
        at_boundary = time_since_action >= self.dwell_observations
        
        if not at_boundary:
            self.logger.debug(f"Tenant {tenant_id}: {time_since_action}/{self.dwell_observations} "
                            "observations since last action")
        
        return at_boundary
        
    def _is_in_cooldown(self, tenant_id: str) -> bool:
        """Check if tenant is in post-action cooldown period"""
        return self.observation_count < self.cooldown_until[tenant_id]
        
    def _record_action(self, tenant_id: str):
        """
        Record action and set cooldown period
        
        Updates internal state to enforce dwell time and cooldown.
        Critical for preventing oscillations.
        """
        self.last_action_time[tenant_id] = self.observation_count
        self.cooldown_until[tenant_id] = self.observation_count + self.cooldown_observations
        
        self.logger.info(f"Action recorded for {tenant_id}: "
                        f"cooldown until observation {self.cooldown_until[tenant_id]}")
        
    def get_policy_state(self) -> Dict:
        """Get current policy engine state for debugging"""
        return {
            'observation_count': self.observation_count,
            'tracked_tenants': len(self.violation_history),
            'tenants_in_cooldown': sum(1 for tid in self.violation_history.keys() 
                                     if self._is_in_cooldown(tid)),
            'total_actions': sum(1 for t in self.last_action_time.values() if t > 0)
        }
        
    def reset_tenant(self, tenant_id: str):
        """Reset tracking state for tenant (e.g., when tenant leaves)"""
        self.violation_history.pop(tenant_id, None)
        self.last_action_time.pop(tenant_id, None) 
        self.cooldown_until.pop(tenant_id, None)
        self.logger.info(f"Reset policy state for {tenant_id}")
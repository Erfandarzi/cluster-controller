"""
MIG Scheduling Algorithm

Core algorithm for dynamic MIG profile management and resource allocation.
Implements the upgrade/downgrade logic with resource constraint checking.
"""

import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MIGProfile(Enum):
    """Standard A100 MIG profiles"""
    MIG_1g_10gb = "1g.10gb"
    MIG_2g_20gb = "2g.20gb" 
    MIG_3g_40gb = "3g.40gb"
    MIG_4g_40gb = "4g.40gb"
    MIG_7g_80gb = "7g.80gb"
    
    @property
    def compute_units(self) -> int:
        """Get compute unit count"""
        return int(self.value.split('g.')[0])
        
    @property
    def memory_gb(self) -> int:
        """Get memory size in GB"""
        return int(self.value.split('.')[1].replace('gb', ''))


@dataclass 
class MIGAllocation:
    tenant_id: str
    gpu_id: int
    profile: MIGProfile
    instance_id: str
    utilization: float = 0.0


class MIGScheduler:
    """
    Core Algorithm: Dynamic MIG resource allocation
    
    MIG upgrade/downgrade logic:
    1. Resource constraint checking before allocation
    2. Optimal profile selection based on demand
    3. GPU capacity management across tenants
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_allocations: Dict[str, MIGAllocation] = {}  # tenant_id -> allocation
        self.gpu_capacity: Dict[int, Dict[str, int]] = {}  # gpu_id -> {profile -> available_slots}
        
        # MIG upgrade path sequence  
        self.upgrade_path = {
            MIGProfile.MIG_1g_10gb: MIGProfile.MIG_2g_20gb,
            MIGProfile.MIG_2g_20gb: MIGProfile.MIG_3g_40gb,
            MIGProfile.MIG_3g_40gb: MIGProfile.MIG_4g_40gb,
            MIGProfile.MIG_4g_40gb: MIGProfile.MIG_7g_80gb
        }
        
        self._initialize_gpu_capacity()
        
    def can_upgrade_tenant(self, tenant_id: str) -> bool:
        """
        Check if tenant can be upgraded to next MIG profile
        
        Core constraint checking algorithm for resource availability.
        """
        if tenant_id not in self.current_allocations:
            return False
            
        allocation = self.current_allocations[tenant_id]
        current_profile = allocation.profile
        
        # Check if upgrade path exists
        if current_profile not in self.upgrade_path:
            self.logger.debug(f"No upgrade path from {current_profile}")
            return False
            
        target_profile = self.upgrade_path[current_profile]
        gpu_id = allocation.gpu_id
        
        # Check GPU capacity constraints
        if not self._has_capacity(gpu_id, target_profile):
            self.logger.debug(f"GPU {gpu_id} lacks capacity for {target_profile}")
            return False
            
        return True
        
    def upgrade_tenant_mig(self, tenant_id: str) -> bool:
        """
        Core MIG upgrade algorithm
        
        Performs the actual profile upgrade with resource management.
        Returns True if upgrade successful.
        """
        if not self.can_upgrade_tenant(tenant_id):
            return False
            
        try:
            allocation = self.current_allocations[tenant_id]
            current_profile = allocation.profile
            target_profile = self.upgrade_path[current_profile]
            gpu_id = allocation.gpu_id
            
            self.logger.info(f"Upgrading {tenant_id}: {current_profile.value} -> {target_profile.value}")
            
            # Step 1: Destroy current MIG instance
            if not self._destroy_mig_instance(allocation.instance_id):
                return False
                
            # Step 2: Create new larger MIG instance  
            new_instance_id = self._create_mig_instance(gpu_id, target_profile)
            if not new_instance_id:
                # Rollback: recreate original instance
                self._create_mig_instance(gpu_id, current_profile)
                return False
                
            # Step 3: Update allocation tracking
            allocation.profile = target_profile
            allocation.instance_id = new_instance_id
            
            # Step 4: Update capacity tracking
            self._update_capacity_tracking(gpu_id, current_profile, target_profile)
            
            self.logger.info(f"Successfully upgraded {tenant_id} to {target_profile.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"MIG upgrade failed for {tenant_id}: {e}")
            return False
            
    def downgrade_tenant_mig(self, tenant_id: str) -> bool:
        """
        MIG downgrade for resource relaxation
        
        Reverse of upgrade process - moves tenant to smaller profile.
        """
        if tenant_id not in self.current_allocations:
            return False
            
        try:
            allocation = self.current_allocations[tenant_id]
            current_profile = allocation.profile
            
            # Find downgrade target (reverse lookup)
            target_profile = None
            for source, target in self.upgrade_path.items():
                if target == current_profile:
                    target_profile = source
                    break
                    
            if not target_profile:
                self.logger.debug(f"No downgrade path from {current_profile}")
                return False
                
            gpu_id = allocation.gpu_id
            
            self.logger.info(f"Downgrading {tenant_id}: {current_profile.value} -> {target_profile.value}")
            
            # Similar process to upgrade but in reverse
            if not self._destroy_mig_instance(allocation.instance_id):
                return False
                
            new_instance_id = self._create_mig_instance(gpu_id, target_profile)
            if not new_instance_id:
                self._create_mig_instance(gpu_id, current_profile)
                return False
                
            allocation.profile = target_profile
            allocation.instance_id = new_instance_id
            
            self._update_capacity_tracking(gpu_id, current_profile, target_profile)
            
            self.logger.info(f"Successfully downgraded {tenant_id} to {target_profile.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"MIG downgrade failed for {tenant_id}: {e}")
            return False
            
    def _has_capacity(self, gpu_id: int, profile: MIGProfile) -> bool:
        """
        GPU capacity constraint checking
        
        Verifies that GPU has sufficient compute/memory resources
        for the requested profile.
        """
        if gpu_id not in self.gpu_capacity:
            return False
            
        # Check available slots for this profile
        available_slots = self.gpu_capacity[gpu_id].get(profile.value, 0)
        return available_slots > 0
        
    def _create_mig_instance(self, gpu_id: int, profile: MIGProfile) -> Optional[str]:
        """Create MIG instance via nvidia-smi"""
        try:
            cmd = ['nvidia-smi', 'mig', '-cgi', profile.value, '-i', str(gpu_id)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse instance ID from output
                # Simplified: return synthetic ID
                instance_id = f"MIG-{gpu_id}-{profile.value}-{hash(profile.value) % 1000}"
                self.logger.debug(f"Created MIG instance {instance_id}")
                return instance_id
            else:
                self.logger.error(f"MIG creation failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"MIG creation error: {e}")
            
        return None
        
    def _destroy_mig_instance(self, instance_id: str) -> bool:
        """Destroy MIG instance"""
        try:
            # Simplified: log destruction  
            self.logger.debug(f"Destroying MIG instance {instance_id}")
            return True
        except Exception as e:
            self.logger.error(f"MIG destruction error: {e}")
            return False
            
    def _initialize_gpu_capacity(self):
        """Initialize GPU capacity tracking"""
        try:
            # Query available GPUs
            cmd = ['nvidia-smi', '-L']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
                
                for gpu_id in range(gpu_count):
                    self.gpu_capacity[gpu_id] = {
                        MIGProfile.MIG_1g_10gb.value: 7,  # A100 can fit 7x 1g instances
                        MIGProfile.MIG_2g_20gb.value: 3,  # 3x 2g instances 
                        MIGProfile.MIG_3g_40gb.value: 2,  # 2x 3g instances
                        MIGProfile.MIG_4g_40gb.value: 1,  # 1x 4g instance
                        MIGProfile.MIG_7g_80gb.value: 1   # 1x 7g instance (full GPU)
                    }
                    
                self.logger.info(f"Initialized capacity tracking for {gpu_count} GPUs")
                
        except Exception as e:
            self.logger.error(f"GPU capacity initialization failed: {e}")
            
    def _update_capacity_tracking(self, gpu_id: int, old_profile: MIGProfile, new_profile: MIGProfile):
        """Update capacity tracking after profile change"""
        if gpu_id in self.gpu_capacity:
            # Free old profile slot
            self.gpu_capacity[gpu_id][old_profile.value] += 1
            # Consume new profile slot  
            self.gpu_capacity[gpu_id][new_profile.value] -= 1
            
    def get_allocation_summary(self) -> Dict:
        """Get current allocation state"""
        return {
            'total_tenants': len(self.current_allocations),
            'profile_distribution': {
                profile.value: sum(1 for alloc in self.current_allocations.values() 
                                 if alloc.profile == profile)
                for profile in MIGProfile
            },
            'gpu_utilization': {
                gpu_id: sum(capacity.values()) 
                for gpu_id, capacity in self.gpu_capacity.items()
            }
        }
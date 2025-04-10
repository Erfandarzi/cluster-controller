"""
MIG Manager

Handles dynamic MIG profile changes (1g.10gb -> 2g.20gb etc.) for A100/H100 GPUs.
"""

import logging
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class MIGProfile(Enum):
    """A100 MIG profiles"""
    MIG_1g_10gb = "1g.10gb"
    MIG_2g_20gb = "2g.20gb" 
    MIG_3g_40gb = "3g.40gb"
    MIG_4g_40gb = "4g.40gb"
    MIG_7g_80gb = "7g.80gb"
    
    @property
    def compute_slices(self) -> int:
        return int(self.value.split('g.')[0])
        
    @property 
    def memory_gb(self) -> int:
        return int(self.value.split('.')[1].replace('gb', ''))


@dataclass
class MIGInstance:
    gpu_id: int
    instance_id: int
    profile: MIGProfile
    uuid: str
    tenant_id: Optional[str] = None


class MIGManager:
    """Dynamic MIG reconfiguration for GPU multi-tenancy"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tenant_placements: Dict[str, MIGInstance] = {}
        self.gpu_instances: Dict[int, List[MIGInstance]] = {}
        
    def enable_mig(self, gpu_id: int) -> bool:
        """Enable MIG mode on GPU"""
        try:
            subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "-mig", "1"],
                check=True, capture_output=True
            )
            self.logger.info(f"Enabled MIG on GPU {gpu_id}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to enable MIG: {e}")
            return False
            
    def create_mig_instance(self, gpu_id: int, profile: MIGProfile) -> Optional[str]:
        """Create MIG instance with specified profile"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "mig", "-i", str(gpu_id), "-cgi", profile.value],
                capture_output=True, text=True, check=True
            )
            self.logger.info(f"Created {profile.value} instance on GPU {gpu_id}")
            return f"MIG-{gpu_id}-{profile.value}"
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"MIG creation failed: {e}")
            return None
            
    def can_upgrade_mig(self, tenant_id: str) -> bool:
        """Check if MIG upgrade is possible for tenant"""
        instance = self.tenant_placements.get(tenant_id)
        if not instance:
            return False
            
        current_profile = instance.profile
        return current_profile != MIGProfile.MIG_7g_80gb
        
    def upgrade_mig_profile(self, tenant_id: str) -> bool:
        """Upgrade tenant to larger MIG profile"""
        if not self.can_upgrade_mig(tenant_id):
            return False
            
        instance = self.tenant_placements[tenant_id]
        current = instance.profile
        
        upgrade_map = {
            MIGProfile.MIG_1g_10gb: MIGProfile.MIG_2g_20gb,
            MIGProfile.MIG_2g_20gb: MIGProfile.MIG_3g_40gb,
            MIGProfile.MIG_3g_40gb: MIGProfile.MIG_4g_40gb,
            MIGProfile.MIG_4g_40gb: MIGProfile.MIG_7g_80gb
        }
        
        target_profile = upgrade_map.get(current)
        if not target_profile:
            return False
            
        new_uuid = self.create_mig_instance(instance.gpu_id, target_profile)
        if new_uuid:
            instance.profile = target_profile
            instance.uuid = new_uuid
            self.logger.info(f"Upgraded {tenant_id}: {current.value} -> {target_profile.value}")
            return True
            
        return False
        
    def assign_tenant_to_instance(self, tenant_id: str, instance_uuid: str):
        """Assign tenant to MIG instance"""
        self.logger.info(f"Assigned {tenant_id} to MIG instance {instance_uuid}")
        
    def get_tenant_instance(self, tenant_id: str) -> Optional[MIGInstance]:
        """Get tenant's current MIG instance"""
        return self.tenant_placements.get(tenant_id)
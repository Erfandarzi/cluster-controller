"""
Guardrails Module

Implements lightweight isolation mechanisms:
- MPS quota management via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
- cgroup v2 I/O throttling and CPU affinity
- Process-level resource containment
"""

import os
import logging
import subprocess
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GuardrailConfig:
    mps_min_quota: int = 50
    mps_step_size: int = 20
    io_throttle_mbps: int = 200
    cpu_isolation_enabled: bool = True


class GuardrailManager:
    """Manages MPS quotas, cgroup throttling, and CPU isolation"""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tenant_states: Dict[str, Dict] = {}
        
    def apply_mps_quota(self, tenant_id: str, quota_pct: int) -> bool:
        """Set MPS active thread percentage for tenant"""
        try:
            # Get tenant process PIDs
            pids = self._get_tenant_pids(tenant_id)
            if not pids:
                self.logger.warning(f"No processes found for tenant {tenant_id}")
                return False
                
            # Set MPS quota via environment variable for new processes
            mps_env = f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={quota_pct}"
            
            # Update running processes via nvidia-cuda-mps-control
            try:
                cmd = f"echo 'set_default_active_thread_percentage {quota_pct}' | nvidia-cuda-mps-control"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"Applied MPS quota {quota_pct}% to {tenant_id}")
                    self.tenant_states[tenant_id] = {'mps_quota': quota_pct}
                    return True
                else:
                    self.logger.error(f"MPS control failed: {result.stderr}")
            except Exception as e:
                self.logger.error(f"MPS quota application failed: {e}")
                
        except Exception as e:
            self.logger.error(f"MPS quota setup failed: {e}")
            
        return False
        
    def reduce_mps_quota(self, tenant_id: str) -> bool:
        """Reduce MPS quota by configured step size"""
        current_quota = self.tenant_states.get(tenant_id, {}).get('mps_quota', 100)
        new_quota = max(self.config.mps_min_quota, current_quota - self.config.mps_step_size)
        return self.apply_mps_quota(tenant_id, new_quota)
        
    def apply_io_throttling(self, tenant_id: str, limit_mbps: int) -> bool:
        """Apply cgroup v2 I/O throttling"""
        try:
            cgroup_path = f"/sys/fs/cgroup/gpu-tenants/{tenant_id}"
            
            # Create cgroup if it doesn't exist
            os.makedirs(cgroup_path, exist_ok=True)
            
            # Get major:minor for primary storage device
            storage_dev = self._get_primary_storage_device()
            if not storage_dev:
                return False
                
            # Set read/write bandwidth limits
            limit_bytes = limit_mbps * 1024 * 1024  # Convert MB/s to bytes/s
            throttle_rule = f"{storage_dev} rbps={limit_bytes} wbps={limit_bytes}\n"
            
            io_max_path = f"{cgroup_path}/io.max"
            with open(io_max_path, 'w') as f:
                f.write(throttle_rule)
                
            # Move tenant processes to the cgroup
            procs_path = f"{cgroup_path}/cgroup.procs"
            pids = self._get_tenant_pids(tenant_id)
            
            for pid in pids:
                try:
                    with open(procs_path, 'w') as f:
                        f.write(str(pid))
                except ProcessLookupError:
                    continue  # Process may have exited
                    
            self.logger.info(f"Applied I/O throttling: {limit_mbps}MB/s for {tenant_id}")
            return True
            
        except PermissionError:
            self.logger.error(f"Permission denied writing to cgroup (need root)")
            return False
        except Exception as e:
            self.logger.error(f"I/O throttling failed: {e}")
            return False
            
    def isolate_cpu_cores(self, tenant_id: str, excluded_cores: list) -> bool:
        """Isolate tenant to specific CPU cores"""
        if not self.config.cpu_isolation_enabled:
            return False
            
        try:
            cgroup_path = f"/sys/fs/cgroup/gpu-tenants/{tenant_id}"
            os.makedirs(cgroup_path, exist_ok=True)
            
            # Get total CPU count
            total_cores = os.cpu_count()
            allowed_cores = [i for i in range(total_cores) if i not in excluded_cores]
            
            if not allowed_cores:
                self.logger.warning("No cores available for isolation")
                return False
                
            # Write allowed cores to cpuset
            cpuset_path = f"{cgroup_path}/cpuset.cpus"
            core_list = ','.join(map(str, allowed_cores))
            
            with open(cpuset_path, 'w') as f:
                f.write(core_list)
                
            # Move processes to isolated cores
            procs_path = f"{cgroup_path}/cgroup.procs"
            pids = self._get_tenant_pids(tenant_id)
            
            for pid in pids:
                try:
                    with open(procs_path, 'w') as f:
                        f.write(str(pid))
                except ProcessLookupError:
                    continue
                    
            self.logger.info(f"Isolated {tenant_id} to CPUs: {core_list}")
            return True
            
        except Exception as e:
            self.logger.error(f"CPU isolation failed: {e}")
            return False
            
    def remove_guardrails(self, tenant_id: str):
        """Remove all guardrails for tenant"""
        try:
            # Reset MPS quota to 100%
            self.apply_mps_quota(tenant_id, 100)
            
            # Remove cgroup constraints
            cgroup_path = f"/sys/fs/cgroup/gpu-tenants/{tenant_id}"
            if os.path.exists(cgroup_path):
                # Reset I/O limits
                io_max_path = f"{cgroup_path}/io.max"
                if os.path.exists(io_max_path):
                    with open(io_max_path, 'w') as f:
                        f.write("max\n")
                        
                # Reset CPU affinity
                cpuset_path = f"{cgroup_path}/cpuset.cpus"
                if os.path.exists(cpuset_path):
                    with open(cpuset_path, 'w') as f:
                        f.write(f"0-{os.cpu_count()-1}")
                        
            self.tenant_states.pop(tenant_id, None)
            self.logger.info(f"Removed guardrails for {tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Guardrail removal failed: {e}")
            
    def _get_tenant_pids(self, tenant_id: str) -> list:
        """Get process IDs for tenant (simplified lookup)"""
        try:
            # In real implementation, this would lookup tenant processes
            # via container runtime, systemd service, or process name matching
            result = subprocess.run(['pgrep', '-f', tenant_id], capture_output=True, text=True)
            if result.returncode == 0:
                return [int(pid) for pid in result.stdout.strip().split('\n') if pid]
        except:
            pass
        return []
        
    def _get_primary_storage_device(self) -> Optional[str]:
        """Get major:minor for primary storage device"""
        try:
            # Find root filesystem device
            result = subprocess.run(['findmnt', '-n', '-o', 'SOURCE', '/'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                device = result.stdout.strip()
                # Get major:minor from device
                stat_result = subprocess.run(['stat', '-c', '%t:%T', device], 
                                           capture_output=True, text=True)
                if stat_result.returncode == 0:
                    maj_min = stat_result.stdout.strip()
                    # Convert hex to decimal
                    major, minor = maj_min.split(':')
                    return f"{int(major, 16)}:{int(minor, 16)}"
        except:
            pass
        return "8:0"  # Default to /dev/sda

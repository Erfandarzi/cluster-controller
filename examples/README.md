# Examples

## Minimal Controller

```c
// Basic daemon setup - see controller.py for full implementation
// Requires CAP_SYS_ADMIN, nvidia-ml-py, /sys/fs/cgroup write access

config = ControllerConfig(
    tail_threshold_ms=15.0,    // SLO threshold
    persistence_windows=3,     // Consecutive violations required
    dwell_time_observations=256 // Anti-thrashing
)

controller = MultiTenancyController(config)
controller.start()  // Spawns monitoring thread
```

## Production Tuning

```bash
# Kernel-level PCIe settings
echo 'pci=realloc=off' >> /boot/grub/grub.cfg

# cgroup v2 hierarchy setup  
mkdir -p /sys/fs/cgroup/gpu-tenants/{high,med,low}
echo "+memory +io" > /sys/fs/cgroup/cgroup.subtree_control

# NUMA topology discovery
numactl --hardware | grep 'node.*cpus'
```

## Signal Verification

```bash
# PCIe bandwidth monitoring (requires root)
cat /sys/class/pci_bus/*/device/*/current_link_speed
lspci -vv | grep -A5 "Kernel driver in use: nvidia"

# Validate MIG isolation
nvidia-smi mig -lgip | awk '/GPU/{print $4}'
```

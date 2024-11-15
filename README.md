# cluster-controller

SLO-aware GPU resource controller. Adaptive isolation via PCIe placement, dynamic MIG profiles, and cgroup-based guardrails.

Built for multi-tenant HPC workloads on A100/H100 clusters where noisy neighbors violate latency SLOs.

## Features

- **MIG reconfiguration**: Runtime profile changes (1g.10gb→7g.80gb) via nvidia-smi
- **Topology-aware placement**: PCIe/NUMA hot-spot avoidance using lspci + hwloc  
- **SLO monitoring**: Per-tenant p99 tail latency tracking
- **Cgroup guardrails**: I/O throttling, MPS quotas for containment
- **Anti-thrashing**: Configurable dwell/cooldown periods

## Prerequisites

```bash
# Linux kernel 5.4+ with cgroup v2
mount | grep cgroup2
uname -r

# NVIDIA datacenter driver + MIG support
nvidia-smi -q | grep "MIG Mode"
modinfo nvidia | grep version

# Root access or CAP_SYS_ADMIN + CAP_SYS_RESOURCE
getcap /usr/bin/python3
```

## Build

```bash
# Install build deps
sudo apt install build-essential python3-dev libnuma-dev

# Compile native extensions (optional)
python setup.py build_ext --inplace

# System daemon install
sudo pip install -e .
sudo systemctl enable gpu-controller.service
```

## Configuration

Controller reads `/etc/gpu-controller.conf`:

```ini
[controller]
tail_threshold_ms=15.0
persistence_windows=3  
dwell_time_observations=256
cooldown_observations=128

[placement]
numa_weight=0.7
pcie_weight=0.3
```

Logs to journald. Metrics exported to `/var/run/gpu-controller/`.
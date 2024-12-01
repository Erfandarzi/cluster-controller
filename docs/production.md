# Production Deployment

## Systemd Service

```ini
# /etc/systemd/system/gpu-controller.service
[Unit]
Description=GPU Multi-Tenancy Controller
After=nvidia-persistenced.service
Requires=nvidia-persistenced.service

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 -m gpu_multi_tenancy.core.controller --config /etc/gpu-controller.conf
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

## Kernel Parameters

```bash
# /etc/sysctl.d/99-gpu-controller.conf
kernel.pid_max = 4194304
vm.max_map_count = 1048576

# PCIe tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
```

## Monitoring Integration

- Exports metrics to `/var/run/gpu-controller/metrics`
- Compatible with node_exporter textfile collector
- Logs to journald with structured format

## Troubleshooting

```bash
# Verify MIG topology
nvidia-smi mig -lgip | awk '/Instance/{print $0}'
cat /proc/driver/nvidia/gpus/*/information

# PCIe bandwidth debugging  
cat /sys/class/pci_bus/*/device/*/max_link_speed
lspci -vvv | grep -A10 "10de:" | grep LnkSta

# cgroup hierarchy validation
find /sys/fs/cgroup -name "gpu-tenants*" -type d
systemd-cgls | grep gpu

# Signal collection verification
journalctl -u gpu-controller -f --output=json | jq .MESSAGE
strace -e openat,write -p $(pidof gpu-controller) 2>&1 | grep -E "(cgroup|nvidia)"
```

## Performance Tuning

```bash
# Kernel IRQ affinity (isolate latency-sensitive cores)
echo 2 > /proc/irq/24/smp_affinity  # Pin to CPU1
echo rcu_nocbs=2-7 >> /boot/grub/grub.cfg

# PCIe link optimization
setpci -s 01:00.0 CAP_EXP+10.w=5012  # Force Gen4 x16
echo 1 > /sys/module/nvidia/parameters/NVreg_EnableMSI
```

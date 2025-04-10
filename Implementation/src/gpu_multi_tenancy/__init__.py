"""
GPU Multi-Tenancy Controller

A host-level controller for predictable GPU cluster dynamic partitioning 
and PCIe-aware placement for LLM inference workloads.
"""

__version__ = "0.1.0"
__author__ = "Erfan Darzi"

from .core.controller import MultiTenancyController
from .monitoring.signals import SignalCollector
from .placement.pcie_aware import PCIeAwarePlacement
from .guardrails.throttling import GuardrailsManager

__all__ = [
    "MultiTenancyController",
    "SignalCollector", 
    "PCIeAwarePlacement",
    "GuardrailsManager",
]
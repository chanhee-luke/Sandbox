"""Module defining various utilities."""

from sandbox.utils.misc import aeq, use_gpu
from sandbox.utils.report_manager import ReportMgr, build_report_manager
from sandbox.utils.statistics import Statistics
from sandbox.utils.optimizers import build_optim, MultipleOptimizer, \
    Optimizer

__all__ = ["aeq", "use_gpu", "ReportMgr",
           "build_report_manager", "Statistics",
           "build_optim", "MultipleOptimizer", "Optimizer"]

__version__ = "0.0.1"
""" Main entry point of the Sandbox library """

from __future__ import division, print_function

import sandbox.input
import sandbox.network
import sandbox.utils
import sandbox.modules
from sandbox.trainer import Trainer

import sys
import sandbox.utils.optimizers
sandbox.utils.optimizers.Optim = sandbox.utils.optimizers.Optimizer
sys.modules["sandbox.Optim"] = sandbox.utils.optimizers

# For Flake
__all__ = [sandbox.input, sandbox.network, sandbox.modules,
           sandbox.utils, "Trainer"]

__version__ = "0.0.1"
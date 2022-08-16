"""
forward

Dev environment for investigating forward gradient.
"""

from __future__ import annotations

from . import objectives as objectives
from . import optimisers as optim
from . import samplers as samplers
from .grad import df as df
from .grad import df_fwd as df_fwd

__version__ = "0.1"

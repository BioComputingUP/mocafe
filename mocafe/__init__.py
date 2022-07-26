"""
Mocafe (modeling cancer with FEniCS) is composed of different subpackages:

* ``angie`` contains an open implementation of the Phase Field Model for angiogenesis first reported by Travasso et
  al. (2011) :cite:`Travasso2011a`
* ``fenut`` contains general-purpose modules and classes
* ``litforms`` contains the implementation of other Phase Field Models presented to literature

And submodules:

* ``expressions`` contains useful fenics Expression for cancer Phase Field Models
* ``math`` contains useful mathematical functions for cancer Phase Field Models

See the extensive documentation for each subpackage and submodule in the sections below.

"""

import configparser
from pathlib import Path

# get version
config = configparser.ConfigParser()
pf = Path(__file__).parent.resolve().parent
with open(f"{pf}/setup.cfg", 'r') as cfg_file:
    config.read_file(cfg_file)

__version__ = config["metadata"]["version"]

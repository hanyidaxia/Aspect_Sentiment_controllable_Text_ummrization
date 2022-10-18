import operator
import os
import psutil
import sys
import numpy as np
import inspect

def listify(val):
    """If val is an element the func wraps it into a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    return [val]
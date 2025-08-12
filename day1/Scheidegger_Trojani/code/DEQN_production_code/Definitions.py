import importlib
import numpy as np
import tensorflow as tf
from Parameters import definitions, definition_bounds_hard, MODEL_NAME
import sys

Model_Definitions = importlib.import_module(MODEL_NAME + ".Definitions")

for d in definitions:
    if (d in definition_bounds_hard['lower']) or (d in definition_bounds_hard['upper']):
        setattr(sys.modules[__name__],d, 
                (lambda de: lambda s, ps: tf.clip_by_value(getattr(Model_Definitions,de)(s, ps), definition_bounds_hard['lower'].get(de, np.NINF), definition_bounds_hard['upper'].get(de,np.Inf)))(d)
                )
    else:
        setattr(sys.modules[__name__],d, (lambda de: lambda s, ps: getattr(Model_Definitions,de)(s, ps))(d))
        
    # always add a 'raw' attribute as well - this can be used for penalties
    setattr(sys.modules[__name__],d + "_RAW", (lambda de: lambda s, ps: getattr(Model_Definitions,de)(s, ps))(d))
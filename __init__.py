#----------------------------------------------------------

# set to False to disable auto update
AutoUpdate = True  
# always download last stable version
version = "main"   

#----------------------------------------------------------
#update logic
import os
if AutoUpdate:
    try:
        from .update import update as update
        currentPath = os.path.dirname(os.path.realpath(__file__))
        #run update/update.py to update the node class mappings
        update.update(currentPath,branch_name=version)
    except ImportError:
        print("Failed to auto update `Quality of Life Suit` ")

#----------------------------------------------------------

from .src.QualityOfLifeSuit_Omar92 import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_SUIT
try:
    from .src.QualityOfLife_deprecatedNodes import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_DEPRECATED
except ImportError:
    NODE_CLASS_MAPPINGS_DEPRECATED = {}


__all__ = ['NODE_CLASS_MAPPINGS_SUIT', 'NODE_CLASS_MAPPINGS_DEPRECATED', 'NODE_CLASS_MAPPINGS']
NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS_SUIT,
    **NODE_CLASS_MAPPINGS_DEPRECATED
}
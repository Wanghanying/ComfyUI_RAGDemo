
from .test_rag import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

from . import (
    test_rag
)


def update_mappings(module):
    NODE_CLASS_MAPPINGS.update(**module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(**module.NODE_DISPLAY_NAME_MAPPINGS)


update_mappings(test_rag)
from .backbone import *
from .global_ape_decoder import *
from .global_rpe_decomp_decoder import *
from .matcher import *
from .plain_detr import build
from .position_encoding import *
from .segmentation import *
from .swin_transformer_v2 import *
from .transformer import *
from .utils import *


def build_model(args):
    return build(args)
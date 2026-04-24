# CAST-G: Token-Agnostic Neural Architecture
# Hierarchical MI-Segmented Architecture with Mixture-of-Depths

from .model import CASTGModel
from .config import get_config, CONFIGS
from .encoder import ByteEncoder
from .boundary import MIBoundaryDetector, MISegmentationLoss, AdaptiveLagrangian
from .hierarchy import HierarchicalSegmenter
from .global_stack import MoDTransformerStack, MoDSegmentRouter, CausalTransformerBlock
from .decoder import CausalLocalDecoder

__all__ = [
    'CASTGModel',
    'get_config',
    'CONFIGS',
    'ByteEncoder',
    'MIBoundaryDetector',
    'MISegmentationLoss',
    'AdaptiveLagrangian',
    'HierarchicalSegmenter',
    'MoDTransformerStack',
    'MoDSegmentRouter',
    'CausalTransformerBlock',
    'CausalLocalDecoder',
]

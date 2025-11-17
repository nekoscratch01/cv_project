"""Pipeline package for perception → features → semantics → retrieval."""

from .config import SystemConfig
from .perception import VideoPerception, TrackRecord
from .features import TrackFeatureExtractor, TrackFeatures
from .evidence import EvidencePackage, build_evidence_packages
from .recall import RecallEngine
from .vlm_client import QwenVLMClient, QueryResult

__all__ = [
    "SystemConfig",
    "VideoPerception",
    "TrackRecord",
    "TrackFeatureExtractor",
    "TrackFeatures",
    "EvidencePackage",
    "build_evidence_packages",
    "RecallEngine",
    "QwenVLMClient",
    "QueryResult",
]

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.config import SystemConfig  # noqa: E402
from core.evidence import EvidencePackage  # noqa: E402
from pipeline.recall import RecallEngine  # noqa: E402


class SiglipStub:
    embedding_dim = 2

    def encode_text(self, texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def encode_images(self, images):
        return np.ones((len(images), 2), dtype=np.float32)


def test_visual_filter_prefers_higher_similarity(tmp_path):
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=SiglipStub())
    pkg1 = EvidencePackage("demo", 1, [1], [(0, 0, 1, 1)], ["c1.jpg"], 30.0, None, embedding=[1.0, 0.0])
    pkg2 = EvidencePackage("demo", 2, [1], [(0, 0, 1, 1)], ["c2.jpg"], 30.0, None, embedding=[0.0, 1.0])
    filtered = engine.visual_filter([pkg1, pkg2], "person", ["red"], top_k=1)
    assert len(filtered) == 1
    assert filtered[0].track_id == 1


def test_recall_limit_passthrough(tmp_path):
    config = SystemConfig(embedding_cache_dir=tmp_path / "embeddings")
    engine = RecallEngine(config=config, siglip_client=SiglipStub())
    packages = {
        i: EvidencePackage("demo", i, [1], [(0, 0, 1, 1)], [f"c{i}.jpg"], 30.0, None) for i in range(5)
    }
    limited = engine.recall("anything", packages, limit=2)
    assert len(limited) == 2


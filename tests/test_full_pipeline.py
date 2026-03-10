"""
End-to-end integration test for the ArchitectAI pipeline.

Pipeline under test:
  parse_prompt → Architecture
  generate_diagram → PNG file
  encode_diagram → torch.Tensor embedding
  generate_explanation → plain-English string

Run with::

    pytest tests/test_full_pipeline.py -v
    # or with live output:
    pytest tests/test_full_pipeline.py -v -s
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

# ── project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── test constants ─────────────────────────────────────────────────────────
TEST_PROMPT = "React frontend with FastAPI backend and PostgreSQL database"
CONVNEXT_EMBED_DIM = 768  # ConvNeXt-Tiny output dim after head removal


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def architecture():
    """Parse the test prompt once for the whole module."""
    from backend.core.prompt_parser.parser import parse_prompt

    arch = parse_prompt(TEST_PROMPT)
    print(f"\n[parse_prompt] nodes={[n.id for n in arch.nodes]}")
    print(f"[parse_prompt] edges={[(e.from_node, e.to_node) for e in arch.edges]}")
    return arch


@pytest.fixture(scope="module")
def diagram_png(architecture):
    """Generate diagram PNG into a persistent temp directory."""
    from backend.core.diagram.generator import generate_diagram

    with tempfile.TemporaryDirectory(prefix="architectai_test_") as tmpdir:
        output_base = str(Path(tmpdir) / "test_arch")
        t0 = time.perf_counter()
        png_path = generate_diagram(architecture, output_path=output_base)
        elapsed = time.perf_counter() - t0
        print(f"\n[generate_diagram] path={png_path}  ({elapsed:.2f}s)")
        yield png_path


@pytest.fixture(scope="module")
def vision_embedding(diagram_png):
    """Run ConvNeXt encoder on the generated PNG."""
    import torch

    from backend.core.vision.encoder import encode_diagram

    t0 = time.perf_counter()
    embedding = encode_diagram(diagram_png)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[encode_diagram] shape={tuple(embedding.shape)} "
        f"device={embedding.device}  ({elapsed:.2f}s)"
    )
    return embedding


# ===========================================================================
# Step 1 – Prompt parsing
# ===========================================================================


class TestParsePrompt:
    def test_returns_architecture(self, architecture):
        from backend.api.schemas.architecture import Architecture

        assert isinstance(architecture, Architecture)

    def test_has_nodes(self, architecture):
        assert len(architecture.nodes) > 0, "Expected at least one node"

    def test_contains_frontend(self, architecture):
        from backend.api.schemas.architecture import NodeType

        types = {n.type for n in architecture.nodes}
        assert NodeType.Frontend in types, f"Frontend not detected. Got: {types}"

    def test_contains_backend(self, architecture):
        from backend.api.schemas.architecture import NodeType

        types = {n.type for n in architecture.nodes}
        assert NodeType.Backend in types, f"Backend not detected. Got: {types}"

    def test_contains_database(self, architecture):
        from backend.api.schemas.architecture import NodeType

        types = {n.type for n in architecture.nodes}
        assert NodeType.Database in types, f"Database not detected. Got: {types}"

    def test_has_edges(self, architecture):
        assert len(architecture.edges) >= 2, (
            f"Expected ≥ 2 edges for 3-tier prompt, got {len(architecture.edges)}"
        )

    def test_unique_node_ids(self, architecture):
        ids = [n.id for n in architecture.nodes]
        assert len(ids) == len(set(ids)), f"Duplicate node ids: {ids}"

    def test_edge_referential_integrity(self, architecture):
        valid_ids = {n.id for n in architecture.nodes}
        for e in architecture.edges:
            assert e.from_node in valid_ids, f"edge.from={e.from_node!r} missing"
            assert e.to_node in valid_ids, f"edge.to={e.to_node!r} missing"

    def test_metadata_version(self, architecture):
        assert architecture.metadata.version >= 1

    def test_serialisation_round_trip(self, architecture):
        from backend.api.schemas.architecture import Architecture

        payload = architecture.model_dump(by_alias=True)
        restored = Architecture.model_validate(payload)
        assert restored.node_ids() == architecture.node_ids()


# ===========================================================================
# Step 2 – Diagram generation
# ===========================================================================


class TestGenerateDiagram:
    def test_png_exists(self, diagram_png):
        assert Path(diagram_png).exists(), f"PNG not found at {diagram_png}"

    def test_png_extension(self, diagram_png):
        assert diagram_png.endswith(".png"), f"Expected .png, got {diagram_png}"

    def test_png_non_empty(self, diagram_png):
        size = Path(diagram_png).stat().st_size
        assert size > 1024, f"PNG suspiciously small: {size} bytes"

    def test_png_is_valid_image(self, diagram_png):
        from PIL import Image

        img = Image.open(diagram_png)
        img.verify()  # raises if corrupt


# ===========================================================================
# Step 3 – Vision encoder
# ===========================================================================


class TestVisionEncoder:
    def test_embedding_is_tensor(self, vision_embedding):
        import torch

        assert isinstance(vision_embedding, torch.Tensor)

    def test_embedding_shape(self, vision_embedding):
        assert vision_embedding.shape == (CONVNEXT_EMBED_DIM,), (
            f"Expected ({CONVNEXT_EMBED_DIM},), got {tuple(vision_embedding.shape)}"
        )

    def test_embedding_on_cpu(self, vision_embedding):
        assert vision_embedding.device.type == "cpu"

    def test_embedding_is_finite(self, vision_embedding):
        import torch

        assert torch.isfinite(vision_embedding).all(), "Embedding contains NaN/Inf"

    def test_embedding_nonzero(self, vision_embedding):
        assert vision_embedding.norm().item() > 0.0, "Embedding is all zeros"

    def test_embedding_statistics(self, vision_embedding):
        mean = vision_embedding.mean().item()
        std = vision_embedding.std().item()
        print(f"\n[embedding] mean={mean:.4f}  std={std:.4f}")
        # Not rigid numeric assertions — just sanity bounds for ReLU activations.
        assert std > 1e-4, "Embedding has near-zero variance — model may be broken"


# ===========================================================================
# Step 4 – Explanation generation (rule-based fallback)
# ===========================================================================


class TestExplanation:
    def test_rule_based_explanation(self, architecture):
        """Use the rule-based explainer (no GPU/LLM required for CI)."""
        from backend.core.vlm.explainer import generate_explanation_rule_based

        explanation = generate_explanation_rule_based(architecture)
        print(f"\n[explanation] {explanation!r}")
        assert isinstance(explanation, str)
        assert len(explanation.strip()) > 20, "Explanation is too short"

    def test_explanation_mentions_components(self, architecture):
        from backend.core.vlm.explainer import generate_explanation_rule_based

        explanation = generate_explanation_rule_based(architecture).lower()
        # At least one component type should be mentioned.
        component_terms = {"frontend", "backend", "database", "api", "service"}
        assert any(t in explanation for t in component_terms), (
            f"Explanation doesn't mention any known components: {explanation!r}"
        )

    def test_llm_explanation_with_embedding(self, architecture, vision_embedding):
        """
        Full LLM explanation test — skipped automatically when the model is
        not cached locally (prevents CI from downloading Qwen on every run).
        """
        import os

        if os.environ.get("ARCHITECTAI_SKIP_LLM", "1") == "1":
            pytest.skip(
                "LLM test skipped (set ARCHITECTAI_SKIP_LLM=0 to enable). "
                "Requires ~4 GB model download."
            )

        from backend.core.vlm.explainer import generate_explanation

        t0 = time.perf_counter()
        explanation = generate_explanation(architecture, vision_embedding)
        elapsed = time.perf_counter() - t0

        print(f"\n[LLM explanation] ({elapsed:.1f}s)\n{explanation}")
        assert isinstance(explanation, str)
        assert len(explanation.strip()) > 20


# ===========================================================================
# Full pipeline smoke test (combined)
# ===========================================================================


class TestFullPipeline:
    def test_pipeline_end_to_end(self):
        """
        Single test that exercises all 4 steps in sequence without relying
        on module-scoped fixtures.  Uses rule-based explanation only.
        """
        import tempfile
        from pathlib import Path

        from backend.api.schemas.architecture import Architecture
        from backend.core.diagram.generator import generate_diagram
        from backend.core.prompt_parser.parser import parse_prompt
        from backend.core.vision.encoder import encode_diagram
        from backend.core.vlm.explainer import generate_explanation_rule_based

        prompt = "Vue frontend, Express backend, MongoDB database, Redis cache"

        # Step 1
        arch = parse_prompt(prompt)
        assert isinstance(arch, Architecture)
        assert len(arch.nodes) >= 3

        # Step 2
        with tempfile.TemporaryDirectory() as tmpdir:
            png = generate_diagram(arch, output_path=str(Path(tmpdir) / "pipeline"))
            assert Path(png).exists()

            # Step 3
            embedding = encode_diagram(png)
            assert embedding.shape[0] == CONVNEXT_EMBED_DIM

            # Step 4
            explanation = generate_explanation_rule_based(arch)
            assert len(explanation.strip()) > 10

            print(
                f"\n[full-pipeline]\n"
                f"  nodes      : {[n.id for n in arch.nodes]}\n"
                f"  png        : {png}\n"
                f"  embedding  : {tuple(embedding.shape)}\n"
                f"  explanation: {explanation!r}"
            )

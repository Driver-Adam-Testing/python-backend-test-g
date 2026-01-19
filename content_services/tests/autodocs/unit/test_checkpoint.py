"""Unit tests for AutoDocs checkpoint data class and storage.

Tests cover:
- Pydantic model serialization/deserialization
- Phase ordering and comparison
- Config hash computation
- S3 operations (upload, download, delete)
- Validation logic
- Async wrapper behavior
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest


class TestAutoDocsCheckpoint:
    """Tests for the AutoDocsCheckpoint Pydantic model."""

    def test_checkpoint_serialization_roundtrip(self):
        """Checkpoint should serialize to JSON and deserialize identically."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test-svn-123",
            config_hash="abc123def456",
            toml_content="[document]\ngoal = 'test'",
            started_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
            phase_current=50,
            phase_total=200,
            use_tagging=True,
            annotations={"src/main.py": [1, 2, 0]},
        )

        # Serialize
        json_str = checkpoint.model_dump_json()

        # Deserialize
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        # Verify identical
        assert restored.source_version_node_id == checkpoint.source_version_node_id
        assert restored.config_hash == checkpoint.config_hash
        assert restored.toml_content == checkpoint.toml_content
        assert restored.started_at == checkpoint.started_at
        assert restored.current_phase == checkpoint.current_phase
        assert restored.phase_current == checkpoint.phase_current
        assert restored.phase_total == checkpoint.phase_total
        assert restored.use_tagging == checkpoint.use_tagging
        assert restored.annotations == checkpoint.annotations

    def test_checkpoint_version_included(self):
        """Checkpoint should include version for compatibility checks."""
        from autodocs.src.checkpoint import (
            AUTODOCS_CHECKPOINT_VERSION,
            AutoDocsCheckpoint,
            AutoDocsPhase,
        )

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.INITIALIZING,
        )

        data = json.loads(checkpoint.model_dump_json())
        assert data["version"] == AUTODOCS_CHECKPOINT_VERSION

    def test_checkpoint_default_timestamps(self):
        """last_updated_at should default to now."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        before = datetime.now(UTC)
        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.INITIALIZING,
        )
        after = datetime.now(UTC)

        assert before <= checkpoint.last_updated_at <= after

    def test_checkpoint_with_scatter_state(self):
        """Checkpoint should serialize scatter_state correctly."""
        from autodocs.src.checkpoint import (
            AutoDocsCheckpoint,
            AutoDocsPhase,
            ScatterState,
        )

        scatter = ScatterState(
            file_by_file_content={"src/a.py": "content a", "src/b.py": "content b"},
            nodes_processed=50,
            nodes_total=100,
        )

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
            scatter_state={"Architecture": scatter},
        )

        # Roundtrip
        json_str = checkpoint.model_dump_json()
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        assert "Architecture" in restored.scatter_state
        assert restored.scatter_state["Architecture"].nodes_processed == 50
        assert (
            restored.scatter_state["Architecture"].file_by_file_content["src/a.py"]
            == "content a"
        )

    def test_checkpoint_with_gather_state(self):
        """Checkpoint should serialize gather_state correctly."""
        from autodocs.src.checkpoint import (
            AutoDocsCheckpoint,
            AutoDocsPhase,
            GatherState,
        )

        gather = GatherState(
            aggregate_docs=["aggregated content 1", "aggregated content 2"],
            aggregation_round=1,
        )

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
            gather_state={"Architecture": gather},
        )

        # Roundtrip
        json_str = checkpoint.model_dump_json()
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        assert "Architecture" in restored.gather_state
        assert restored.gather_state["Architecture"].aggregation_round == 1
        assert len(restored.gather_state["Architecture"].aggregate_docs) == 2

    def test_checkpoint_with_none_optional_fields(self):
        """Checkpoint should serialize correctly with None optional fields."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.INITIALIZING,
            annotations=None,
            scatter_state=None,
            gather_state=None,
        )

        json_str = checkpoint.model_dump_json()
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        assert restored.annotations is None
        assert restored.scatter_state is None
        assert restored.gather_state is None

    def test_toml_content_preserved(self):
        """Checkpoint should store and restore full TOML content string."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        toml_content = """[document]
goal = "Create architecture documentation"
format = "markdown"

[[sections]]
title = "Overview"
instruction = "Describe the system"
"""

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content=toml_content,
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
        )

        json_str = checkpoint.model_dump_json()
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        assert restored.toml_content == toml_content


class TestConfigHash:
    """Tests for config hash computation."""

    def test_config_hash_deterministic(self):
        """Same TOML content should produce same hash."""
        from autodocs.src.checkpoint import compute_config_hash

        toml = "[document]\ngoal = 'test'"

        hash1 = compute_config_hash(toml)
        hash2 = compute_config_hash(toml)

        assert hash1 == hash2

    def test_config_hash_detects_changes(self):
        """Different TOML content should produce different hash."""
        from autodocs.src.checkpoint import compute_config_hash

        toml1 = "[document]\ngoal = 'test'"
        toml2 = "[document]\ngoal = 'different'"

        hash1 = compute_config_hash(toml1)
        hash2 = compute_config_hash(toml2)

        assert hash1 != hash2

    def test_config_hash_length(self):
        """Config hash should be 16 characters (truncated SHA256)."""
        from autodocs.src.checkpoint import compute_config_hash

        hash_value = compute_config_hash("test content")

        assert len(hash_value) == 16


class TestCheckpointValidation:
    """Tests for checkpoint validation logic."""

    def test_valid_checkpoint_passes(self):
        """Valid checkpoint should pass validation."""
        from autodocs.src.checkpoint import (
            AutoDocsCheckpoint,
            AutoDocsPhase,
            compute_config_hash,
        )

        toml = "[document]\ngoal = 'test'"
        config_hash = compute_config_hash(toml)

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash=config_hash,
            toml_content=toml,
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
            use_tagging=True,
        )

        assert checkpoint.is_valid_for_resume(config_hash, use_tagging=True) is True

    def test_config_hash_mismatch_fails(self):
        """Checkpoint with different config hash should fail validation."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="old_hash_12345",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
            use_tagging=True,
        )

        assert (
            checkpoint.is_valid_for_resume("new_hash_67890", use_tagging=True) is False
        )

    def test_use_tagging_mismatch_fails(self):
        """Checkpoint with different use_tagging should fail validation."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
            use_tagging=True,
        )

        assert checkpoint.is_valid_for_resume("abc123", use_tagging=False) is False

    def test_version_mismatch_fails(self):
        """Checkpoint with wrong version should fail validation."""
        from autodocs.src.checkpoint import _validate_checkpoint_version

        checkpoint_data = {
            "version": "0.9",  # Old version
            "source_version_node_id": "test",
        }

        assert _validate_checkpoint_version(checkpoint_data) is False

    def test_version_match_passes(self):
        """Checkpoint with correct version should pass validation."""
        from autodocs.src.checkpoint import (
            AUTODOCS_CHECKPOINT_VERSION,
            _validate_checkpoint_version,
        )

        checkpoint_data = {
            "version": AUTODOCS_CHECKPOINT_VERSION,
            "source_version_node_id": "test",
        }

        assert _validate_checkpoint_version(checkpoint_data) is True


class TestCheckpointS3Storage:
    """Tests for S3 checkpoint upload/download/delete operations."""

    def test_upload_checkpoint_creates_correct_key(self):
        """Should upload checkpoint JSON to correct S3 path."""
        from autodocs.src.checkpoint import (
            AutoDocsCheckpoint,
            AutoDocsPhase,
            _upload_checkpoint_sync,
        )

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test-svn-uuid",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
        )

        mock_s3 = MagicMock()
        _upload_checkpoint_sync(checkpoint, bucket="test-bucket", s3_client=mock_s3)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "autodocs/test-svn-uuid/checkpoint.json"
        assert call_kwargs["ContentType"] == "application/json"

    def test_upload_checkpoint_writes_valid_json(self):
        """Uploaded bytes should parse as valid checkpoint."""
        from autodocs.src.checkpoint import (
            AutoDocsCheckpoint,
            AutoDocsPhase,
            _upload_checkpoint_sync,
        )

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test-svn-uuid",
            config_hash="abc123",
            toml_content="[doc]\ngoal='test'",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
            phase_current=50,
            phase_total=100,
        )

        mock_s3 = MagicMock()
        _upload_checkpoint_sync(checkpoint, bucket="test-bucket", s3_client=mock_s3)

        # Get the body that was uploaded
        body = mock_s3.put_object.call_args.kwargs["Body"]
        data = json.loads(body.decode("utf-8"))

        assert data["source_version_node_id"] == "test-svn-uuid"
        assert data["current_phase"] == "section_update"
        assert data["phase_current"] == 50

    def test_download_checkpoint_returns_valid_checkpoint(self):
        """Should download and parse existing checkpoint."""
        from autodocs.src.checkpoint import (
            AUTODOCS_CHECKPOINT_VERSION,
            AutoDocsPhase,
            _download_checkpoint_sync,
        )

        checkpoint_data = {
            "version": AUTODOCS_CHECKPOINT_VERSION,
            "source_version_node_id": "test-svn-uuid",
            "config_hash": "abc123",
            "toml_content": "",
            "started_at": "2024-01-15T10:30:00Z",
            "last_updated_at": "2024-01-15T11:00:00Z",
            "current_phase": "annotating",
            "phase_current": 50,
            "phase_total": 200,
            "use_tagging": True,
        }

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(checkpoint_data).encode())
        }

        result = _download_checkpoint_sync(
            bucket="test-bucket",
            source_version_node_id="test-svn-uuid",
            s3_client=mock_s3,
        )

        assert result is not None
        assert result.source_version_node_id == "test-svn-uuid"
        assert result.current_phase == AutoDocsPhase.ANNOTATING
        assert result.phase_current == 50

    def test_download_checkpoint_handles_missing(self):
        """Should return None when checkpoint doesn't exist."""
        from autodocs.src.checkpoint import _download_checkpoint_sync
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        result = _download_checkpoint_sync(
            bucket="test-bucket",
            source_version_node_id="test-svn-uuid",
            s3_client=mock_s3,
        )

        assert result is None

    def test_download_checkpoint_handles_corrupted(self):
        """Should return None when checkpoint JSON is corrupted."""
        from autodocs.src.checkpoint import _download_checkpoint_sync

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: b"not valid json{{{")
        }

        result = _download_checkpoint_sync(
            bucket="test-bucket",
            source_version_node_id="test-svn-uuid",
            s3_client=mock_s3,
        )

        assert result is None

    def test_download_checkpoint_handles_old_version(self):
        """Should return None when checkpoint has old version."""
        from autodocs.src.checkpoint import _download_checkpoint_sync

        checkpoint_data = {
            "version": "0.1",  # Old version
            "source_version_node_id": "test",
        }

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(checkpoint_data).encode())
        }

        result = _download_checkpoint_sync(
            bucket="test-bucket", source_version_node_id="test", s3_client=mock_s3
        )

        assert result is None

    def test_delete_checkpoint_calls_s3(self):
        """Should delete checkpoint from S3."""
        from autodocs.src.checkpoint import _delete_checkpoint_sync

        mock_s3 = MagicMock()
        _delete_checkpoint_sync(
            bucket="test-bucket",
            source_version_node_id="test-svn-uuid",
            s3_client=mock_s3,
        )

        mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="autodocs/test-svn-uuid/checkpoint.json"
        )


class TestAnnotationsListTupleCompatibility:
    """Tests for JSON list/tuple serialization compatibility."""

    def test_annotations_with_list_values(self):
        """Annotations should work with list values (as stored in JSON)."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        # JSON deserializes tuples as lists
        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
            annotations={
                "src/main.py": [1, 2, 0],  # list, not tuple
                "src/utils.py": [0, 1, 2],
            },
        )

        json_str = checkpoint.model_dump_json()
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        # Should work regardless of list/tuple
        assert restored.annotations["src/main.py"] == [1, 2, 0]
        assert restored.annotations["src/utils.py"] == [0, 1, 2]

    def test_pdf_annotations_nested_structure(self):
        """PDF annotations with nested dict structure should serialize correctly."""
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
            pdf_annotations={
                "doc.pdf": {
                    0: [1, 2, 0],  # page 0
                    1: [0, 1, 2],  # page 1
                }
            },
        )

        json_str = checkpoint.model_dump_json()
        restored = AutoDocsCheckpoint.model_validate_json(json_str)

        # Note: JSON keys become strings, need to handle that
        assert "doc.pdf" in restored.pdf_annotations


class TestCheckpointStateMutation:
    """Tests for checkpoint state mutation methods."""

    def test_update_phase_mutates_state(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.INITIALIZING,
        )

        checkpoint.update_phase(AutoDocsPhase.ANNOTATING, current=50, total=100)

        assert checkpoint.current_phase == AutoDocsPhase.ANNOTATING
        assert checkpoint.phase_current == 50
        assert checkpoint.phase_total == 100

    def test_update_annotations_converts_integers(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
        )

        # Simulate Category enums as IntEnum values (represented as ints here)
        annotations = {"src/main.py": [1, 2, 0], "src/utils.py": [0, 1, 2]}
        pdf_annotations = {"doc.pdf": {0: [1, 2], 1: [0, 1]}}

        checkpoint.update_annotations(annotations, pdf_annotations)

        assert checkpoint.annotations == {
            "src/main.py": [1, 2, 0],
            "src/utils.py": [0, 1, 2],
        }
        assert checkpoint.pdf_annotations == {"doc.pdf": {0: [1, 2], 1: [0, 1]}}

    def test_update_sections(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
        )

        sections = [{"order_idx": 0, "title": "Overview", "content": "Content"}]
        init_nodes = {"src/main.py", "src/utils.py"}

        checkpoint.update_sections(sections, init_nodes)

        assert checkpoint.sections_content == sections
        assert set(checkpoint.init_node_set) == init_nodes

    def test_update_scatter_state(self):
        from autodocs.src.checkpoint import (
            AutoDocsCheckpoint,
            AutoDocsPhase,
            ScatterState,
        )

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
        )

        state = ScatterState(
            file_by_file_content={"src/a.py": "content"},
            nodes_processed=10,
            nodes_total=100,
        )
        checkpoint.update_scatter_state("Architecture", state)

        assert checkpoint.scatter_state["Architecture"] == state

    def test_update_topo_index(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.SECTION_UPDATE,
        )

        checkpoint.update_topo_index(42)

        assert checkpoint.current_topo_index == 42

    def test_update_pdf_index(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.UPDATING_PDFS,
        )

        checkpoint.update_pdf_index(5)

        assert checkpoint.current_pdf_index == 5


class TestCheckpointFactory:
    """Tests for checkpoint factory methods."""

    def test_create_initial_factory(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint.create_initial(
            source_version_node_id="test-svn-123",
            hatchet_id="hatchet-456",
            toml_content="[document]\ngoal = 'test'",
            config_hash="hash123",
            use_tagging=True,
            appended_reverse_topo_paths=["src/a.py", "src/b.py"],
        )

        assert checkpoint.source_version_node_id == "test-svn-123"
        assert checkpoint.hatchet_id == "hatchet-456"
        assert checkpoint.toml_content == "[document]\ngoal = 'test'"
        assert checkpoint.config_hash == "hash123"
        assert checkpoint.use_tagging is True
        assert checkpoint.current_phase == AutoDocsPhase.INITIALIZING
        assert checkpoint.appended_reverse_topo_paths == ["src/a.py", "src/b.py"]


class TestCheckpointPersistenceMethods:
    """Tests for checkpoint persistence methods (save/delete/load_for_resume)."""

    @pytest.mark.asyncio
    async def test_save_calls_upload(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
        )

        mock_s3 = MagicMock()

        with patch("autodocs.src.checkpoint.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None
            await checkpoint.save("test-bucket", mock_s3)

            mock_to_thread.assert_called_once()
            assert mock_to_thread.call_args[0][0].__name__ == "_upload_checkpoint_sync"

    @pytest.mark.asyncio
    async def test_delete_calls_delete(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint, AutoDocsPhase

        checkpoint = AutoDocsCheckpoint(
            source_version_node_id="test-svn-uuid",
            config_hash="abc123",
            toml_content="",
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.ANNOTATING,
        )

        mock_s3 = MagicMock()

        with patch("autodocs.src.checkpoint.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None
            await checkpoint.delete("test-bucket", mock_s3)

            mock_to_thread.assert_called_once()
            assert mock_to_thread.call_args[0][0].__name__ == "_delete_checkpoint_sync"

    @pytest.mark.asyncio
    async def test_load_for_resume_returns_valid_checkpoint(self):
        from autodocs.src.checkpoint import (
            AUTODOCS_CHECKPOINT_VERSION,
            AutoDocsCheckpoint,
            AutoDocsPhase,
        )

        checkpoint_data = {
            "version": AUTODOCS_CHECKPOINT_VERSION,
            "source_version_node_id": "test-svn",
            "config_hash": "abc123",
            "toml_content": "[doc]",
            "started_at": "2024-01-15T10:30:00Z",
            "last_updated_at": "2024-01-15T11:00:00Z",
            "current_phase": "annotating",
            "phase_current": 50,
            "phase_total": 100,
            "use_tagging": True,
        }

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(checkpoint_data).encode())
        }

        with patch("autodocs.src.checkpoint.boto3.client", return_value=mock_s3):
            result = await AutoDocsCheckpoint.load_for_resume(
                bucket="test-bucket",
                svn_id="test-svn",
                config_hash="abc123",
                use_tagging=True,
                s3_client=mock_s3,
            )

        assert result is not None
        assert result.source_version_node_id == "test-svn"
        assert result.current_phase == AutoDocsPhase.ANNOTATING

    @pytest.mark.asyncio
    async def test_load_for_resume_returns_none_on_config_mismatch(self):
        from autodocs.src.checkpoint import (
            AUTODOCS_CHECKPOINT_VERSION,
            AutoDocsCheckpoint,
        )

        checkpoint_data = {
            "version": AUTODOCS_CHECKPOINT_VERSION,
            "source_version_node_id": "test-svn",
            "config_hash": "old_hash",
            "toml_content": "[doc]",
            "started_at": "2024-01-15T10:30:00Z",
            "last_updated_at": "2024-01-15T11:00:00Z",
            "current_phase": "annotating",
            "phase_current": 50,
            "phase_total": 100,
            "use_tagging": True,
        }

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(checkpoint_data).encode())
        }

        result = await AutoDocsCheckpoint.load_for_resume(
            bucket="test-bucket",
            svn_id="test-svn",
            config_hash="new_hash",  # Different hash
            use_tagging=True,
            s3_client=mock_s3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_load_for_resume_returns_none_when_not_found(self):
        from autodocs.src.checkpoint import AutoDocsCheckpoint
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )

        result = await AutoDocsCheckpoint.load_for_resume(
            bucket="test-bucket",
            svn_id="nonexistent",
            config_hash="abc123",
            use_tagging=True,
            s3_client=mock_s3,
        )

        assert result is None

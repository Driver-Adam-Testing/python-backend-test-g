"""Checkpoint data class and storage for AutoDocs resume capability.

This module provides:
- AutoDocsCheckpoint: Pydantic model with persistence methods (save, delete, load_for_resume)
- AutoDocsPhase: Enum for tracking execution phases
- ScatterState, GatherState: Models for mid-phase progress tracking
- compute_config_hash: Hash function for TOML config validation

Design follows the analytics checkpoint pattern but adapted for AutoDocs:
- Async methods on checkpoint class wrap sync S3 operations
- Phase-based progress tracking instead of commit-based
- Config hash validation instead of git SHA validation
- TOML content preservation for dynamically-generated configs
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Increment this when checkpoint schema changes incompatibly
AUTODOCS_CHECKPOINT_VERSION = "1.0"


class AutoDocsPhase(StrEnum):
    """Ordered phases of AutoDocs execution.

    Phase ordering is used for resume logic - we skip phases
    that are strictly less than the checkpoint's current_phase.

    Note on granularity:
    - ANNOTATING: Mid-phase progress tracked via phase_current/phase_total and annotations dict
    - SECTION_UPDATE: Mid-phase progress tracked via scatter_state/gather_state per section
      (covers both scatter-gather and sequential-edit methods)
    """

    INITIALIZING = "initializing"
    ANNOTATING = (
        "annotating"  # Granular: batches of 50 nodes, checkpoint annotations dict
    )
    SECTION_UPDATE = (
        "section_update"  # Granular: scatter batches (100 nodes), gather levels
    )
    UPDATING_PDFS = "updating_pdfs"
    FORMATTING = "formatting"
    BEFORE_ASSEMBLY = "before_assembly"
    ASSEMBLING = "assembling"
    COMPLETE = "complete"


class ScatterState(BaseModel):
    """Per-section scatter progress for mid-scatter checkpointing."""

    file_by_file_content: dict[str, str]  # path -> generated content (accumulated)
    nodes_processed: int
    nodes_total: int


class GatherState(BaseModel):
    """Per-section gather progress for mid-gather checkpointing."""

    aggregate_docs: list[str]  # Current aggregation level
    aggregation_round: int  # 0 = initial gather, 1+ = recursive


class AutoDocsCheckpoint(BaseModel):
    """Checkpoint state for fault-tolerant AutoDocs execution.

    This model captures all state needed to resume AutoDocs generation
    after a failure or timeout. It is saved to S3 at phase boundaries
    and mid-phase during long-running operations.
    """

    # Schema version for compatibility
    version: str = Field(default=AUTODOCS_CHECKPOINT_VERSION)

    # Identity
    source_version_node_id: str
    content_kind: (
        str  # e.g. "application_note" - part of checkpoint key to avoid collisions
    )
    hatchet_id: str | None = None
    config_hash: str  # Hash of TOML config for validation
    toml_content: (
        str  # Full TOML content - sections are defined here, must be preserved
    )

    # Timing
    started_at: datetime
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Phase tracking (discrete integers for precise resume)
    current_phase: AutoDocsPhase
    phase_current: int = 0  # Current item index (0-based)
    phase_total: int = 0  # Total items in this phase
    last_processed_node: str | None = None  # For validation on resume

    # Configuration flags (needed to validate resume compatibility)
    use_tagging: bool = True

    # Annotation state - None if use_tagging=False
    annotations: dict[str, list[int]] | None = (
        None  # node_path -> [Category per section]
    )
    pdf_annotations: dict[str, dict[int, list[int]]] | None = (
        None  # pdf_path -> {page_idx -> [Category]}
    )

    # Section state (accumulated during SECTION_UPDATE phase)
    sections_content: list[dict] | None = (
        None  # [{"order_idx": int, "title": str, "content": str}, ...]
    )
    init_node_set: list[str] | None = (
        None  # Nodes used for initial drafts (list for JSON, SEQUENTIAL_EDIT only)
    )

    # SECTION_UPDATE phase - scatter-gather tracking (for mid-phase checkpointing)
    scatter_state: dict[str, ScatterState] | None = (
        None  # section_title -> ScatterState
    )
    gather_state: dict[str, GatherState] | None = None  # section_title -> GatherState
    current_section_index: int = (
        0  # Which section we're processing (for sequential section processing)
    )

    # SECTION_UPDATE phase - sequential-edit tracking (SEQUENTIAL_EDIT method only)
    current_topo_index: int = (
        0  # Position in appended_reverse_topo (1-indexed like pidx)
    )

    # PDF update state
    current_pdf_index: int = 0  # Position in PDF processing

    # Traversal context - PATHS ONLY (reconstruct TechDocsContent from DriverDocsContent.content)
    appended_reverse_topo_paths: list[str] | None = (
        None  # Just paths, not full TechDocsContent
    )

    # === State Mutation ===

    def update_phase(
        self, phase: "AutoDocsPhase", current: int = 0, total: int = 0
    ) -> None:
        self.current_phase = phase
        self.phase_current = current
        self.phase_total = total
        self.last_updated_at = datetime.now(UTC)

    def update_annotations(
        self,
        annotations: dict[str, list],
        pdf_annotations: dict[str, dict[int, list]] | None = None,
    ) -> None:
        # Convert Category enums (IntEnum) to int for JSON serialization
        self.annotations = {
            path: [int(cat) for cat in cats] for path, cats in annotations.items()
        }
        if pdf_annotations is not None:
            self.pdf_annotations = {
                path: {
                    page_idx: [int(cat) for cat in cats]
                    for page_idx, cats in pages.items()
                }
                for path, pages in pdf_annotations.items()
            }
        self.last_updated_at = datetime.now(UTC)

    def update_sections(
        self,
        sections_content: list[dict],
        init_node_set: set | None = None,
    ) -> None:
        self.sections_content = sections_content
        self.init_node_set = list(init_node_set) if init_node_set else None
        self.last_updated_at = datetime.now(UTC)

    def update_scatter_state(self, section_title: str, state: "ScatterState") -> None:
        if self.scatter_state is None:
            self.scatter_state = {}
        self.scatter_state[section_title] = state
        self.last_updated_at = datetime.now(UTC)

    def update_topo_index(self, index: int) -> None:
        self.current_topo_index = index
        self.last_updated_at = datetime.now(UTC)

    def update_pdf_index(self, index: int) -> None:
        self.current_pdf_index = index
        self.last_updated_at = datetime.now(UTC)

    # === Validation ===

    def is_valid_for_resume(self, config_hash: str, use_tagging: bool) -> bool:
        if self.config_hash != config_hash:
            logger.warning(
                f"Autodocs checkpoint config mismatch: "
                f"{self.config_hash} != {config_hash} (current)"
            )
            return False

        if self.use_tagging != use_tagging:
            logger.warning(
                f"Autodocs checkpoint use_tagging mismatch: "
                f"{self.use_tagging} != {use_tagging} (current)"
            )
            return False

        return True

    # === Persistence ===

    async def save(self, bucket: str, s3_client: Any = None) -> None:
        try:
            await asyncio.to_thread(_upload_checkpoint_sync, self, bucket, s3_client)
            logger.info(
                f"Checkpoint saved: phase={self.current_phase}, "
                f"progress={self.phase_current}/{self.phase_total}"
            )
        except Exception as e:
            logger.warning(f"Failed to save checkpoint (non-fatal): {e}")

    async def delete(self, bucket: str, s3_client: Any = None) -> None:
        await asyncio.to_thread(
            _delete_checkpoint_sync,
            bucket,
            self.source_version_node_id,
            self.content_kind,
            s3_client,
        )

    # === Factory/Loading ===

    @classmethod
    async def load_for_resume(
        cls,
        bucket: str,
        svn_id: str,
        content_kind: str,
        config_hash: str,
        use_tagging: bool,
        s3_client: Any = None,
    ) -> Self | None:
        """Load and validate checkpoint in one step. Returns None if not found or invalid."""
        checkpoint = await asyncio.to_thread(
            _download_checkpoint_sync, bucket, svn_id, content_kind, s3_client
        )

        if checkpoint is None:
            return None

        if not checkpoint.is_valid_for_resume(config_hash, use_tagging):
            logger.warning("Checkpoint invalid for resume, starting fresh")
            return None

        return checkpoint

    @classmethod
    def create_initial(
        cls,
        source_version_node_id: str,
        content_kind: str,
        hatchet_id: str | None,
        toml_content: str,
        config_hash: str,
        use_tagging: bool,
        appended_reverse_topo_paths: list[str] | None = None,
    ) -> Self:
        return cls(
            source_version_node_id=source_version_node_id,
            content_kind=content_kind,
            hatchet_id=hatchet_id,
            config_hash=config_hash,
            toml_content=toml_content,
            started_at=datetime.now(UTC),
            current_phase=AutoDocsPhase.INITIALIZING,
            use_tagging=use_tagging,
            appended_reverse_topo_paths=appended_reverse_topo_paths,
        )


def compute_config_hash(toml_content: str) -> str:
    """Compute a hash of TOML config content for validation.

    Args:
        toml_content: Raw TOML configuration string

    Returns:
        SHA-256 hash of the content (first 16 chars for readability)
    """
    return hashlib.sha256(toml_content.encode("utf-8")).hexdigest()[:16]


def _get_autodocs_checkpoint_key(source_version_node_id: str, content_kind: str) -> str:
    """Get the S3 key for an autodocs checkpoint.

    The key includes content_kind to avoid collisions when multiple autodoc types
    (e.g., application_note, LONG_DESCRIPTION) are generated for the same version node.
    """
    return f"autodocs/{source_version_node_id}/{content_kind}/checkpoint.json"


# =============================================================================
# Sync S3 Operations (internal - used via asyncio.to_thread)
# =============================================================================


def _upload_checkpoint_sync(
    checkpoint: AutoDocsCheckpoint,
    bucket: str,
    s3_client: Any = None,
) -> None:
    if s3_client is None:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    key = _get_autodocs_checkpoint_key(
        checkpoint.source_version_node_id, checkpoint.content_kind
    )
    body = checkpoint.model_dump_json().encode("utf-8")

    logger.info(
        f"Uploading autodocs checkpoint to s3://{bucket}/{key} "
        f"(phase={checkpoint.current_phase}, {len(body)} bytes)"
    )
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def _download_checkpoint_sync(
    bucket: str,
    source_version_node_id: str,
    content_kind: str,
    s3_client: Any = None,
) -> AutoDocsCheckpoint | None:
    if s3_client is None:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    key = _get_autodocs_checkpoint_key(source_version_node_id, content_kind)

    # Fetch from S3
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.debug(f"No autodocs checkpoint found at s3://{bucket}/{key}")
            return None
        raise

    # Parse JSON
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Checkpoint JSON corrupted")
        return None

    # Validate version
    if not _validate_checkpoint_version(data):
        logger.warning("Discarding autodocs checkpoint due to version mismatch")
        return None

    # Deserialize to Pydantic model
    try:
        checkpoint = AutoDocsCheckpoint.model_validate_json(body)
    except ValidationError:
        logger.warning("Checkpoint validation failed")
        return None

    logger.info(
        f"Downloaded autodocs checkpoint: phase={checkpoint.current_phase}, "
        f"progress={checkpoint.phase_current}/{checkpoint.phase_total}"
    )
    return checkpoint


def _delete_checkpoint_sync(
    bucket: str,
    source_version_node_id: str,
    content_kind: str,
    s3_client: Any = None,
) -> None:
    """Delete checkpoint from S3 (sync version for use with asyncio.to_thread).

    Note: This function does NOT have internal error handling.
    Callers should wrap in try/except if deletion failure should be non-fatal.

    Args:
        bucket: S3 bucket name
        source_version_node_id: ID to delete checkpoint for
        content_kind: The content kind (e.g., "application_note")
        s3_client: Optional boto3 S3 client (for testing)
    """
    if s3_client is None:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    key = _get_autodocs_checkpoint_key(source_version_node_id, content_kind)
    logger.info(f"Deleting autodocs checkpoint at s3://{bucket}/{key}")
    s3_client.delete_object(Bucket=bucket, Key=key)


# =============================================================================
# Validation Functions (internal)
# =============================================================================


def _validate_checkpoint_version(checkpoint_data: dict) -> bool:
    """Check if checkpoint version matches current version."""
    version = checkpoint_data.get("version", "unknown")
    if version != AUTODOCS_CHECKPOINT_VERSION:
        logger.warning(
            f"Autodocs checkpoint version mismatch: {version} != {AUTODOCS_CHECKPOINT_VERSION}"
        )
        return False
    return True

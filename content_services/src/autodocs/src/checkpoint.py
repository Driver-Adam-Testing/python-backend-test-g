"""Checkpoint data class and storage for AutoDocs resume capability.

This module provides:
- AutoDocsCheckpoint: Pydantic model for checkpoint state
- Async S3 upload/download/delete functions (for use in async generate())
- Validation logic for checkpoint compatibility

Design follows the analytics checkpoint pattern but adapted for AutoDocs:
- Async wrappers around sync S3 operations (generate() is async)
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
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

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


# Phase ordering for comparison (phases are linearly ordered)
_PHASE_ORDER = {phase: idx for idx, phase in enumerate(AutoDocsPhase)}


def phase_is_before_or_equal(phase_a: AutoDocsPhase, phase_b: AutoDocsPhase) -> bool:
    """Check if phase_a is before or equal to phase_b in execution order."""
    return _PHASE_ORDER[phase_a] <= _PHASE_ORDER[phase_b]


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


def compute_config_hash(toml_content: str) -> str:
    """Compute a hash of TOML config content for validation.

    Args:
        toml_content: Raw TOML configuration string

    Returns:
        SHA-256 hash of the content (first 16 chars for readability)
    """
    return hashlib.sha256(toml_content.encode("utf-8")).hexdigest()[:16]


def _get_autodocs_checkpoint_key(source_version_node_id: str) -> str:
    """Get the S3 key for an autodocs checkpoint."""
    return f"autodocs/{source_version_node_id}/checkpoint.json"


# =============================================================================
# Sync S3 Operations (internal - used via asyncio.to_thread)
# =============================================================================


def _upload_checkpoint_sync(
    checkpoint: AutoDocsCheckpoint,
    bucket: str,
    s3_client: Any = None,
) -> None:
    """Upload checkpoint to S3 (sync version for use with asyncio.to_thread).

    Args:
        checkpoint: Checkpoint to upload
        bucket: S3 bucket name
        s3_client: Optional boto3 S3 client (for testing)
    """
    if s3_client is None:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    key = _get_autodocs_checkpoint_key(checkpoint.source_version_node_id)
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
    s3_client: Any = None,
) -> AutoDocsCheckpoint | None:
    """Download checkpoint from S3 (sync version for use with asyncio.to_thread).

    Args:
        bucket: S3 bucket name
        source_version_node_id: ID to download checkpoint for
        s3_client: Optional boto3 S3 client (for testing)

    Returns:
        AutoDocsCheckpoint if found and valid, None otherwise
    """
    if s3_client is None:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    key = _get_autodocs_checkpoint_key(source_version_node_id)

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()

        # Parse and validate version BEFORE Pydantic deserialization
        data = json.loads(body)
        if not _validate_checkpoint_version(data):
            logger.warning("Discarding autodocs checkpoint due to version mismatch")
            return None

        checkpoint = AutoDocsCheckpoint.model_validate_json(body)

        logger.info(
            f"Downloaded autodocs checkpoint: phase={checkpoint.current_phase}, "
            f"progress={checkpoint.phase_current}/{checkpoint.phase_total}"
        )
        return checkpoint

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.debug(f"No autodocs checkpoint found at s3://{bucket}/{key}")
            return None
        logger.warning(f"Error downloading autodocs checkpoint: {e}")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Autodocs checkpoint corrupted or invalid: {e}")
        return None


def _delete_checkpoint_sync(
    bucket: str,
    source_version_node_id: str,
    s3_client: Any = None,
) -> None:
    """Delete checkpoint from S3 (sync version for use with asyncio.to_thread).

    Note: This function does NOT have internal error handling.
    Callers should wrap in try/except if deletion failure should be non-fatal.

    Args:
        bucket: S3 bucket name
        source_version_node_id: ID to delete checkpoint for
        s3_client: Optional boto3 S3 client (for testing)
    """
    if s3_client is None:
        s3_client = boto3.client(
            "s3", endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
        )

    key = _get_autodocs_checkpoint_key(source_version_node_id)
    logger.info(f"Deleting autodocs checkpoint at s3://{bucket}/{key}")
    s3_client.delete_object(Bucket=bucket, Key=key)


# =============================================================================
# Async S3 Operations (public API)
# =============================================================================


async def upload_autodocs_checkpoint(
    checkpoint: AutoDocsCheckpoint,
    bucket: str,
    s3_client: Any = None,
) -> None:
    """Upload checkpoint to S3 asynchronously.

    Args:
        checkpoint: Checkpoint to upload
        bucket: S3 bucket name
        s3_client: Optional boto3 S3 client (for testing)
    """
    await asyncio.to_thread(_upload_checkpoint_sync, checkpoint, bucket, s3_client)


async def download_autodocs_checkpoint(
    bucket: str,
    source_version_node_id: str,
    s3_client: Any = None,
) -> AutoDocsCheckpoint | None:
    """Download checkpoint from S3 asynchronously.

    Args:
        bucket: S3 bucket name
        source_version_node_id: ID to download checkpoint for
        s3_client: Optional boto3 S3 client (for testing)

    Returns:
        AutoDocsCheckpoint if found and valid, None otherwise
    """
    return await asyncio.to_thread(
        _download_checkpoint_sync, bucket, source_version_node_id, s3_client
    )


async def delete_autodocs_checkpoint(
    bucket: str,
    source_version_node_id: str,
    s3_client: Any = None,
) -> None:
    """Delete checkpoint from S3 asynchronously.

    Note: This function does NOT have internal error handling.
    Callers should wrap in try/except if deletion failure should be non-fatal.

    Args:
        bucket: S3 bucket name
        source_version_node_id: ID to delete checkpoint for
        s3_client: Optional boto3 S3 client (for testing)
    """
    await asyncio.to_thread(
        _delete_checkpoint_sync, bucket, source_version_node_id, s3_client
    )


# =============================================================================
# Validation Functions
# =============================================================================


def _validate_checkpoint_version(checkpoint_data: dict) -> bool:
    """Check if checkpoint version matches current version.

    Args:
        checkpoint_data: Raw checkpoint data dict

    Returns:
        True if version matches, False otherwise
    """
    version = checkpoint_data.get("version", "unknown")
    if version != AUTODOCS_CHECKPOINT_VERSION:
        logger.warning(
            f"Autodocs checkpoint version mismatch: {version} != {AUTODOCS_CHECKPOINT_VERSION}"
        )
        return False
    return True


def validate_autodocs_checkpoint(
    checkpoint: AutoDocsCheckpoint,
    config_hash: str,
    use_tagging: bool,
) -> bool:
    """Validate checkpoint is compatible with current run configuration.

    Checks:
    1. Config hash matches (TOML hasn't changed)
    2. use_tagging flag matches (consistent annotation state)

    Args:
        checkpoint: Checkpoint to validate
        config_hash: Hash of current TOML config
        use_tagging: Current use_tagging setting

    Returns:
        True if valid for resume, False if checkpoint should be discarded
    """
    # Check config hash
    if checkpoint.config_hash != config_hash:
        logger.warning(
            f"Autodocs checkpoint config mismatch: "
            f"{checkpoint.config_hash} != {config_hash} (current)"
        )
        return False

    # Check use_tagging consistency
    if checkpoint.use_tagging != use_tagging:
        logger.warning(
            f"Autodocs checkpoint use_tagging mismatch: "
            f"{checkpoint.use_tagging} != {use_tagging} (current)"
        )
        return False

    return True

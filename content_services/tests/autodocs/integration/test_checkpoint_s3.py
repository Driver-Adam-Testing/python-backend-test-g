"""Integration tests for AutoDocs checkpoint S3 operations.

These tests require MinIO to be running (via docker-compose).
Run with: uv run python -m pytest content_services/tests/autodocs/integration -v -m integration

The tests verify the full checkpoint lifecycle against real S3-compatible storage:
- Upload checkpoint to MinIO
- Download and verify checkpoint
- Delete checkpoint
- Verify deletion

Skip conditions:
- MinIO not running on localhost:9000
- AWS_S3_ENDPOINT_URL not set to MinIO endpoint
"""

import os
from datetime import UTC, datetime

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

# MinIO configuration (matches docker-compose.yml)
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
TEST_BUCKET = "autodocs-test-bucket"


def is_minio_available() -> bool:
    """Check if MinIO is running and accessible."""
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        # Try to list buckets - this will fail if MinIO is not running
        s3_client.list_buckets()
        return True
    except (EndpointConnectionError, ClientError, Exception):
        return False


@pytest.fixture(scope="module")
def minio_client():
    """Fixture providing a configured MinIO/S3 client.

    Skips tests if MinIO is not available.
    Creates test bucket if needed, cleans up after tests.
    """
    if not is_minio_available():
        pytest.skip("MinIO not available - start with 'docker-compose up minio'")

    # Set env var so checkpoint functions use MinIO
    original_endpoint = os.environ.get("AWS_S3_ENDPOINT_URL")
    original_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    original_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    os.environ["AWS_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    # Create test bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=TEST_BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=TEST_BUCKET)

    yield s3_client

    # Cleanup: delete any leftover test objects
    try:
        response = s3_client.list_objects_v2(Bucket=TEST_BUCKET, Prefix="autodocs/")
        if "Contents" in response:
            for obj in response["Contents"]:
                s3_client.delete_object(Bucket=TEST_BUCKET, Key=obj["Key"])
    except ClientError:
        pass

    # Restore original env vars
    if original_endpoint is not None:
        os.environ["AWS_S3_ENDPOINT_URL"] = original_endpoint
    elif "AWS_S3_ENDPOINT_URL" in os.environ:
        del os.environ["AWS_S3_ENDPOINT_URL"]

    if original_access_key is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = original_access_key
    if original_secret_key is not None:
        os.environ["AWS_SECRET_ACCESS_KEY"] = original_secret_key


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_roundtrip_s3(minio_client):
    """Verify checkpoint can be saved, loaded, and deleted from S3.

    This is the key integration test that validates:
    1. Async wrappers work correctly with real S3
    2. Checkpoint serialization survives S3 storage
    3. All checkpoint fields are preserved through roundtrip
    4. Delete actually removes the checkpoint
    """
    from autodocs.src.checkpoint import (
        AutoDocsCheckpoint,
        AutoDocsPhase,
        ScatterState,
        delete_autodocs_checkpoint,
        download_autodocs_checkpoint,
        upload_autodocs_checkpoint,
    )

    # Create a checkpoint with representative data
    svn_id = "integration-test-svn-12345"
    toml_content = """[document]
goal = "Create architecture documentation"
format = "markdown"

[[sections]]
title = "Overview"
instruction = "Describe the system architecture"
"""

    scatter_state = {
        "Overview": ScatterState(
            file_by_file_content={
                "src/main.py": "Generated content for main.py",
                "src/utils.py": "Generated content for utils.py",
            },
            nodes_processed=50,
            nodes_total=100,
        )
    }

    checkpoint = AutoDocsCheckpoint(
        source_version_node_id=svn_id,
        config_hash="abc123def456gh",
        toml_content=toml_content,
        started_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        current_phase=AutoDocsPhase.SECTION_UPDATE,
        phase_current=50,
        phase_total=100,
        use_tagging=True,
        annotations={"src/main.py": [1, 2, 0], "src/utils.py": [0, 1, 2]},
        scatter_state=scatter_state,
    )

    # Upload checkpoint to MinIO
    await upload_autodocs_checkpoint(checkpoint, TEST_BUCKET, minio_client)

    # Download and verify
    downloaded = await download_autodocs_checkpoint(TEST_BUCKET, svn_id, minio_client)

    assert downloaded is not None, "Checkpoint should be downloaded"
    assert downloaded.source_version_node_id == svn_id
    assert downloaded.config_hash == "abc123def456gh"
    assert downloaded.toml_content == toml_content
    assert downloaded.current_phase == AutoDocsPhase.SECTION_UPDATE
    assert downloaded.phase_current == 50
    assert downloaded.phase_total == 100
    assert downloaded.use_tagging is True

    # Verify annotations preserved
    assert downloaded.annotations is not None
    assert downloaded.annotations["src/main.py"] == [1, 2, 0]

    # Verify scatter_state preserved
    assert downloaded.scatter_state is not None
    assert "Overview" in downloaded.scatter_state
    assert downloaded.scatter_state["Overview"].nodes_processed == 50
    assert (
        downloaded.scatter_state["Overview"].file_by_file_content["src/main.py"]
        == "Generated content for main.py"
    )

    # Delete checkpoint
    await delete_autodocs_checkpoint(TEST_BUCKET, svn_id, minio_client)

    # Verify deletion - should return None
    deleted_check = await download_autodocs_checkpoint(
        TEST_BUCKET, svn_id, minio_client
    )
    assert deleted_check is None, "Checkpoint should be deleted"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_download_missing_returns_none(minio_client):
    """Downloading non-existent checkpoint should return None, not raise."""
    from autodocs.src.checkpoint import download_autodocs_checkpoint

    result = await download_autodocs_checkpoint(
        TEST_BUCKET, "nonexistent-svn-id-xyz", minio_client
    )

    assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_delete_nonexistent_succeeds(minio_client):
    """Deleting non-existent checkpoint should not raise an error."""
    from autodocs.src.checkpoint import delete_autodocs_checkpoint

    # This should not raise - S3 delete_object is idempotent
    await delete_autodocs_checkpoint(
        TEST_BUCKET, "nonexistent-svn-id-for-delete", minio_client
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_overwrite(minio_client):
    """Uploading checkpoint twice should overwrite, not create duplicates."""
    from autodocs.src.checkpoint import (
        AutoDocsCheckpoint,
        AutoDocsPhase,
        delete_autodocs_checkpoint,
        download_autodocs_checkpoint,
        upload_autodocs_checkpoint,
    )

    svn_id = "overwrite-test-svn"

    # Create first checkpoint
    checkpoint_v1 = AutoDocsCheckpoint(
        source_version_node_id=svn_id,
        config_hash="hash_v1",
        toml_content="version 1",
        started_at=datetime.now(UTC),
        current_phase=AutoDocsPhase.ANNOTATING,
        phase_current=10,
        phase_total=100,
    )
    await upload_autodocs_checkpoint(checkpoint_v1, TEST_BUCKET, minio_client)

    # Upload second checkpoint (same svn_id, different data)
    checkpoint_v2 = AutoDocsCheckpoint(
        source_version_node_id=svn_id,
        config_hash="hash_v2",
        toml_content="version 2",
        started_at=datetime.now(UTC),
        current_phase=AutoDocsPhase.SECTION_UPDATE,
        phase_current=75,
        phase_total=100,
    )
    await upload_autodocs_checkpoint(checkpoint_v2, TEST_BUCKET, minio_client)

    # Download should get v2
    downloaded = await download_autodocs_checkpoint(TEST_BUCKET, svn_id, minio_client)

    assert downloaded is not None
    assert downloaded.config_hash == "hash_v2"
    assert downloaded.toml_content == "version 2"
    assert downloaded.current_phase == AutoDocsPhase.SECTION_UPDATE
    assert downloaded.phase_current == 75

    # Cleanup
    await delete_autodocs_checkpoint(TEST_BUCKET, svn_id, minio_client)

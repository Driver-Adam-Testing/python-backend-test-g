import asyncio
import logging
import os
import uuid
from math import ceil
from typing import Any

from database.models_enums import ContentKind
from shared.file_storage.aws_s3_client import org_id_to_hash
from shared.inspector.onboarding.onboard_utils import create_bucket_if_dne
from workflows.autodocs_functions import WriteAutoDocLogInput, write_autodoc_log_task

from .auto_toml.auto_toml import AutoToml
from .autodoc_log import AutoDocLog
from .autodocs_prototype import (
    AutoDocCfg,
    AutoDocInitState,
    ExecutionMode,
    FullyQualifiedDriverPathCode,
    FullyQualifiedDriverPathPdf,
    Scope,
    get_autodoc_elapsed_time,
    update_autodocs_status,
)
from .checkpoint import (
    compute_config_hash,
    delete_autodocs_checkpoint,
    download_autodocs_checkpoint,
    validate_autodocs_checkpoint,
)
from .common import wait_for_guard_duty_tag

logger = logging.getLogger(__name__)


async def run_autodoc(
    version_node_id: uuid.UUID,  # TODO: naming here
    config_kind: Any,  # TODO: the actual type is a deferred import here, not sure how to resolve?
    document_goal: str | None = None,
    user_context: str | None = None,
    content_kind: ContentKind | None = None,
    hatchet_id: str | None = None,
    organization_id: str | None = None,  # For checkpoint bucket computation
) -> None:
    import hashlib

    import boto3
    from database.db import get_session
    from database.models import (
        DerivedContent,
        DocumentSource,
        UserCache,
        Version,
        VersionCreator,
        VersionNode,
    )
    from database.models_enums import (
        AutoDocConfigKind,
        AutoDocStatusMessageKind,
        ContentKind,
        PrimaryAssetKind,
        VersionStatus,
    )
    from sqlalchemy.orm import selectinload
    from sqlmodel import delete, select

    is_page = content_kind == ContentKind.application_note or content_kind is None

    toml_content = ""

    if config_kind == AutoDocConfigKind.FROM_DOCUMENT_GOAL and not document_goal:
        raise ValueError(
            "document_goal is required when config_kind is FROM_DOCUMENT_GOAL"
        )

    # org_id is extracted from DB when possible, falls back to organization_id parameter
    org_id = None

    try:
        # Get document sources given page id
        if is_page:
            with get_session() as session, session.begin():
                document_sources = session.exec(
                    select(DocumentSource)
                    .where(DocumentSource.page_version_node_id == version_node_id)
                    .options(
                        selectinload(DocumentSource.source_version_node)
                        .selectinload(VersionNode.version)
                        .selectinload(Version.primary_asset)
                    )
                ).all()

                scope = Scope(
                    preamble="",
                    code=[],
                    pdfs=[],
                )

                for source in document_sources:
                    if not org_id:
                        org_id = source.source_version_node.version.primary_asset.organization_id
                    if (
                        source.source_version_node.version.primary_asset.kind
                        == PrimaryAssetKind.CODEBASE
                    ):
                        code_cfg = FullyQualifiedDriverPathCode(
                            version_id=str(source.source_version_node.version_id),
                            node_path=source.source_version_node.relative_path.rstrip(
                                "/"
                            ),
                        )
                        scope.code.append(code_cfg)
                    elif (
                        source.source_version_node.version.primary_asset.kind
                        == PrimaryAssetKind.FILE
                    ):
                        pdf_cfg = FullyQualifiedDriverPathPdf(
                            version_id=str(source.source_version_node.version_id),
                            pdf_name=source.source_version_node.version.primary_asset.display_name,
                        )
                        scope.pdfs.append(pdf_cfg)
        else:
            scope = Scope(
                preamble="",
                code=[],
                pdfs=[],
            )
            with get_session() as session, session.begin():
                source_version_node = session.get(VersionNode, version_node_id)
                code_cfg = FullyQualifiedDriverPathCode(
                    version_id=str(source_version_node.version_id),
                    node_path=source_version_node.relative_path.rstrip("/"),
                )
                scope.code.append(code_cfg)

        # Fallback to organization_id parameter if not extracted from DB
        if not org_id and organization_id:
            org_id = organization_id

        match config_kind:
            case AutoDocConfigKind.ADI_DRIVER:
                config = AutoDocCfg.from_file("/autodocs_configs/adi_driver_page.toml")
            case AutoDocConfigKind.ARCHITECTURE:
                config = AutoDocCfg.from_file(
                    "/autodocs_configs/architecture_modal.toml"
                )
            case AutoDocConfigKind.CUSTOM:
                if org_id:
                    bucket = os.environ.get("DROPZONE_BUCKET_NAME")
                    hashed_org_id = hashlib.sha256(org_id.encode()).hexdigest()[:63]
                    key = f"assets/{hashed_org_id}/{version_node_id}/custom_config.toml"

                    s3 = boto3.client("s3")
                    # download the file from S3
                    if wait_for_guard_duty_tag(bucket=bucket, key=key):
                        print(
                            f"Downloading custom config from bucket {bucket} with key {key}."
                        )
                        s3.download_file(
                            bucket,
                            key,
                            "/autodocs_configs/custom_config.toml",
                        )
                    else:
                        raise ValueError(
                            f"GuardDuty tag not found for bucket {bucket} and key {key}. "
                        )
                    config = AutoDocCfg.from_file(
                        "/autodocs_configs/custom_config.toml"
                    )
                    with open("/autodocs_configs/custom_config.toml") as f:
                        toml_content = f.read()

            case AutoDocConfigKind.FROM_DOCUMENT_GOAL:
                toml_uuid = str(uuid.uuid4())
                toml_file = f"config_{toml_uuid}.toml"
                if is_page:
                    auto_toml = AutoToml.from_page_id(
                        version_node_id, enable_auto_scaling=True
                    )
                else:
                    auto_toml = AutoToml.from_root_node_id(
                        root_node_id=version_node_id, enable_auto_scaling=True
                    )
                toml_content = await auto_toml.generate(
                    document_goal=document_goal,
                    user_context=user_context if user_context else "",
                )
                with open(toml_file, "w") as f:
                    f.write(toml_content)
                config = AutoDocCfg.from_file(toml_file=toml_file)
            case _:
                raise ValueError(f"Unsupported config kind: {config_kind}")

        scope.preamble = config.scope.preamble
        config.scope = scope
        print(config.scope)

        # Checkpoint loading (only enabled when toml_content is set, e.g., FROM_DOCUMENT_GOAL)
        bucket = None
        checkpoint = None
        config_hash = ""

        if toml_content and org_id:
            # Compute config hash for checkpoint validation
            config_hash = compute_config_hash(toml_content)

            # Compute bucket from org_id (same pattern as analytics)
            bucket = org_id_to_hash(org_id)

            # Ensure bucket exists (creates if not present)
            await asyncio.to_thread(create_bucket_if_dne, bucket)

            # Try to load existing checkpoint
            checkpoint = await download_autodocs_checkpoint(
                bucket=bucket,
                source_version_node_id=str(version_node_id),
            )

            if checkpoint:
                # Validate checkpoint against current config
                use_tagging = config.document.use_tagging
                if validate_autodocs_checkpoint(checkpoint, config_hash, use_tagging):
                    logger.info(
                        f"Resuming from checkpoint: phase={checkpoint.current_phase}, "
                        f"progress={checkpoint.phase_current}/{checkpoint.phase_total}"
                    )
                    # Use saved TOML content from checkpoint for section consistency
                    if checkpoint.toml_content:
                        toml_content = checkpoint.toml_content
                        config = AutoDocCfg.from_string(toml_content)
                        config.scope = scope  # Preserve the scope we built
                        logger.info("Using TOML content from checkpoint")
                else:
                    logger.warning(
                        "Checkpoint invalid (config/version mismatch), starting fresh"
                    )
                    checkpoint = None
        elif not toml_content:
            logger.debug("Checkpointing disabled: no toml_content (legacy config flow)")
        elif not org_id:
            logger.warning("Checkpointing disabled: no organization_id available")

        init_state = await AutoDocInitState.from_cfg(
            cfg=config,
            execution_mode=ExecutionMode.MODAL,
            page_version_node_id=version_node_id,
        )
        doc = await init_state.generate(
            execution_mode=ExecutionMode.MODAL,
            source_version_node_id=str(version_node_id),
            hatchet_id=hatchet_id,
            bucket=bucket,
            checkpoint=checkpoint,
            toml_content=toml_content,
            config_hash=config_hash,
        )
        elapsed_time_s = await get_autodoc_elapsed_time(
            source_version_node_id=str(version_node_id), hatchet_id=hatchet_id
        )
        elapsed_time_min = ceil(elapsed_time_s / 60)
        if elapsed_time_min == 1:
            doc += f" in {elapsed_time_min} minute"
        else:
            doc += f" in {elapsed_time_min} minutes"
        await update_autodocs_status(
            source_version_node_id=str(version_node_id),
            status_kind=AutoDocStatusMessageKind.GENERATION_COMPLETE,
            content=doc,
            hatchet_id=hatchet_id,
        )

        sections = []
        section_refs = []
        for (
            section_title,
            source_list,
        ) in init_state.section_sources.items():
            section = f"{section_title}\n\n" + "\n".join(source_list)
            sections.append(section)
            section_refs.append((section_title, source_list))
        source_string = "\n\n".join(sections)

        # dataset to return
        name = None
        user_context_str = user_context
        sources = section_refs
        config_content = toml_content
        doc_content = doc

        if is_page:
            with get_session() as session, session.begin():
                derived_content = session.exec(
                    select(DerivedContent)
                    .join(VersionNode, DerivedContent.node_id == VersionNode.node_id)
                    .where(
                        VersionNode.id == version_node_id,
                    )
                ).first()
                if not derived_content:
                    print("No existing derived content found for this page.")
                    return
                else:
                    derived_content.content = doc
                page_version_node = session.exec(
                    select(VersionNode)
                    .where(VersionNode.id == version_node_id)
                    .options(
                        selectinload(VersionNode.version).selectinload(
                            Version.primary_asset
                        )
                    )
                ).one()

                page_version_node.version.status = VersionStatus.GENERATION_COMPLETE
                session.add(page_version_node.version)

                user_cache = session.exec(
                    select(UserCache)
                    .join(VersionCreator, UserCache.id == VersionCreator.user_id)
                    .where(VersionCreator.version_id == page_version_node.version_id)
                ).first()

                env = os.environ.get("MODAL_ENVIRONMENT")

                name = derived_content.content_name if derived_content else "UNKNOWN"
                if env in ["prod", "staging"]:
                    print("Writing AutoDoc log to Notion")

                    log = AutoDocLog(
                        title=name,
                        user_email=user_cache.email if user_cache else "UNKNOWN",
                        organization_id=page_version_node.version.primary_asset.organization_id
                        if page_version_node
                        else "UNKNOWN",
                        sources=source_string,
                        toml_content=toml_content,
                        autodoc_content=doc,
                        user_context=user_context,
                        env=env,
                        page_id=str(version_node_id),
                        config_kind=str(config_kind),
                    )
                    log_input = WriteAutoDocLogInput(log=log)

                    write_autodoc_log_task.aio_run(log_input)

            print("Updated derived content for page version node:", version_node_id)
        else:
            with get_session() as session, session.begin():
                source_version_node = session.get(VersionNode, version_node_id)
                dc_delete_query = delete(DerivedContent).where(
                    DerivedContent.node_id == source_version_node.node_id,
                    DerivedContent.content_kind == content_kind,
                )
                session.exec(dc_delete_query)

                derived_content = DerivedContent(
                    node_id=source_version_node.node_id,
                    relative_path=source_version_node.relative_path,
                    content_kind=content_kind,  # TODO: doc kind as input
                    content=doc,
                    misc_metadata=None,
                )
                session.add(derived_content)

        # Clean up checkpoint after successful completion
        if bucket:
            try:
                await delete_autodocs_checkpoint(bucket, str(version_node_id))
                logger.info(f"Deleted autodocs checkpoint for {version_node_id}")
            except Exception as e:
                # Deletion failure should not fail the job
                logger.warning(f"Failed to delete checkpoint (non-fatal): {e}")

        return (
            content_kind,
            name,
            user_context_str,
            sources,
            config_content,
            doc_content,
        )

    except Exception as e:
        print("Error:", e)
        await update_autodocs_status(
            source_version_node_id=str(version_node_id),
            status_kind=AutoDocStatusMessageKind.GENERATION_ERROR,
            content="An error as has occurred during generation.",
            hatchet_id=hatchet_id,
        )
        with get_session() as session, session.begin():
            version_node = session.get(VersionNode, version_node_id)
            version_node.version.status = VersionStatus.GENERATION_ERROR
            session.add(version_node.version)
        raise e

import asyncio
import uuid

from database.models_enums import ContentKind
from shared.inspector.utils.dag import FlatTopoFileDiffDag
from workflows.autodocs_workflow import AutodocInput, autodocs_task
from workflows.deep_context_functions import MakeChangelogInput, make_changelog_task


async def deep_context_docs(
    old_version_id: uuid.UUID | None,
    old_version_content: list | None,
    code_diff: FlatTopoFileDiffDag | None,
    new_version_id: uuid.UUID,
    install_id: str | None,
) -> list:
    from database.db import async_engine
    from database.models import DerivedContent
    from database.models_enums import AutoDocConfigKind
    from shared.inspector.utils.db import get_version_by_id
    from shared.inspector.utils.synthesis.deep_context import (
        DeepContextDocKind,
    )
    from shared.inspector.utils.synthesis.deep_context_prompts import (
        ARCHITECTURE_OVERVIEW_INTENT,
        LLM_ONBOARDING_INTENT,
    )
    from sqlmodel import delete
    from sqlmodel.ext.asyncio.session import AsyncSession

    new_version = await get_version_by_id(new_version_id)
    root_version_node_id = new_version.root_version_node.id
    root_node_id = new_version.root_version_node.node_id
    organization_id = new_version.primary_asset.organization_id

    if old_version_content and code_diff:
        print(
            f"Updating deep context docs for new version ({new_version_id}) from old version ({old_version_id})"
        )
        # TODO: Fix, actually pull this data from storage when we have it. Mirroring what is done
        # TODO: in `run_autodoc` when possible or empty if lost (e.g., the sources list and config
        # TODO: content supposed to contain the TOML as a string).
        update_set = {
            DeepContextDocKind.ARCHITECTURE,
            DeepContextDocKind.LLM_ONBOARDING,
        }

        update_tasks = [
            doc.update_from_diff(diff_collection=code_diff)
            for doc in old_version_content
            if doc.doc_kind in update_set
        ]
        if install_id is not None:
            changelog_input = MakeChangelogInput(
                version_id=str(new_version_id),
                install_id=install_id,
                previous_version_id=str(old_version_id),
            )
            await make_changelog_task.aio_run(
                changelog_input,
            )
        completed_docs = await asyncio.gather(*update_tasks)

        # NOTE: the IO takes place in the other modal functions for changelog and fresh autodocs
        # but doing it here for update case since it all runs in main thread anyway.
        async with AsyncSession(async_engine) as session, session.begin():
            for doc in completed_docs:
                content_kind = doc.doc_kind.into_content_kind()
                dc_delete_query = delete(DerivedContent).where(
                    DerivedContent.node_id == root_node_id,
                    DerivedContent.content_kind == content_kind,
                )
                await session.exec(dc_delete_query)
                new_dc = DerivedContent(
                    node_id=root_node_id,
                    relative_path=new_version.root_version_node.relative_path,
                    content_kind=content_kind,
                    content=doc.doc_content,
                    misc_metadata=None,
                )
                session.add(new_dc)
    else:
        print("Creating deep context docs for version:", new_version_id)
        deep_context_doc_tasks = [
            autodocs_task.aio_run(
                AutodocInput(
                    version_node_id=str(root_version_node_id),
                    config_kind=AutoDocConfigKind.FROM_DOCUMENT_GOAL,
                    document_goal=ARCHITECTURE_OVERVIEW_INTENT,
                    user_context="SHORT",
                    content_kind=ContentKind.DEEP_CONTEXT_ARCHITECTURE,
                    organization_id=organization_id,
                )
            ),
            autodocs_task.aio_run(
                AutodocInput(
                    version_node_id=str(root_version_node_id),
                    config_kind=AutoDocConfigKind.FROM_DOCUMENT_GOAL,
                    document_goal=LLM_ONBOARDING_INTENT,
                    user_context="SHORT",
                    content_kind=ContentKind.DEEP_CONTEXT_LLM_ONBOARDING,
                    organization_id=organization_id,
                )
            ),
        ]
        if install_id is not None:
            changelog_input = MakeChangelogInput(
                version_id=str(new_version_id),
                install_id=install_id,
                previous_version_id=str(old_version_id) if old_version_id else None,
            )
            deep_context_doc_tasks.append(
                make_changelog_task.aio_run(
                    changelog_input,
                )
            )
        completed_docs = []

        await asyncio.gather(*deep_context_doc_tasks)

    return {"status": "completed"}  # TODO:

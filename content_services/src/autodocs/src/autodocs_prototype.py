import argparse
import asyncio
import concurrent.futures
import copy
import hashlib
import json
import logging
import os
import tempfile
import tomllib
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import IntEnum, StrEnum
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Self

import boto3
import modal
import pymupdf4llm
import tqdm
from aiolimiter import AsyncLimiter
from botocore.config import Config
from database.models_enums import AutoDocStatusMessageKind, ContentKind
from google import genai
from hatchet_sdk.exceptions import FailedTaskRunExceptionGroup
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from shared.agent.chat_openai_async import ChatOpenAI, OutputConfig, OutputConfigKind
from shared.chunking.text_splitter import get_num_tokens, split_text
from shared.prompts.structured_prompting import (
    GENERAL_STE_STYLE_INSTRUCTION,
    USE_BACKTICKS_STYLE_INSTRUCTION,
    USE_TRIPLE_BACKTICS_FOR_CODE_BLOCKS_STYLE_INSTRUCTION,
    Component,
    Prompt,
)
from tqdm.asyncio import tqdm_asyncio
from workflows.autodocs_functions import LLMGenerateInput, llm_generate_task

from .checkpoint import (
    AutoDocsCheckpoint,
    AutoDocsPhase,
    ScatterState,
)

try:
    with open("local_setup.json") as f:
        LOCAL_FILES: dict[str, str] = json.load(f)
except FileNotFoundError:
    LOCAL_FILES = None

OPENAI_SEM = asyncio.Semaphore(75)
PDF_DOWNLOAD_DIR = "pdfs/"
OPENAI_LIMITER = AsyncLimiter(50, 1)  # 50 requests per second
MAX_CONCURRENT_ANNOTATIONS = 50

logger = logging.getLogger(__name__)


async def llm_generate(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> str:
    try:
        async with OPENAI_SEM, OPENAI_LIMITER:
            llm_generate_input = LLMGenerateInput(
                model=llm.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            result = await llm_generate_task.aio_run(llm_generate_input)
            return result["result"]  # hatchet needs to return a dict
    except (
        FailedTaskRunExceptionGroup
    ) as e:  # This exception occurs as a result of a timeout
        token_ct = get_num_tokens(system_prompt + user_prompt)
        try:
            print(
                f"retrying llm_generate because of the following exception:\n\n{e}\n\ntoken count = {token_ct}"
            )
            async with OPENAI_SEM, OPENAI_LIMITER:
                return await llm_generate_task.aio_run(llm_generate_input)
        except modal.exception.FunctionTimeoutError as e:
            print(
                f"retrying llm_generate failed because of the following exception:\n\n{e}\n\ntoken count = {token_ct}"
            )
            raise


GREEN = "\033[92m"  # Green
RED = "\033[91m"  # Red
CYAN = "\033[96m"  # Cyan
BLUE = "\033[94m"  # Blue
ORANGE = "\033[38;5;214m"  # Orange
RESET = "\033[0m"  # Reset color to default


class Category(IntEnum):
    HighlyRelevant = 0
    SomewhatRelevant = 1
    Irrelevant = 2

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls(int(s))


class ExecutionMode(StrEnum):
    LOCAL = "local"
    MODAL = "modal"


class NamedFlag(BaseModel):
    index: int
    name: str
    flag: bool


class SectionFlags(BaseModel):
    sections: list[NamedFlag]

    @classmethod
    async def from_llm(
        cls,
        llm: ChatOpenAI,
        goal: str,
        preamble: str,
        optional_sections: list[tuple[str, str, str]],
        long_descriptions: str,
    ) -> Self:
        system_prompt_template = """
You are an expert engineer and technical writer that specializes in documenting software.

Your job is to review the long description of source code provided to you and decide if certain topics are relevant or not. Specifically, we are considering if certain sections are relevant and should be included in a document we are writing to describe some code. Here is a description of the goal of the document we are wrting:

{goal}
{preamble_content}

The optional sections we are considering are described are listed below with the format <section index> <section name>: <description of section>:

{optional_section_descriptions}

Your job is to decide if these sections are relevant or not looking by considering descriptions of the code that will be provided to you. You return your decisions in a list, with the index, section name, and your decision for that section name (as a boolean value, where true indicates the section should be included and false indicates the section should not be included), provided for each section. The index is provided because it is possible for there to be more than one section with the same name, wherein the index as well as the description will differentiate them. Make sure the order, index, and section name in your output list matches the order provided to you as input.
"""
        user_prompt = f"Here is a collection of long descriptions for files and folders associated with the code. This is the context for you to decide if the optional sections are relevant:\n\n{long_descriptions}"
        preamble_content = (
            f"\nHere is further context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )

        optional_section_descriptions = ""
        for idx, _level, name, desc in optional_sections:
            optional_section_descriptions += f"- {idx} {name}: {desc}\n"
        system_prompt = system_prompt_template.format(
            goal=goal,
            preamble_content=preamble_content,
            optional_section_descriptions=optional_section_descriptions,
        )

        return cls.model_validate_json(
            await llm.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_cfg=OutputConfig(kind=OutputConfigKind.JSON_STRICT, payload=cls),
            )
        )


def _get_path_on_disk(codebase_name: str) -> Path:
    return Path(LOCAL_FILES[codebase_name]["abs_path_of_root_loc"]) / (
        codebase_name + ".json"
    )


def _get_pdf_paths(pdf_names: list[str], execution_mode: ExecutionMode) -> Path:
    match execution_mode:
        case ExecutionMode.LOCAL:
            pdf_paths = [
                Path(LOCAL_FILES[p]["abs_path_of_root_loc"]) / p for p in pdf_names
            ]
        case ExecutionMode.MODAL:
            pdf_paths = [Path(PDF_DOWNLOAD_DIR) / p for p in pdf_names]

    return pdf_paths


def _get_target_name(path: str) -> str:
    return Path(path).name


def _get_codebase_name(path: str) -> str:
    for p in Path(path).parts:
        if p != os.sep:
            return p

    raise ValueError(f"Could not construct codebase name for path: `{path}`")


async def update_autodocs_status(
    source_version_node_id: str,
    status_kind: AutoDocStatusMessageKind,
    content: str,
    hatchet_id: str | None = None,
) -> None:
    from database.db import async_engine
    from database.models import AutoDocStatusHistory
    from sqlmodel.ext.asyncio.session import AsyncSession

    async with AsyncSession(async_engine) as session, session.begin():
        status_update = AutoDocStatusHistory(
            source_version_node_id=source_version_node_id,
            status_kind=status_kind,
            content=content,
            call_id=hatchet_id,
        )
        session.add(status_update)
        await session.commit()


async def get_autodoc_elapsed_time(
    source_version_node_id: str, hatchet_id: str | None = None
) -> float:
    from database.db import async_engine
    from database.models import AutoDocStatusHistory
    from sqlmodel import select
    from sqlmodel.ext.asyncio.session import AsyncSession

    async with AsyncSession(async_engine) as session:
        states = (
            await session.exec(
                select(AutoDocStatusHistory)
                .where(
                    AutoDocStatusHistory.source_version_node_id
                    == source_version_node_id,
                    AutoDocStatusHistory.call_id == hatchet_id,
                )
                .order_by(AutoDocStatusHistory.created_at.asc())
            )
        ).all()
        start_state = states[0]
        end_state = states[-1]
        elapsed_time = end_state.created_at - start_state.created_at
    return elapsed_time.total_seconds()


class TechDocsContent(BaseModel):
    name: str
    source: str | None
    short_sentence_description: str
    long_description: str
    short_paragraph_description: str
    split_source: str | None = None
    split_scatter_user_prompt: str | None = None


class DriverDocsContent(BaseModel):
    codebase_name: str
    version_id: str
    dag: dict[str, set[str]]
    content: dict[str, TechDocsContent]
    topo_order: list[str]

    def to_disk(self, p: Path) -> None:
        as_json = self.model_dump_json()
        with open(p, "w") as f:
            f.write(as_json)

    @classmethod
    def from_disk(cls, p: Path) -> Self:
        with open(p) as f:
            json_raw = f.read()

        return cls.model_validate_json(json_raw)

    @classmethod
    def from_db(cls, version_id: str, relative_path: str) -> Self:
        from database.db import get_session
        from database.models import Version
        from database.models_enums import ContentKind
        from sqlalchemy.orm import selectinload
        from sqlmodel import select

        with get_session() as session:
            version = session.exec(
                select(Version)
                .where(Version.id == version_id)
                .options(selectinload(Version.primary_asset))
            ).first()
            codebase_name = version.primary_asset.display_name
            primary_asset_id = version.primary_asset_id
            organization_id = version.primary_asset.organization_id
        bucket = hashlib.sha256(organization_id.encode()).hexdigest()[:63]

        ss = _get_derived_contents(
            version_id=version_id,
            relative_path=relative_path,
            dc_kind=ContentKind.SHORT_SENTENCE_DESCRIPTION,
        )
        ld = _get_derived_contents(
            version_id=version_id,
            relative_path=relative_path,
            dc_kind=ContentKind.LONG_DESCRIPTION,
        )
        sp = _get_derived_contents(
            version_id=version_id,
            relative_path=relative_path,
            dc_kind=ContentKind.SHORT_PARAGRAPH_DESCRIPTION,
        )
        # TODO: this will download everything right now, vs. just the subgraph of interest
        with tempfile.TemporaryDirectory() as download_dir:
            try:
                # Parallelize S3 downloads using ThreadPoolExecutor
                content = {}
                s3_client = boto3.client("s3", config=Config(max_pool_connections=50))
                with ThreadPoolExecutor(max_workers=10) as executor:
                    # Submit all download tasks
                    future_to_key = {
                        executor.submit(
                            _get_source_from_s3,
                            s3_client,
                            version_id,
                            primary_asset_id,
                            k,
                            bucket,
                            download_dir,
                        ): k
                        for k in ss
                    }

                    # Collect results as they complete
                    with tqdm.tqdm(
                        total=len(future_to_key), desc="Downloading files"
                    ) as pbar:
                        for future in as_completed(future_to_key):
                            k = future_to_key[future]
                            try:
                                source = future.result()
                                content[k] = TechDocsContent(
                                    name=k,
                                    source=source,
                                    short_sentence_description=ss[k],
                                    long_description=ld[k],
                                    short_paragraph_description=sp[k],
                                )
                            except Exception as exc:
                                print(f"Error downloading {k}: {exc}")
                                # Create entry with None source on error
                                content[k] = TechDocsContent(
                                    name=k,
                                    source=None,
                                    short_sentence_description=ss[k],
                                    long_description=ld[k],
                                    short_paragraph_description=sp[k],
                                )
                            pbar.update(1)
            except Exception as e:
                print(codebase_name)
                raise e

            dag = build_file_tree_dag(
                codebase_name=codebase_name,
                content=content,
                codebase_root=download_dir,
                exeuction_mode=ExecutionMode.MODAL,
            )
            toposort = TopologicalSorter(dag)

            root_short_paragraph = content[codebase_name].short_paragraph_description
            for k in content:
                if content[k].source is not None:
                    try:
                        chunks = split_text(
                            content[k].source,
                            chunk_size=64_000,
                            chunk_overlap=0,
                        )
                        content[k].split_source = chunks[0].text
                    except Exception as e:
                        print(f"Error splitting source for {k}: {e}")
                        content[k].split_source = None
                else:
                    content[k].split_source = None

                content[k].split_scatter_user_prompt = _scatter_user_prompt_constructor(
                    root_short_paragraph=root_short_paragraph,
                    node_long_description=ld[k],
                    node_source=content[k].source,
                    node_path=k,
                )

        return DriverDocsContent(
            codebase_name=codebase_name,
            version_id=version_id,
            dag=dag,
            content=content,
            topo_order=list(toposort.static_order()),
        )

    def walk_topo(self) -> Generator[tuple[str, TechDocsContent], None, None]:
        return ((p, self.content[p]) for p in self.topo_order)


def _scatter_user_prompt_constructor(
    root_short_paragraph: str,
    node_long_description: str,
    node_source: str | None,
    node_path: str,
) -> str:
    user_prompt = (
        Prompt.empty()
        .append(
            Component(
                string=(
                    f"Short description of the full codebase:\n\n{root_short_paragraph}\n\n"
                )
            )
        )
        .append(
            Component(
                string=f"DESCRIPTION OF `{node_path}`:\n\n{node_long_description}\n\n"
            )
        )
    )
    if node_source is not None:
        if len(node_source.strip()) > 0:
            user_prompt.append(
                Component(
                    string=f"Source code for  `{node_path}`:\n\n{node_source}\n\n"
                )
            )
        else:
            user_prompt.append(
                Component(string=f"Source code for  `{node_path}`:\n\nEmpty file\n\n")
            )
    user_prompt_str = user_prompt.into_str()
    chunks = split_text(user_prompt_str, chunk_size=96_000, chunk_overlap=0)
    if len(chunks) > 1:
        return chunks[0].text
    return user_prompt_str


def _get_derived_contents(
    version_id: str, relative_path: str, dc_kind: ContentKind
) -> dict[str, str]:
    from database.db import get_session
    from database.models import DerivedContent, VersionNode
    from sqlmodel import select

    with get_session() as session:
        dc_query = (
            select(DerivedContent, VersionNode)
            .join(VersionNode, DerivedContent.node_id == VersionNode.node_id)
            .where(
                VersionNode.version_id == version_id,
                VersionNode.relative_path.like(f"{relative_path}%"),
                DerivedContent.content_kind.in_([dc_kind]),
            )
        )
        rows = session.exec(dc_query).all()
        return {vn.relative_path.rstrip("/"): dc.content for (dc, vn) in rows}


def _get_source_from_s3(
    s3_client: boto3.client,
    version_id: str,
    primary_asset_id: str,
    relative_path: str,
    bucket: str,
    download_dir: str,
) -> str:
    try:
        # s3_client = boto3.client("s3")
        download_key = f"{primary_asset_id}/{version_id}/{relative_path}"
        local_download_path = Path(download_dir) / relative_path
        local_download_path.parent.mkdir(parents=True, exist_ok=True)
        # print(bucket, download_key)
        s3_client.download_file(bucket, download_key, str(local_download_path))

        with open(local_download_path) as f:
            return f.read()
    except Exception:
        return None


def _download_pdf_from_s3(version_id: str) -> str:
    from database.db import get_session
    from database.models import Version, VersionNode
    from sqlalchemy.orm import selectinload
    from sqlmodel import select

    with get_session() as session:
        version = session.exec(
            select(Version)
            .where(Version.id == version_id)
            .options(selectinload(Version.primary_asset))
        ).first()
        primary_asset_id = version.primary_asset_id
        pdf_name = version.primary_asset.display_name
        organization_id = version.primary_asset.organization_id

        version_node = session.exec(
            select(VersionNode).where(VersionNode.version_id == version_id)
        ).first()

    bucket = hashlib.sha256(organization_id.encode()).hexdigest()[:63]
    download_dir = Path(PDF_DOWNLOAD_DIR)
    download_dir.mkdir(exist_ok=True)
    local_download_path = download_dir / pdf_name

    s3_client = boto3.client("s3")
    download_key = f"{primary_asset_id}/{version_id}/{version_node.relative_path}"
    s3_client.download_file(bucket, download_key, str(local_download_path))


def build_file_tree_dag(
    codebase_name: str,
    content: dict[str, TechDocsContent],
    codebase_root: str | None,
    exeuction_mode: ExecutionMode,
) -> dict[str, set[str]]:
    if exeuction_mode == ExecutionMode.LOCAL and codebase_root is None:
        codebase_root = Path(LOCAL_FILES[codebase_name]["abs_path_of_root_loc"])
    elif codebase_root is not None:
        codebase_root = Path(codebase_root)
    included_nodes = {str(k) for k in content}
    dag = dict()

    for (
        local_root,
        dirs,
        files,
    ) in os.walk(codebase_root / codebase_name):
        children = set()
        for d in dirs:
            root_rel_path = str((Path(local_root) / d).relative_to(codebase_root))
            if root_rel_path in included_nodes:
                children.add(root_rel_path)
        for f in files:
            root_rel_path = str((Path(local_root) / f).relative_to(codebase_root))
            if root_rel_path in included_nodes:
                dag[root_rel_path] = set()
                children.add(root_rel_path)
        local_root_rel_path = str(Path(local_root).relative_to(codebase_root))
        if local_root_rel_path in included_nodes:
            dag[local_root_rel_path] = children

    return dag


def build_subgraph(dag: dict[str, set[str]], start: str) -> dict[str, set[str]] | None:
    if start not in dag:
        print(f"Node {start} is not present in the DAG.")
        return None

    subgraph = dict()

    def dfs(node: str) -> None:
        if node in subgraph:
            # We've already visited this node.
            return
        # Add the node to the subgraph with a copy of its children.
        subgraph[node] = dag[node]
        for child in dag[node]:
            dfs(child)

    dfs(start)
    return subgraph


class DocKind(StrEnum):
    DEFINED_SECTIONS = "defined_sections"
    UNDEFINED = "undefined"
    FROM_EXAMPLE = "from_example"
    ARCHITECTURE = "architecture"


class SectionCreationMethod(StrEnum):
    SEQUENTIAL_EDIT = "sequential_edit"
    SCATTER_GATHER = "scatter_gather"
    ONLY_PDFS = "only_pdfs"
    CODE_EXAMPLE = "code_example"


class LlmCfg(BaseModel):
    tag_model: str
    section_init_model: str
    section_update_model: str
    section_format_model: str
    assembly_model: str
    copy_editor_model: str

    @classmethod
    def default(cls) -> Self:
        return cls(
            tag_model="gpt-4.1",
            section_init_model="o3-mini",
            section_update_model="gpt-4.1",
            section_format_model="gpt-5",
            assembly_model="o3-mini",
            copy_editor_model="gpt-4.1",
        )


class DocumentCfg(BaseModel):
    goal: str
    fmt: DocKind
    use_tagging: bool
    config_name: str
    config_version: str

    @classmethod
    def default(cls) -> Self:
        return cls(
            goal="",
            fmt=DocKind.DEFINED_SECTIONS,
            use_tagging=True,
            config_name="",
            config_version="",
        )


class FullyQualifiedDriverPathPdf(BaseModel):
    version_id: str
    pdf_name: str


class FullyQualifiedDriverPathCode(BaseModel):
    version_id: str
    node_path: str


class Scope(BaseModel):
    preamble: str
    pdfs: list[FullyQualifiedDriverPathPdf]
    code: list[FullyQualifiedDriverPathCode]

    @classmethod
    def default(cls) -> Self:
        return cls(
            preamble="",
            pdfs=[],
            code=[],
        )


class SectionCfg(BaseModel):
    title: str
    level: int
    required: bool  # this setting is ignored when committed_with is not None
    instruction: str
    content_structure: str
    section_creation_method: SectionCreationMethod
    committed_with: str | None

    @classmethod
    def default(cls) -> Self:
        return cls(
            title="",
            level=1,
            required=True,
            instruction="",
            content_structure="",
            section_creation_method=SectionCreationMethod.SCATTER_GATHER,
            committed_with=None,
        )


class SectionCommitted(BaseModel):
    title: str
    level: int
    instruction: str
    content_structure: str
    section_creation_method: SectionCreationMethod

    @classmethod
    def from_section_cfg(cls, cfg: SectionCfg) -> Self:
        return cls(
            title=cfg.title,
            level=cfg.level,
            instruction=cfg.instruction,
            content_structure=cfg.content_structure,
            section_creation_method=cfg.section_creation_method,
        )

    def annotation_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer that specializes in annotating files and folders of a codebase for their relevance to writing a specific section for a specific document.

The specific document being written has the following goal:

{goal}
{preamble_content}

The specific section of the document you are assessing relevance for has the following description:

{heading} {title}
{instruction}

Your job is to review source code provided to you and decide which of the following categories it belongs to:

**Highly Relevant**: This means the file contains critical information to write about this specific section.

**Somewhat Relevant**: This means the file contains some information needed to write about this specific section, but it is not necessarily critical.

**Irrelevant**: This means the file contains content that should not be considered when writing this section because it is not relevant. It is important to identify irrelevant content so we do not consider the wrong information or add noise into the process of writing the specific section of the specific document.

You will be given the source code for an entire file and will respond with a single number corresponding to the category you identify for the source code. Only output this single number:

- 0 for a Highly Relevant file
- 1 for a Somewhat Relevant file
- 2 for an Irrelevant file
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return system_prompt_template.format(
            goal=goal,
            preamble_content=preamble_content,
            heading=heading,
            title=self.title,
            instruction=self.instruction,
        )

    def pdf_annotation_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer that specializes in annotating files and folders of a codebase for their relevance to writing a specific section for a specific document.

The specific document being written has the following goal:

{goal}
{preamble_content}

The specific section of the document you are assessing relevance for has the following description:

{heading} {title}
{instruction}

Your job is to review a page from a pdf provided to you and decide which of the following categories it belongs to:

**Highly Relevant**: This means the page contains critical information to write about this specific section.

**Somewhat Relevant**: This means the page contains some information needed to write about this specific section, but it is not necessarily critical.

**Irrelevant**: This means the page contains content that should not be considered when writing this section because it is not relevant. It is important to identify irrelevant content so we do not consider the wrong information or add noise into the process of writing the specific section of the specific document.

You will be given a single page from a pdf in markdown rendered text from and will respond with a single number corresponding to the category you identify for the source code. Only output this single number:

- 0 for a Highly Relevant file
- 1 for a Somewhat Relevant file
- 2 for an Irrelevant file
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return system_prompt_template.format(
            goal=goal,
            preamble_content=preamble_content,
            heading=heading,
            title=self.title,
            instruction=self.instruction,
        )

    def init_draft_system_prompt_code(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write an initial draft of one section in a larger document.

You will be given a high level description of the content in a codebase that you should use to write the content for you section. This high level description will pertain to one or more root folders and their immediate children or just a single file. This is a limited set of information.

The goal is to generate a starting point for the section of your document. In subsequent writing steps, others will refine this document based on your initial draft. Because you have limited information, the most valuable thing you can do is provide a good outline and structure. Fill in content for documents of the document as best you can, but focus on a solid structure with subsections that are most meaningful and relevant to the context given to you. For example, since you will not be looking at source code directly, you should not try and provide any source code examples.

The section you are writing about is titled {title}. Here is the a description of the kind of content you should include for the section:

{heading} {title}
{instruction}

Your output should be markdown formatted text including the section title as a top level header, important subsections, and content included for each subsection as appropriate.
        """
        preamble_content = (
            f"\nHere is further context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def init_draft_system_prompt_pdf(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write a draft of a section in a larger document based on the content of the PDF.

The section you are writing about is titled {title}. Here is a description of the kind of content you should include in the section:

{heading} {title}
{instruction}

Here is a description of how your output should be formated:

{content_structure}

Your output should be markdown formatted text.
        """
        preamble_content = (
            f"\nHere is further context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                        content_structure=self.content_structure,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def update_from_file_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to take a current version of a section for a larger document that will be provided to you and update it, as appropriate, from details of a particular file that will also be provided to you. The contents follow Markdown syntax, which you will also follow in any updates you make. This is part of an iterative process wherein an initial, high level draft is updated with details from critical files and folders.

The details of the particular file will include a description of the contents of the file in human language as well as the source code.

Your job is to update the existing section content by editing or adding details based only on the content of the particular file provided. You will have access to the source code of just this one particular file, so you should focus on using this to add detail (descriptive, conceptual, technical) to the section content.

It is very important to add and include significant technical details from the source code. Here is a description of the section you are editing and instruction on what content should be in this section:

{heading} {title}
{instruction}

The details of this file may not provide information relevant to this section. If there are no clear and meaningful updates to make to the section draft based on the particular file contents, then do not make any edits.

Your output is the full content for the section of the document you have been provided with updates made based on your analysis of the particular file contents provided to you.

Your expected audience is a technical engineer.
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def update_from_pdf_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to take a current version of a section for a larger document that will be provided to you and update it, as appropriate, from details of a particular page from a pdf that will also be provided to you. The contents follow Markdown syntax, which you will also follow in any updates you make. This is part of an iterative process wherein an initial, high level draft is updated with details from critical files and pdf pages.

You will be provided the pdf page as markdown rendered text.

Your job is to update the existing section content by editing or adding details based only on the content of the particular page provided.

It is very important to add and include any significant details from the page. Here is a description of the section you are editing and instruction on what content should be in this section:

{heading} {title}
{instruction}

The details of this pdf page may not provide information relevant to this section. If there are no clear and meaningful updates to make to the section draft based on the particular file contents, then do not make any edits.

Your output is the full content for the section of the document you have been provided with updates made based on your analysis of the particular file contents provided to you.

Your expected audience is a technical engineer.
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def update_from_folder_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to take a current version of a section for a larger document that will be provided to you and update it, as appropriate, from details of a particular folder that will also be provided to you. The contents follow Markdown syntax, which you will also follow in any updates you make. This is part of an iterative process wherein an initial, high level draft is updated with details from critical files and folders.

The details of the particular folder will include a brief description of all of the child files and subfolders of the folder.

Your job is to update the existing section content by editing or adding details based only on the content of the particular folder provided. Since you will only have access to summary information (brief descriptions of all children of this folder), you should focus on higher level organization and structure to the section instead of modifying or adding technical details.

Here is a description of the section you are editing and instruction on what content should be in this section:

{heading} {title}
{instruction}

The details of this folder may not provide information relevant to this section. If there are no clear and meaningful updates to make to the section draft based on the particular folder contents, then do not make any edits.

Your output is the full content for the section of the document you have been provided with updates made based on your analysis of the particular folder contents provided to you.

Your expected audience is a technical engineer.
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def final_output_format(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer, technical writer, and copy editor. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write the final form of the {title} section for a document using Markdown syntax. You will be given detailed content for this section built up iteratively. Your job is to edit, rewrite, and reformat this content to conform to the following structure:

{heading} {title}
{content_structure}

Your output is the content of the section reformatted to fit the described content structure and your expected audience is a technical engineer.
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        content_structure=self.content_structure,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def scatter_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write a section a larger document through the lens of a relevant file, page from a pdf, or folder.

You will be given the source content from a single file or a page from a pdf identified to be relevant to the section. This is a limited set of information.

The goal is to write a version for the section of your document using just the context from this file.

The section you are writing about is titled {title}. Here is the a description of the kind of content you should include for the section:

{heading} {title}
{instruction}

Your output should be markdown formatted text including the section title as a top level header, important subsections, and content included for each subsection as appropriate.

Technical detail is very important in this document. As you write the document, cite specific examples from the source code or pdf page to support your documentation.

Use only content directly from the source code or pdf page to write the document. Do not make up any content that is not directly from the source code or pdf page.

It is okay to just return "no relevant content" if the source code or pdf page does not provide any relevant content for the section.
"""
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def gather_system_prompt(self, goal: str, preamble: str) -> str:
        aggregate_system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write a section of a larger document.

You will be given many sections written about a set of files identified to be highly relevant to the section.

The goal is to generate the section by aggregating the information from these file specific sections.

The section you are writing is titled {title}. Here is the a description of the kind of content you should include for the section:

{heading} {title}
{instruction}

Your output should be a document with important sections/subsections using Markdown syntax with content included for each subsection as appropriate.

Technical detail is very important in this document. Try to keep as much technical detail from each file as possible, but combine and organize the information in a logical way.

Use only content directly from the sections you've been given in your aggregation. Do not make up any content that is not directly from the provided sections. Do not change code examples, copy these directly from the source sections provided.
"""
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=aggregate_system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def gather_multiple_system_prompt(self, goal: str, preamble: str) -> str:
        aggregate_system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write a section of a larger document.

You will be given many sections written about aggregate sets of files identified to be highly relevant to the section.

The goal is to generate the section by aggregating the infromation from these aggregate file specific sections.

The section you are writing is titled {title}. Here is the a description of the kind of content you should include for the section:

{heading} {title}
{instruction}

Your output should be a document with important sections/subsections using Markdown syntax with content included for each subsection as appropriate.

Technical detail is very important in this document. Try to keep as much technical detail from each section as possible, but combine and organize the information in a logical way.

Use only content directly from the sections you've been given in your aggregation. Do not make up any content that is not directly from the provided sections. Do not change code examples, copy these directly from the source sections provided.
"""
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=aggregate_system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def code_example_single_pass_system_prompt(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to write a draft a code example that will be used in a larger document based on the content of the code provided.

You will be provided source code of one or more source files deemed to be relevant to the code example you are constructing.

The section you are writing a code example for is titled: {title}. Here is a description of the kind of content you should include in the section:

{heading} {title}
{instruction}

Here is a description of how your output should be formated:

{content_structure}

Your output should be markdown formatted text.
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                        content_structure=self.content_structure,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .append(USE_TRIPLE_BACKTICS_FOR_CODE_BLOCKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def code_example_single_pass_aggregate_pass(self, goal: str, preamble: str) -> str:
        system_prompt_template = """
You are an expert software engineer and technical writer. You specialize in writing documents with the following goal:

{goal}

Your job is to write a draft a code example that will be used in a larger document based on the content of the code provided.

You will be provided code examples generated using different files deemed to be relevant to the code example you are constructing.

Your goal is to combine these code examples into one coherent example that exemplifies the section you're writing a code example for: {title}.

Take care not to make up intermediate code that does not clearly exist from what is given to you. In combining the many examples initially given to you, look for opportunities to combine examples into larger and more complete examples, but only if that is the correct choice for the purpose of the section. Alternatively, you can and should also discard some incoming example content because it is not relevant or less relevant to the goal of the section.

Here is a description of the kind of content you should include in the section:

{heading} {title}
{instruction}

Here is a description of how your output should be formated:

{content_structure}

Your output should be markdown formatted text.
        """
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{preamble}"
            if preamble.strip() != ""
            else ""
        )
        heading = "#" * self.level
        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=goal,
                        preamble_content=preamble_content,
                        heading=heading,
                        title=self.title,
                        instruction=self.instruction,
                        content_structure=self.content_structure,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .append(USE_TRIPLE_BACKTICS_FOR_CODE_BLOCKS_STYLE_INSTRUCTION)
            .into_str()
        )

    async def code_example_few_shot_generator(
        self,
        document_goal: str,
        document_preamble: str,
        reverse_topos: list,
        tagged_nodes: dict[str, list[Category]] | None,
        tag_idx: int | None,
    ) -> str:
        llm = ChatOpenAI(model="gpt-5", temperature=0, request_timeout=300)
        # TODO: this is done naively - if we need to do this, we should keep related content together
        # could be useful to leverage symbol table here
        user_prompts = self.source_code_aggregation_user_prompt_constructor(
            reverse_topos=reverse_topos,
            annotations=tagged_nodes,
            tag_idx=tag_idx,
        )
        coroutines = []
        for user_prompt in user_prompts:
            coroutines.append(
                llm_generate(
                    llm=llm,
                    system_prompt=self.code_example_single_pass_system_prompt(
                        goal=document_goal, preamble=document_preamble
                    ),
                    user_prompt=user_prompt,
                )
            )
        responses = await asyncio.gather(*coroutines)

        # TODO: could be interesting to do a pass with the symbol table here, e.g.
        # identify the symbols used in the code examples and pass those in with the
        # generated code examples to keep some amount of context
        if len(responses) > 1:
            print(
                f"{self.title} requires aggregation of {len(responses)} code examples..."
            )
            aggregate_user_prompt = ""
            for idx, response in enumerate(responses):
                aggregate_user_prompt += (
                    f"Code example for set {idx}:\n\n{response}\n\n"
                )
            response = await llm_generate(
                llm=llm,
                system_prompt=self.code_example_single_pass_aggregate_pass(
                    goal=document_goal,
                    preamble=document_preamble,
                ),
                user_prompt=aggregate_user_prompt,
            )
            return response
        else:
            return responses[0]

    async def create_section_scatter_gather(
        self,
        goal: str,
        preamble: str,
        reverse_topos: list,
        pdf_pages_dict: dict[str, list[str]] | None,
        tagged_nodes: dict | None,
        pdf_tagged_nodes: dict | None,
        tag_idx: int | None,
        section_name: str,
        checkpoint: AutoDocsCheckpoint,
        bucket: str,
        scatter_state: dict[str, ScatterState] | None = None,
    ) -> str:
        print(f"Creating node sections for {section_name}...")
        file_by_file_content = await self.create_node_sections(
            goal,
            preamble,
            reverse_topos,
            pdf_pages_dict,
            tagged_nodes,
            pdf_tagged_nodes,
            tag_idx,
            section_title=section_name,
            checkpoint=checkpoint,
            bucket=bucket,
            scatter_state=scatter_state,
        )

        aggregate_docs = await self.aggregate_node_sections(
            goal,
            preamble,
            file_by_file_content,
            section_name,
        )

        while len(aggregate_docs) > 1:
            aggregate_docs = await self.aggregate_aggregate_sections(
                goal,
                preamble,
                aggregate_docs,
                section_name,
            )
        return aggregate_docs[0]

    async def create_node_sections(
        self,
        goal: str,
        preamble: str,
        reverse_topos: list,
        pdf_pages_dict: dict[str, list[str]] | None,
        tagged_nodes: dict | None,
        pdf_tagged_nodes: dict | None,
        tag_idx: int | None,
        section_title: str | None = None,
        checkpoint: AutoDocsCheckpoint | None = None,
        bucket: str | None = None,
        scatter_state: dict[str, ScatterState] | None = None,
    ) -> dict:
        init_model = "o3-mini"
        llm = ChatOpenAI(model=init_model, request_timeout=500, temperature=0)

        # Check if we're resuming from a checkpoint for this section
        existing_content: dict[str, str] = {}
        nodes_already_processed = 0
        if section_title and scatter_state and section_title in scatter_state:
            existing_state = scatter_state[section_title]
            existing_content = existing_state.file_by_file_content
            nodes_already_processed = existing_state.nodes_processed
            logger.info(
                f"Resuming scatter for '{section_title}': "
                f"{nodes_already_processed} nodes already processed"
            )

        file_by_file_content = dict(existing_content)  # Start with existing content
        node_coroutines = []
        ordered_nodes = []

        for reverse_topo in reverse_topos:
            root_p, root_content = reverse_topo[0]
            # Always force at least the root of every dag to be used to generate the section
            ordered_nodes.append(root_p)
            user_prompt = root_content.split_scatter_user_prompt
            node_coroutines.append(
                llm_generate(
                    llm=llm,
                    system_prompt=self.scatter_system_prompt(goal, preamble),
                    user_prompt=user_prompt,
                )
            )
            for p, tech_docs in reverse_topo[1:]:
                # print(f"Generating node section for {p}...")
                if (
                    tag_idx is None
                    or tagged_nodes is None
                    or (
                        tagged_nodes[p][tag_idx] == Category.HighlyRelevant
                        or tagged_nodes[p][tag_idx] == Category.SomewhatRelevant
                    )
                ):
                    ordered_nodes.append(p)

                    user_prompt = tech_docs.split_scatter_user_prompt
                    node_coroutines.append(
                        llm_generate(
                            llm=llm,
                            system_prompt=self.scatter_system_prompt(goal, preamble),
                            user_prompt=user_prompt,
                        )
                    )

        print("Generating node sections for pdfs...")
        for pdf_path in pdf_pages_dict:
            for idx, page_content in enumerate(pdf_pages_dict[pdf_path]):
                if (
                    pdf_tagged_nodes is None
                    or tag_idx is None
                    or pdf_tagged_nodes[pdf_path][idx][tag_idx]
                    == Category.HighlyRelevant
                ):
                    ordered_nodes.append(str(pdf_path) + f" page {idx}")
                    user_prompt = f"Page content from {pdf_path}:\n\n{page_content}"
                    node_coroutines.append(
                        llm_generate(
                            llm=llm,
                            system_prompt=self.scatter_system_prompt(goal, preamble),
                            user_prompt=user_prompt,
                        )
                    )

        # Do it in batches to reduce heartbeat errors in Hatchet
        MAX_CONCURRENT_SCATTER_SECTIONS = 100
        total_nodes = len(node_coroutines)
        print(f"Generating {total_nodes} node sections...")

        for i in range(0, len(node_coroutines), MAX_CONCURRENT_SCATTER_SECTIONS):
            # Skip batches that were already processed (resume case)
            if i + MAX_CONCURRENT_SCATTER_SECTIONS <= nodes_already_processed:
                continue

            batch = node_coroutines[i : i + MAX_CONCURRENT_SCATTER_SECTIONS]
            batch_nodes = ordered_nodes[i : i + MAX_CONCURRENT_SCATTER_SECTIONS]

            # For partial batch resume, skip nodes already in file_by_file_content
            if i < nodes_already_processed:
                # We're in the middle of a partially-completed batch
                skip_count = nodes_already_processed - i
                batch = batch[skip_count:]
                batch_nodes = batch_nodes[skip_count:]
                if not batch:
                    continue

            print(
                f"Generating batch {i // MAX_CONCURRENT_SCATTER_SECTIONS + 1} "
                f"with {len(batch)} node sections..."
            )
            batch_responses = await asyncio.gather(*batch)
            for ordered_node, response in zip(batch_nodes, batch_responses):
                file_by_file_content[ordered_node] = response

            # Update scatter_state and checkpoint after each batch
            if (
                checkpoint is not None
                and bucket
                and section_title
                and scatter_state is not None
            ):
                current_nodes_processed = min(
                    i + MAX_CONCURRENT_SCATTER_SECTIONS, total_nodes
                )
                state = ScatterState(
                    file_by_file_content=file_by_file_content,
                    nodes_processed=current_nodes_processed,
                    nodes_total=total_nodes,
                )
                scatter_state[section_title] = state
                checkpoint.update_phase(
                    AutoDocsPhase.SECTION_UPDATE,
                    current=current_nodes_processed,
                    total=total_nodes,
                )
                checkpoint.update_scatter_state(section_title, state)
                await checkpoint.save(bucket)
                logger.info(
                    f"Scatter checkpoint for '{section_title}': "
                    f"{current_nodes_processed}/{total_nodes} nodes"
                )

        print(f"Created {len(file_by_file_content)} node sections")
        return file_by_file_content

    async def aggregate_node_sections(
        self,
        goal: str,
        preamble: str,
        file_by_file_content: dict,
        section_name: str,
    ) -> list:
        model = "gpt-5"
        llm = ChatOpenAI(model=model, request_timeout=500, temperature=0)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            user_prompts = await loop.run_in_executor(
                pool,
                self.gather_user_prompt_constructor,
                file_by_file_content,
                section_name,
            )

        aggregate_coroutines = []
        aggregate_docs = []

        for prompt in user_prompts:
            aggregate_coroutines.append(
                llm_generate(
                    llm=llm,
                    system_prompt=self.gather_system_prompt(goal, preamble),
                    user_prompt=prompt,
                )
            )
        print(
            f"Aggregating nodes with {len(aggregate_coroutines)} buckets for {section_name}"
        )
        aggregate_responses = await asyncio.gather(*aggregate_coroutines)
        for response in aggregate_responses:
            aggregate_docs.append(response)
        return aggregate_docs

    async def aggregate_aggregate_sections(
        self,
        goal: str,
        preamble: str,
        aggregate_docs: list,
        section_name: str,
    ) -> list:
        model = "gpt-5"
        llm = ChatOpenAI(model=model, request_timeout=500, temperature=0)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            user_prompts = await loop.run_in_executor(
                pool,
                self.gather_aggregate_user_prompt_constructor,
                aggregate_docs,
                section_name,
            )
        aggregate_coroutines = []
        new_aggregate_docs = []
        for prompt in user_prompts:
            aggregate_coroutines.append(
                llm_generate(
                    llm=llm,
                    system_prompt=self.gather_multiple_system_prompt(goal, preamble),
                    user_prompt=prompt,
                )
            )
        print(
            f"Aggregating aggregates with {len(aggregate_coroutines)} buckets for {section_name}"
        )
        aggregate_responses = await asyncio.gather(*aggregate_coroutines)
        for response in aggregate_responses:
            new_aggregate_docs.append(response)
        return new_aggregate_docs

    def source_code_aggregation_user_prompt_constructor(
        self,
        reverse_topos: list,
        annotations: dict[str, list[Category]] | None,
        tag_idx: int | None,
        chunk_size: int = 100_000,
    ) -> list:
        user_prompt = ""
        for reverse_topo in reverse_topos:
            for p, tech_docs in reverse_topo:
                if (
                    tag_idx is None
                    or annotations is None
                    or (annotations[p][tag_idx] == Category.HighlyRelevant)
                ) and tech_docs.source is not None:
                    user_prompt += (
                        f"Source code of file `{p}`:\n\n{tech_docs.source}\n\n"
                    )
        chunks_required = split_text(
            user_prompt, chunk_size=chunk_size, chunk_overlap=0
        )
        if len(chunks_required) > 1:
            num_tokens = get_num_tokens(user_prompt)
            token_threshold = num_tokens / len(chunks_required)
            user_prompts = [""]
            for reverse_topo in reverse_topos:
                for p, tech_docs in reverse_topo:
                    if (
                        tag_idx is None
                        or annotations is None
                        or (
                            annotations[p][tag_idx] == Category.HighlyRelevant
                            or annotations[p][tag_idx] == Category.SomewhatRelevant
                        )
                    ):
                        new_prompt = (
                            f"Source code of file `{p}`:\n\n{tech_docs.source}\n\n"
                        )
                        new_tokens = get_num_tokens(new_prompt)
                        found_prompt = False
                        for idx, prompt in reversed(list(enumerate(user_prompts))):
                            if new_tokens + get_num_tokens(prompt) > token_threshold:
                                pass
                            else:
                                user_prompts[idx] += new_prompt
                                found_prompt = True
                                break
                        if not found_prompt:
                            user_prompts.append(new_prompt)
        else:
            user_prompts = [user_prompt]

        return user_prompts

    def gather_user_prompt_constructor(
        self,
        file_by_file_content: dict,
        section_name: str,
    ) -> list:
        chunk_size = 64_000  # in TOKENS

        # Pre-format each file's section and compute tokens once per section
        items = list(file_by_file_content.items())

        formatted_sections = []
        section_token_counts = []
        total_tokens = 0

        for p, content in items:
            s = f"{section_name} of file or folder `{p}`:\n\n{content}\n\n"
            formatted_sections.append(s)
            t = get_num_tokens(s)  # one tokenization per file
            section_token_counts.append(t)
            total_tokens += t

        # If everything fits in one chunk, just join once and return
        if total_tokens <= chunk_size:
            return ["".join(formatted_sections)]

        # how many chunk_size buckets are needed for total_tokens?
        chunks_est = total_tokens / chunk_size

        token_threshold = total_tokens / chunks_est

        # Build chunks greedily (preserve input order), tracking token sums
        prompts_parts = [[]]  # list[list[str]]
        prompts_token_sums = [0]  # parallel list[int]

        for s, t in zip(formatted_sections, section_token_counts):
            placed = False
            # Try to place into most recent chunk first
            for idx in range(len(prompts_parts) - 1, -1, -1):
                # Don't exceed average threshold AND never exceed hard chunk_size
                next_sum = prompts_token_sums[idx] + t
                if next_sum <= token_threshold and next_sum <= chunk_size:
                    prompts_parts[idx].append(s)
                    prompts_token_sums[idx] = next_sum
                    placed = True
                    break

            if not placed:
                # Start a new chunk; if a single section is larger than threshold,
                # it can still occupy its own chunk (up to chunk_size).
                prompts_parts.append([s])
                prompts_token_sums.append(t)

        # Join buffers into final strings
        user_prompts = ["".join(parts) for parts in prompts_parts if parts]
        return user_prompts

    def gather_aggregate_user_prompt_constructor(
        self,
        aggregate_docs: list,
        section_name: str,
    ) -> list:
        # TODO: multiple prompt handling... That would be VERY large
        user_prompts = [""]
        for idx, doc in enumerate(aggregate_docs):
            user_prompts[0] += f"{section_name} doc for set {idx}:\n\n{doc}\n\n"
            if get_num_tokens(user_prompts[0]) > 100_000:
                return user_prompts
        return user_prompts


class AutoDocCfg(BaseModel):
    llm: LlmCfg
    document: DocumentCfg
    scope: Scope
    sections: list[SectionCfg]

    @classmethod
    def _apply_defaults_and_validate(cls, raw_data: dict) -> Self:
        """Apply default values and validate the config.

        This is the shared implementation for from_file() and from_string().
        AutoTOML generates minimal TOML that relies on defaults being applied.
        """
        llm_raw_default = LlmCfg.default().model_dump()
        if "llm" in raw_data:
            raw_data["llm"] = {**llm_raw_default, **raw_data["llm"]}
        else:
            raw_data.setdefault("llm", llm_raw_default)

        document_raw_default = DocumentCfg.default().model_dump()
        if "document" in raw_data:
            raw_data["document"] = {**document_raw_default, **raw_data["document"]}
        else:
            raw_data.setdefault("document", document_raw_default)

        scope_raw_default = Scope.default().model_dump()
        if "scope" in raw_data:
            raw_data["scope"] = {**scope_raw_default, **raw_data["scope"]}
        else:
            raw_data.setdefault("scope", scope_raw_default)

        sections_raw_default = SectionCfg.default().model_dump()
        if "sections" in raw_data:
            raw_data["sections"] = [
                {**sections_raw_default, **section} for section in raw_data["sections"]
            ]
        else:
            raw_data.setdefault("sections", [sections_raw_default])

        cfg = AutoDocCfg.model_validate(raw_data)

        if cfg.document.goal == "":
            raise ValueError("A document goal is required.")
        for section in cfg.sections:
            if section.title == "":
                raise ValueError("A section title is required.")
            if section.instruction == "":
                raise ValueError("A section instruction is required.")
            if section.content_structure == "":
                raise ValueError("A section content structure is required.")

        if "substitutions" in raw_data:
            mapping = {item["key"]: item["value"] for item in raw_data["substitutions"]}
            for section in cfg.sections:
                section.instruction = section.instruction.format_map(mapping)
                section.content_structure = section.content_structure.format_map(
                    mapping
                )

        return cfg

    @classmethod
    def from_file(cls, toml_file: str) -> Self:
        with open(toml_file, "rb") as f:
            raw_data = tomllib.load(f)
        return cls._apply_defaults_and_validate(raw_data)

    @classmethod
    def from_string(cls, toml_content: str) -> Self:
        """Load config from a TOML string.

        Applies the same default merging as from_file() since AutoTOML
        generates minimal TOML that relies on defaults being applied.
        """
        raw_data = tomllib.loads(toml_content)
        return cls._apply_defaults_and_validate(raw_data)

    async def eval_optional_sections(
        self, llm: ChatOpenAI, long_descriptions: str
    ) -> list[SectionCommitted]:
        optional_sections = [
            (idx, s.level, s.title, s.instruction)
            for idx, s in enumerate(self.sections)
            if s.required is False and s.committed_with is None
        ]
        if len(optional_sections) > 0:
            print(
                f"Optional sections for target scope:\n{[title for _, _, title, _ in optional_sections]}\n\n({BLUE}{llm.model}{RESET}) Evaluating optional sections for inclusion..."
            )
            flags = await SectionFlags.from_llm(
                llm=llm,
                goal=self.document.goal,
                preamble=self.scope.preamble,
                optional_sections=optional_sections,
                long_descriptions=long_descriptions,
            )
            included_idxs = set()
            for (idx, level, title, instruction), sf in zip(
                optional_sections, flags.sections
            ):
                if idx != sf.index or title != sf.name:
                    raise ValueError(
                        f"Optional section evaluation returned invalid section: {sf.index} {sf.name}. Expected: {idx} {title}"
                    )
                if sf.flag:
                    print(
                        f"{GREEN}+ {'#' * level} {sf.name}{RESET} ({instruction[:100]} ...)"
                    )
                    included_idxs.add(idx)
                else:
                    print(
                        f"{RED}- {'#' * level} {sf.name}{RESET} ({instruction[:100]} ...)"
                    )

            for idx, section_cfg in enumerate(self.sections):
                if section_cfg.committed_with:
                    for parent_idx, parent_cfg in enumerate(self.sections):
                        if parent_cfg.title == section_cfg.committed_with and (
                            parent_cfg.required or parent_idx in included_idxs
                        ):
                            included_idxs.add(idx)
                            break

            return [
                SectionCommitted.from_section_cfg(cfg)
                for idx, cfg in enumerate(self.sections)
                if (
                    (cfg.required is True and cfg.committed_with is None)
                    or idx in included_idxs
                )
            ]
        else:
            return [SectionCommitted.from_section_cfg(cfg) for cfg in self.sections]


def _autogen_sections(cfg: AutoDocCfg) -> list[SectionCfg]:
    raise NotImplementedError("TODO")


def build_state_filename(scope_roots: list[str], fmt: str, ext: str = ".json") -> str:
    targets = list(
        dict.fromkeys(
            [_get_target_name(code_cfg.node_path) for code_cfg in scope_roots]
        )
    )[:3]
    name = "_".join(targets)
    return f"{name}_{fmt}_iterations{ext}"


def build_annotations_filename(
    scope_roots: list[FullyQualifiedDriverPathCode], fmt: str, ext: str = ".json"
) -> str:
    targets = list(
        dict.fromkeys(
            [_get_codebase_name(code_cfg.node_path) for code_cfg in scope_roots]
        )
    )[:3]
    name = "_".join(targets)
    return f"{name}_{fmt}_annotations{ext}"


class AutoDocInitState(BaseModel):
    llm: LlmCfg
    document: DocumentCfg
    scope: Scope
    sections: list[SectionCommitted]
    driver_docs: list[DriverDocsContent] = []
    _source_list: dict[str, list[str]] = {}

    @property
    def section_sources(self) -> str:
        return self._source_list

    def assembly_system_prompt(self) -> str:
        system_prompt_template = """
You are an expert software engineer, technical writer, and copy editor. You specialize in writing documents with the following goal:

{goal}
{preamble_content}

Your job is to assemble a final draft of a document based on a particular structure of sections. You will be given set of detailed content for each section, together covering all of the content we want in the final document.

The content for each section was built up iteratively and these detailed documents were developed independently from each other.

Your job is to combine the information and produce a single high quality, detailed document. Specifically, your primary focus is on consolidating all of the different section content into a single cohesive document.

Your output is the complete document, with all sections combined using Markdown format. Your expected audience is a technical engineer.

I've provided the top-level sections you should use with a description of the kind of content that should be included for each section and the output format expected for that content below. The detailed contents for each section should have all of this content, but you shold focus on making sure the transition between sections is smooth and remove any major redundancies. Use the following top-level structure and output format for the sections of the final document:

{sections}
        """
        sections = ""
        for section in self.sections:
            heading = "#" * section.level
            sections += f"{heading} {section.title}\nContent:{section.instruction}\nFormat:{section.content_structure}\n\n"
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{self.scope.preamble}"
            if self.scope.preamble.strip() != ""
            else ""
        )

        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=self.document.goal,
                        preamble_content=preamble_content,
                        sections=sections,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .append(USE_TRIPLE_BACKTICS_FOR_CODE_BLOCKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def final_copy_editor_system_prompt(self) -> str:
        system_prompt_template = """
You are an expert software engineer, technical writer, and copy editor that specializes in documenting software.

Your job is to provide copy editing for a complete draft of a document using Markdown syntax. The goal of this document is:

{goal}
{preamble_content}

The following section structure should appear in the document. Other subsections are OK, but at least these sections need to be present:

{sections}

This draft was built up iteratively over time.

Your goal is to provide final polish and edits to produce a complete, coherent, and high quality final document. Your job is not to comment or change the content of the document. Focus only on typical copy editing duties:
- Make sure all content is formatted with proper Markdown syntax, but **do not** enclose the whole document in a Markdown codefenced block. You are merely ensuring the overall content follows Markdown syntax.
  - Ensure the syntax for header and list hierarchies are correct.
  - Ensure the syntax for bulleted and numbered lists is correct.
  - Ensure the syntax for fenced source code blocks, if present, is correct.
- Remove any references to this being a draft document, early draft, or iterative draft.
- Keep all technical or conceptual details.
- Make sure transitions between sections and subsections flow smoothly.
- For any MermaidJS diagrams, make sure to not have any parentheses in the MermaidJS content (for example, in the labels for components of the diagram). Parentheses will lead to syntax errors in parsing and rendering the diagram and cannot be allowed. Additionally, make sure labels and names in the diagram are relatively short.

Your output is the full content of the document with editing updates based on your analysis as a copy editor.
        """
        sections = ""
        for section in self.sections:
            heading = "#" * section.level
            sections += f"{heading} {section.title}\n\n"
        preamble_content = (
            f"\nHere is further overall context about the document we are writing:\n\n{self.scope.preamble}"
            if self.scope.preamble.strip() != ""
            else ""
        )

        return (
            Prompt.empty()
            .append(
                Component(
                    string=system_prompt_template.format(
                        goal=self.document.goal,
                        preamble_content=preamble_content,
                        sections=sections,
                    )
                )
            )
            .append(GENERAL_STE_STYLE_INSTRUCTION)
            .append(USE_BACKTICKS_STYLE_INSTRUCTION)
            .append(USE_TRIPLE_BACKTICS_FOR_CODE_BLOCKS_STYLE_INSTRUCTION)
            .into_str()
        )

    def _generate_sources_list(
        self,
        annotations: dict[str, list[Category]] | None,
        pdf_annotations: dict[str, list[Category]] | None,
    ) -> dict[str, list[str]]:
        sources_dict = {}

        if not annotations and not pdf_annotations:
            return sources_dict

        for idx, section in enumerate(self.sections):
            sources = []
            if annotations:
                sources.extend(
                    path
                    for path, categories in annotations.items()
                    if categories[idx] == Category.HighlyRelevant
                )

            if pdf_annotations:
                sources.extend(
                    f"{pdf_path!s} (page {page_idx + 1})"
                    for pdf_path, pages in pdf_annotations.items()
                    for page_idx, categories in pages.items()
                    if categories[idx] == Category.HighlyRelevant
                )

            sources_dict[section.title] = sources

        return sources_dict

    def to_disk(self, json_p: Path) -> None:
        with open(json_p) as f:
            f.write(self.model_dump_json())

    @classmethod
    def from_disk(cls, json_p: Path) -> Self:
        with open(json_p) as f:
            state = json.load(f)
        return cls(**state["cfg"])

    @classmethod
    async def from_cfg(
        cls,
        cfg: AutoDocCfg,
        execution_mode: ExecutionMode,
        page_version_node_id: str = "",
    ) -> Self:
        preamble_content = (
            f"\nHere is further context about the document we are writing:\n\n{cfg.scope.preamble}"
            if cfg.scope.preamble.strip() != ""
            else ""
        )
        print(
            f"\n🧙 {CYAN}Whizdoodling{RESET}! to create a document with the following goal:\n{cfg.document.goal}\n{preamble_content}\n"
        )
        print(
            f"Configuration: {cfg.document.config_name} {cfg.document.config_version}"
        )
        print(f"Page ID: {page_version_node_id}\n")
        match cfg.document.fmt:
            case DocKind.DEFINED_SECTIONS:
                targets = [
                    [code_cfg.version_id, _get_codebase_name(path=code_cfg.node_path)]
                    for code_cfg in cfg.scope.code
                ]
                match execution_mode:
                    case ExecutionMode.LOCAL:
                        driver_docs = [
                            DriverDocsContent.from_disk(
                                p=_get_path_on_disk(codebase_name=t)
                            )
                            for _, t in targets
                        ]
                    case ExecutionMode.MODAL:

                        def _load_driver_docs(
                            cfg: AutoDocCfg,
                        ) -> list[DriverDocsContent]:
                            return [
                                DriverDocsContent.from_db(
                                    version_id=code_cfg.version_id,
                                    relative_path=code_cfg.node_path,
                                )
                                for code_cfg in cfg.scope.code
                            ]

                        loop = asyncio.get_running_loop()
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as pool:
                            driver_docs = await loop.run_in_executor(
                                pool, _load_driver_docs, cfg
                            )
                    case _:
                        raise ValueError("Invalid execution mode")

                if any(section.required is False for section in cfg.sections):
                    subgraphs = []
                    for dd, code_cfg in zip(driver_docs, cfg.scope.code):
                        subgraph = build_subgraph(dag=dd.dag, start=code_cfg.node_path)
                        if subgraph:
                            subgraphs.append(subgraph)
                    if len(subgraphs) == 0:
                        raise ValueError(
                            "Subgraphs could not be built for any supplied source nodes."
                        )

                    toposorts = [
                        list(TopologicalSorter(sg).static_order()) for sg in subgraphs
                    ]
                    reverse_topos = [
                        [(p, dd.content[p]) for p in ts]
                        for ts, dd in zip(toposorts, driver_docs)
                    ]
                    _ = [rt.reverse() for rt in reverse_topos]

                    user_prompt = ""
                    num_dag_roots = len(reverse_topos)
                    ridx = 1
                    for rt in reverse_topos:
                        root_p, root_content = rt[0]
                        if root_content.source is None:
                            padding = "\n\n" if ridx == 1 else ""
                            user_prompt += f"{padding}Included root folder {ridx} / {num_dag_roots} (`{root_p}`) description:\n\n{root_content.long_description}"
                            for node_p, node_content in rt[1:]:
                                if node_content.source is None:
                                    user_prompt += f"\n\nSubfolder (`{node_p}`) description:\n\n{node_content.long_description}"
                                else:
                                    user_prompt += f"\n\nFile (`{node_p}`) description:\n\n{node_content.long_description}"
                        else:
                            user_prompt += f"\n\nFile (`{root_p}`) description:\n\n{root_content.long_description}"
                        ridx += 1
                    # TODO: Actually figure out how to handle very large aggregations.
                    aggregation_chunks = split_text(
                        text=user_prompt,
                        chunk_size=96_000,
                        chunk_overlap=0,
                    )
                    llm = ChatOpenAI(
                        model=cfg.llm.tag_model, temperature=0, request_timeout=300
                    )
                    long_descriptions = aggregation_chunks[0].text
                    committed_sections = await cfg.eval_optional_sections(
                        llm=llm, long_descriptions=long_descriptions
                    )
                else:
                    committed_sections = [
                        SectionCommitted.from_section_cfg(s) for s in cfg.sections
                    ]
                return cls(
                    llm=cfg.llm,
                    document=cfg.document,
                    scope=cfg.scope,
                    sections=committed_sections,
                    driver_docs=driver_docs,
                )
            case DocKind.UNDEFINED:
                sections = _autogen_sections(cfg=cfg)
                return cls(
                    llm=cfg.llm,
                    document=cfg.document,
                    scope=cfg.scope,
                    sections=sections,
                )
            case _:
                raise NotImplementedError("TODO")

    def save_state(
        self,
        revisions: list[dict[str, Any]],
        init_state: dict[str, Any],
        final_doc_revisions: list[str] | None = None,
    ) -> None:
        state_filename = build_state_filename(
            scope_roots=self.scope.code, fmt=self.document.fmt
        )
        init_state = copy.deepcopy(init_state)
        init_state["appended_reverse_topo"] = [
            (p, td.model_dump_json()) for p, td in init_state["appended_reverse_topo"]
        ]
        init_state["init_node_set"] = list(init_state["init_node_set"])
        state = dict()
        state["init_state"] = init_state
        if final_doc_revisions is not None:
            state["final_doc_revisions"] = final_doc_revisions
        state["revisions"] = revisions
        state["cfg"] = self.model_dump()
        with open(state_filename, "w") as f:
            f.write(json.dumps(state))

    def save_annotations(
        self,
        annotations: dict[str, list[Category]],
    ) -> None:
        state_filename = build_annotations_filename(
            scope_roots=self.scope.code, fmt=self.document.fmt
        )
        state = dict()
        state["annotations"] = annotations
        state["cfg"] = self.model_dump()
        with open(state_filename, "w") as f:
            f.write(json.dumps(state))

    def write_final_output_to_markdown(self, output: str) -> None:
        fmt = "document"
        state_filename = build_state_filename(
            scope_roots=self.scope.code,
            fmt=fmt,
            ext=".md",
        )
        with open(state_filename, "w") as f:
            f.write(output)

    def load_state(self) -> list[dict[str, Any]]:
        state_filename = build_state_filename(
            scope_roots=self.scope.code,
            fmt=self.document.fmt,
        )
        with open(state_filename) as f:
            state = json.load(f)

        state["init_state"]["annotations"] = (
            {
                k: [Category(a) for a in v]
                for k, v in state["init_state"]["annotations"].items()
            }
            if self.document.use_tagging
            else None
        )
        state["init_state"]["appended_reverse_topo"] = [
            (p, TechDocsContent.model_validate_json(td))
            for p, td in state["init_state"]["appended_reverse_topo"]
        ]
        state["init_state"]["init_node_set"] = set(state["init_state"]["init_node_set"])
        return state

    def load_annotations(self) -> list[dict[str, Any]]:
        state_filename = build_annotations_filename(
            scope_roots=self.scope.code,
            fmt=self.document.fmt,
        )
        with open(state_filename) as f:
            state = json.load(f)

        annotations = (
            {k: [Category(a) for a in v] for k, v in state["annotations"].items()}
            if self.document.use_tagging
            else None
        )
        cfg = state["cfg"]
        cfg_cls = AutoDocInitState(**cfg)
        return cfg_cls, annotations

    async def _annotate_file(
        self,
        llm: ChatOpenAI,
        node: tuple[str, TechDocsContent],
    ) -> tuple[str, list[Category]]:
        try:
            tech_docs = node[1]
            assert tech_docs.source is not None

            node_list = []

            user_prompt = f"A single paragraph description of the file to categorize:\n\n{tech_docs.short_paragraph_description}.\n\nThe source code of the file to categorize:\n\n{tech_docs.split_source}"
            async with asyncio.TaskGroup() as tg:
                annotation_task_list = []
                for section in self.sections:
                    annotation_task_list.append(
                        tg.create_task(
                            llm_generate(
                                llm=llm,
                                system_prompt=section.annotation_system_prompt(
                                    goal=self.document.goal,
                                    preamble=self.scope.preamble,
                                ),
                                user_prompt=user_prompt,
                            )
                        )
                    )
            annotations_list = [
                Category.from_str(s=t.result()) for t in annotation_task_list
            ]
            node_list.extend(annotations_list)

        except Exception as e:
            print(f"Error annotating node {node[0]}: {e}")
            return node[0], [Category.Irrelevant for _ in self.sections]

        return node[0], node_list

    async def _annotate_pdf_page(
        self, llm: ChatOpenAI, pdf_text: str, page_idx: int
    ) -> tuple[int, list[Category]]:
        user_prompt = f"The page content of the pdf to categorize:\n\n{pdf_text}"
        annotation_task_list = []
        async with asyncio.TaskGroup() as tg:
            for section in self.sections:
                annotation_task_list.append(
                    tg.create_task(
                        llm_generate(
                            llm=llm,
                            system_prompt=section.pdf_annotation_system_prompt(
                                goal=self.document.goal,
                                preamble=self.scope.preamble,
                            ),
                            user_prompt=user_prompt,
                        )
                    )
                )
        annotations_list = [
            Category.from_str(s=t.result()) for t in annotation_task_list
        ]
        return page_idx, annotations_list

    async def _annotate_nodes(
        self,
        llm: ChatOpenAI,
        topo: list[tuple[str, TechDocsContent]],
        graph: dict[str, set[str]],
        execution_mode: ExecutionMode,
        pdf_pages_dict: dict[str, list[str]],
        checkpoint: AutoDocsCheckpoint,
        bucket: str,
        existing_annotations: dict[str, list[Category]] | None = None,
        existing_pdf_annotations: dict | None = None,
        start_batch_idx: int = 0,
    ) -> dict[str, list[Category]]:
        print(
            f"\n({BLUE}{llm.model}{RESET}) Annotating files for relevance to sections..."
        )
        pidx = 1
        total = len(topo)
        # Initialize with existing annotations if resuming, else empty dict
        tagged_nodes: dict[str, list[Category]] = (
            dict(existing_annotations) if existing_annotations else dict()
        )
        tagged_pdfs = (
            dict(existing_pdf_annotations) if existing_pdf_annotations else dict()
        )

        # Annotate files first
        coroutines = []
        for node in topo:
            if node[1].source is not None:
                coroutines.append(self._annotate_file(llm=llm, node=node))
        # Process this in groups (skip already-processed batches when resuming)
        num_file_coroutines = len(coroutines)
        if start_batch_idx > 0:
            logger.info(
                f"Resuming annotation from batch index {start_batch_idx}/{num_file_coroutines}"
            )
            pidx = start_batch_idx + 1  # Adjust pidx for progress display
        for i in range(start_batch_idx, len(coroutines), MAX_CONCURRENT_ANNOTATIONS):
            batch = coroutines[i : i + MAX_CONCURRENT_ANNOTATIONS]
            print(f"[{pidx} / {total}] Annotating files...")
            node_results = await tqdm_asyncio.gather(*batch)
            for result in node_results:
                tagged_nodes[result[0]] = result[1]
            pidx += len(batch)

            # Mid-annotation checkpoint after each batch
            checkpoint.update_phase(
                AutoDocsPhase.ANNOTATING,
                current=i + len(batch),
                total=num_file_coroutines,
            )
            checkpoint.update_annotations(tagged_nodes)
            await checkpoint.save(bucket)

        # Annotate folders after (since they depend on files)
        for p, tech_docs in topo:
            node_list = []
            if tech_docs.source is None:
                for idx in range(len(self.sections)):
                    if any(
                        tagged_nodes[c][idx] == Category.HighlyRelevant
                        for c in graph[p]
                    ):
                        node_list.append(Category.HighlyRelevant)
                    elif any(
                        tagged_nodes[c][idx] == Category.SomewhatRelevant
                        for c in graph[p]
                    ):
                        node_list.append(Category.SomewhatRelevant)
                    else:
                        node_list.append(Category.Irrelevant)
                tagged_nodes[p] = node_list

        if len(self.scope.pdfs) > 0:
            print(
                f"\n({BLUE}{llm.model}{RESET}) Annotating PDFs for relevance to sections..."
            )
            pidx = 1
            total = len(self.scope.pdfs)
            for pdf_path in pdf_pages_dict:
                coroutines = []
                tagged_pdfs[pdf_path] = dict()
                print(f"[{pidx} / {total}] Annotating `{GREEN}{pdf_path}{RESET}`...")
                for idx, page_content in enumerate(pdf_pages_dict[pdf_path]):
                    coroutines.append(
                        self._annotate_pdf_page(
                            llm=llm, pdf_text=page_content, page_idx=idx
                        )
                    )
                pdf_results = await tqdm_asyncio.gather(*coroutines)
                for result in pdf_results:
                    tagged_pdfs[pdf_path][result[0]] = result[1]

        return tagged_nodes, tagged_pdfs

    async def _initialize_sections(
        self,
        llm: ChatOpenAI,
        reverse_topos_from_start: list[list[str]],
        driver_docs: list[DriverDocsContent],
        annotations: dict[str, list[Category]],
        pdf_annotations: dict,
        execution_mode: ExecutionMode,
        pdf_pages_dict: dict[str, list[str]],
        checkpoint: AutoDocsCheckpoint,
        bucket: str,
        existing_scatter_state: dict[str, ScatterState] | None = None,
    ) -> tuple[set[str], list[dict[str, str]]]:
        if self.scope.pdfs and any(
            s.section_creation_method == SectionCreationMethod.ONLY_PDFS
            for s in self.sections
        ):
            pdf_paths = _get_pdf_paths(
                [pdf_cfg.pdf_name for pdf_cfg in self.scope.pdfs],
                execution_mode=execution_mode,
            )
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            gemini_model_id = "gemini-2.0-flash"
            pdf_handles = []
            for pdf in pdf_paths:
                pdf_handles.append(
                    client.files.upload(file=pdf, config={"display_name": pdf.name})
                )
        else:
            pdf_handles = None

        init_sections_dict = {}
        init_node_set = set()
        # Content from code.
        # Build context from root + first level children only.
        # TODO: the way sections are created could use a refactor - right now, each method is done in sequence
        # We should consider having a function for each method that can be awaited simulatenously
        if any(
            s.section_creation_method == SectionCreationMethod.SEQUENTIAL_EDIT
            for s in self.sections
        ):
            raw_user_prompt = ""
            num_roots = len(reverse_topos_from_start)
            ridx = 1
            for dag_topo, docs in zip(reverse_topos_from_start, driver_docs):
                root_p, root_content = dag_topo[0]
                init_node_set.add(root_p)
                if root_content.source is None:
                    padding = "\n\n" if ridx == 1 else ""
                    raw_user_prompt += f"{padding}Included root folder {ridx} / {num_roots} (`{root_p}`) description:\n\n{root_content.long_description}"
                    children = list(docs.dag[root_p])
                    for child in children:
                        init_node_set.add(child)
                        child_content = docs.content[child]
                        if child_content.source is None:
                            raw_user_prompt += f"\n\nSubfolder (`{child}`) description:\n\n{child_content.long_description}"
                        else:
                            raw_user_prompt += f"\n\nFile (`{child}`) description:\n\n{child_content.long_description}"
                else:
                    raw_user_prompt += f"\n\nFile (`{root_p}`) description:\n\n{root_content.long_description}"
            # TODO: Actually figure out how to handle very large aggregations.
            aggregation_chunks = split_text(
                text=raw_user_prompt,
                chunk_size=96_000,
                chunk_overlap=0,
            )
            user_prompt = aggregation_chunks[0].text

            # TODO: This de-mixing and then re-mixing between lists and dicts is ugly.
            # TODO: Do this better, assume need to separate async part out this way for now.

            # Use `idx` to preserve order info and guard against non-unique section titles.
            prompt_pairs_dict_code = {
                idx: (
                    s.init_draft_system_prompt_code(
                        goal=self.document.goal, preamble=self.scope.preamble
                    ),
                    user_prompt,
                )
                for idx, s in enumerate(self.sections)
                if s.section_creation_method == SectionCreationMethod.SEQUENTIAL_EDIT
            }
            async with asyncio.TaskGroup() as tg:
                init_section_tasks_code = dict()
                for k, (system_prompt, user_prompt) in prompt_pairs_dict_code.items():
                    init_section_tasks_code[k] = tg.create_task(
                        llm_generate(
                            llm=llm,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                    )
            init_sections_dict = {
                k: v.result() for k, v in init_section_tasks_code.items()
            }

        if any(
            s.section_creation_method == SectionCreationMethod.SCATTER_GATHER
            for s in self.sections
        ):
            # Create shared scatter_state dict for mid-scatter checkpointing
            # Start from existing state if resuming from checkpoint
            scatter_state: dict[str, ScatterState] = (
                dict(existing_scatter_state) if existing_scatter_state else {}
            )

            scatter_gather_indices = []
            scatter_gather_coroutines = []
            for idx, s in enumerate(self.sections):
                if s.section_creation_method == SectionCreationMethod.SCATTER_GATHER:
                    print(f"creating {s.title} via scatter-gather...")
                    scatter_gather_indices.append(idx)
                    scatter_gather_coroutines.append(
                        s.create_section_scatter_gather(
                            goal=self.document.goal,
                            preamble=self.scope.preamble,
                            reverse_topos=reverse_topos_from_start,
                            pdf_pages_dict=pdf_pages_dict,
                            tagged_nodes=annotations,
                            pdf_tagged_nodes=pdf_annotations,
                            tag_idx=idx,
                            section_name=s.title,
                            checkpoint=checkpoint,
                            bucket=bucket,
                            scatter_state=scatter_state,
                        )
                    )
            scatter_gather_results = await asyncio.gather(*scatter_gather_coroutines)
            for idx, result in zip(scatter_gather_indices, scatter_gather_results):
                init_sections_dict[idx] = result

        if any(
            s.section_creation_method == SectionCreationMethod.CODE_EXAMPLE
            for s in self.sections
        ):
            code_example_single_pass_indices = []
            code_example_single_pass_coroutines = []
            for idx, s in enumerate(self.sections):
                if s.section_creation_method == SectionCreationMethod.CODE_EXAMPLE:
                    print(f"creating {s.title} via code example single pass...")
                    code_example_single_pass_indices.append(idx)
                    code_example_single_pass_coroutines.append(
                        s.code_example_few_shot_generator(
                            document_goal=self.document.goal,
                            document_preamble=self.scope.preamble,
                            reverse_topos=reverse_topos_from_start,
                            tagged_nodes=annotations,
                            tag_idx=idx,
                        )
                    )
            code_example_single_pass_results = await asyncio.gather(
                *code_example_single_pass_coroutines
            )
            for idx, result in zip(
                code_example_single_pass_indices, code_example_single_pass_results
            ):
                init_sections_dict[idx] = result

        # Content from PDFs.
        prompt_dict_pdf = {
            idx: s.init_draft_system_prompt_pdf(
                goal=self.document.goal, preamble=self.scope.preamble
            )
            for idx, s in enumerate(self.sections)
            if s.section_creation_method == SectionCreationMethod.ONLY_PDFS
        }
        for k, prompt in prompt_dict_pdf.items():
            contents = [prompt]
            contents.extend(pdf_handles)
            init_sections_dict[k] = client.models.generate_content(
                model=gemini_model_id,
                contents=contents,
            ).text

        max_idx = max(init_sections_dict.keys())
        init_sections = []
        for idx in range(max_idx + 1):
            section_info = {
                "order_idx": idx,
                "title": self.sections[idx].title,
                "content": init_sections_dict[idx],
            }
            init_sections.append(section_info)

        assert len(self.sections) == len(init_sections)

        return init_node_set, init_sections

    async def _update_sections(
        self,
        llm: ChatOpenAI,
        node_name: str,
        previous_state: list[dict[str, str]],
        tech_docs: TechDocsContent,
        annotations: list[Category] | None,
    ) -> dict[str, str]:
        assert len(previous_state) == len(self.sections)
        common_update_prompt = f"\n\nNow update this document, as appropriate, given the following detailed content from (`{node_name}`):\n\n"
        common_description_prompt = (
            f"DESCRIPTION of `{node_name}`:\n\n{tech_docs.long_description}\n\n"
        )
        prompt_pairs = []
        if tech_docs.source is not None and tech_docs.source.strip() == "":
            return previous_state

        # TODO: Just do this better here and elsewhere with similar processing:
        # TODO: wastefully creating prompts when annotations say some or irrelevant, and
        # TODO: awkward logic to keep same-sized list processing for every section even though
        # TODO: now some are explicitly not to be processed. This all started because it
        # TODO: wasn't clear what key to use for sections (non-unique titles), etc. Should
        # TODO: just make `SectionCommitted` hashable and good to go.
        for idx in range(len(previous_state)):
            assert self.sections[idx].title == previous_state[idx]["title"]
            user_prompt = f"Current state of the {self.sections[idx].title} section:\n\n{previous_state[idx]['content']}"
            user_prompt += common_update_prompt
            user_prompt += common_description_prompt
            if tech_docs.source is not None:
                system_prompt = self.sections[idx].update_from_file_system_prompt(
                    goal=self.document.goal,
                    preamble=self.scope.preamble,
                )
                # TODO: Actually figure out how to handle very large files.
                code_chunks = split_text(
                    text=tech_docs.source,
                    chunk_size=64_000,
                    chunk_overlap=0,
                )
                user_prompt += (
                    f"SOURCE CODE for `{node_name}`:\n\n{code_chunks[0].text}\n\n"
                )
                if llm.model == "o1-mini":
                    user_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                system_prompt = self.sections[idx].update_from_folder_system_prompt(
                    goal=self.document.goal, preamble=self.scope.preamble
                )

            prompt_pairs.append((system_prompt, user_prompt))

        async with asyncio.TaskGroup() as tg:
            section_tasks = []
            for idx, (system_prompt, user_prompt) in enumerate(prompt_pairs):
                if annotations is not None:
                    not_relevant = annotations[idx] == Category.Irrelevant
                else:
                    not_relevant = False
                if (
                    not_relevant
                    or self.sections[idx].section_creation_method
                    != SectionCreationMethod.SEQUENTIAL_EDIT
                ):
                    section_tasks.append(None)
                else:
                    section_tasks.append(
                        tg.create_task(
                            llm_generate(
                                llm=llm,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                            )
                        )
                    )

        new_state = []
        for idx, task in enumerate(section_tasks):
            if task is None:
                new_state.append(previous_state[idx])
            else:
                new_state.append(
                    {
                        "order_idx": previous_state[idx]["order_idx"],
                        "title": previous_state[idx]["title"],
                        "content": task.result(),
                    }
                )

        return new_state

    async def _update_sections_with_pdf(
        self,
        llm: ChatOpenAI,
        previous_state: list[dict[str, str]],
        pdf_path: str,
        pdf_pages: list[str],
        pdf_annotations: list[Category] | None,
    ) -> dict[str, str]:
        # NOTE: Currently doing the whole PDF in one function, can save state between pages if needed at a later point.

        assert len(previous_state) == len(self.sections)
        common_update_prompt = f"\n\nNow update this document, as appropriate, given the following content from a page of the pdf (`{pdf_path}`):\n\n"

        temp_results = [
            previous_state[idx]["content"] for idx in range(len(previous_state))
        ]
        for pg_idx, page_content in enumerate(pdf_pages):
            prompt_pairs = []
            for idx in range(len(previous_state)):
                user_prompt = f"Current state of the {self.sections[idx].title} section:\n\n{temp_results[idx]}"
                user_prompt += common_update_prompt
                user_prompt += f"PDF PAGE CONTENT for `{pdf_path}`:\n\n{page_content}"
                system_prompt = self.sections[idx].update_from_pdf_system_prompt(
                    goal=self.document.goal,
                    preamble=self.scope.preamble,
                )
                prompt_pairs.append((system_prompt, user_prompt))
            async with asyncio.TaskGroup() as tg:
                section_tasks = []
                for idx, (system_prompt, user_prompt) in enumerate(prompt_pairs):
                    if pdf_annotations is not None:
                        relevant = (
                            pdf_annotations[pg_idx][idx] == Category.HighlyRelevant
                        )  # Only doing highly relevant for pdf pages
                    else:
                        relevant = True
                    if (
                        not relevant
                        or self.sections[idx].section_creation_method
                        != SectionCreationMethod.SEQUENTIAL_EDIT
                    ):
                        section_tasks.append(None)
                    else:
                        section_tasks.append(
                            tg.create_task(
                                llm_generate(
                                    llm=llm,
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                )
                            )
                        )
            new_temp_results = []
            for idx, task in enumerate(section_tasks):
                if task is None:
                    new_temp_results.append(temp_results[idx])
                else:
                    new_temp_results.append(task.result())
            temp_results = new_temp_results

        new_state = []
        for idx, result in enumerate(temp_results):
            new_state.append(
                {
                    "order_idx": previous_state[idx]["order_idx"],
                    "title": previous_state[idx]["title"],
                    "content": result,
                }
            )
        return new_state

    async def _final_section_format(
        self, llm: ChatOpenAI, previous_state: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        assert len(previous_state) == len(self.sections)

        prompt_pairs = []
        for idx in range(len(previous_state)):
            assert self.sections[idx].title == previous_state[idx]["title"]
            user_prompt = f"Detailed {previous_state[idx]['title']} section content:\n\n{previous_state[idx]['content']}"
            system_prompt = self.sections[idx].final_output_format(
                goal=self.document.goal,
                preamble=self.scope.preamble,
            )
            prompt_pairs.append((system_prompt, user_prompt))

        async with asyncio.TaskGroup() as tg:
            section_tasks = []
            for system_prompt, user_prompt in prompt_pairs:
                section_tasks.append(
                    tg.create_task(
                        llm_generate(
                            llm=llm,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                    )
                )

        new_state = []
        for idx, task in enumerate(section_tasks):
            new_state.append(
                {
                    "order_idx": previous_state[idx]["order_idx"],
                    "title": previous_state[idx]["title"],
                    "content": task.result(),
                }
            )

        return new_state

    async def generate(
        self,
        execution_mode: ExecutionMode,
        checkpoint: AutoDocsCheckpoint,
        bucket: str,
        resume: bool = False,
        source_version_node_id: str | None = None,
        hatchet_id: str | None = None,
    ) -> str:
        from shared.v3.utils.post_processing.mermaid import (
            fix_mermaid_syntax_in_response,
        )

        llm_tagging = ChatOpenAI(
            model=self.llm.tag_model, temperature=0, request_timeout=300
        )
        llm_section_init = ChatOpenAI(
            model=self.llm.section_init_model, temperature=0, request_timeout=300
        )
        llm_section_update = ChatOpenAI(
            model=self.llm.section_update_model, temperature=0, request_timeout=300
        )
        llm_section_format = ChatOpenAI(
            model=self.llm.section_format_model, temperature=0, request_timeout=500
        )
        llm_assembly = ChatOpenAI(
            model=self.llm.assembly_model, temperature=0, request_timeout=900
        )
        llm_copy_editor = ChatOpenAI(
            model=self.llm.copy_editor_model, temperature=0, request_timeout=900
        )

        # Download PDFS if needed
        if execution_mode == ExecutionMode.MODAL:
            pdf_ids = [pdf_cfg.version_id for pdf_cfg in self.scope.pdfs]
            for pdf_id in pdf_ids:
                _download_pdf_from_s3(version_id=pdf_id)

        # Convert PDFs to markdown
        pdf_pages_dict = {}
        if len(self.scope.pdfs) > 0:
            print("Converting pdfs to markdown...")
            pdf_paths = _get_pdf_paths(
                [pdf_cfg.pdf_name for pdf_cfg in self.scope.pdfs],
                execution_mode=execution_mode,
            )
            for pdf_path in pdf_paths:
                md_text = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
                pdf_pages_dict[pdf_path] = md_text.split("-----")

        # Handle resume
        if resume:
            # Legacy local resume (for local dev only)
            state = self.load_state()
            revisions = state["revisions"]
            init_state = state["init_state"]
            annotations = init_state["annotations"]
            init_node_set = init_state["init_node_set"]
            appended_reverse_topo = init_state["appended_reverse_topo"]
            section_state = revisions[-1]
            pidx = section_state["_index"]
            preamble_content = (
                f"\nHere is further context about the document we are writing:\n\n{self.scope.preamble}"
                if self.scope.preamble.strip() != ""
                else ""
            )
            print(
                f"\n🧙 {CYAN}Whizdoodling{RESET}! to create a document with the following goal:\n{self.document.goal}\n{preamble_content}\n"
            )
            pdf_annotations = None
        elif checkpoint and checkpoint.current_phase != AutoDocsPhase.INITIALIZING:
            # S3 checkpoint resume - restore state and skip to appropriate phase
            logger.info(
                f"Resuming from S3 checkpoint: phase={checkpoint.current_phase}, "
                f"progress={checkpoint.phase_current}/{checkpoint.phase_total}"
            )

            # Restore annotations from checkpoint
            annotations = checkpoint.annotations
            pdf_annotations = checkpoint.pdf_annotations

            # Get driver_docs for topo reconstruction
            driver_docs = self.driver_docs

            # Reconstruct appended_reverse_topo from checkpoint paths
            # Build a combined content dict from all driver_docs
            combined_content: dict[str, TechDocsContent] = {}
            for dd in driver_docs:
                combined_content.update(dd.content)

            appended_reverse_topo = [
                (p, combined_content[p])
                for p in checkpoint.appended_reverse_topo_paths
                if p in combined_content
            ]

            # Build joined_graph for annotation lookups
            joined_graph = {k: v for dd in driver_docs for k, v in dd.dag.items()}

            # Generate source list if we have annotations
            if annotations:
                self._source_list = self._generate_sources_list(
                    annotations=annotations, pdf_annotations=pdf_annotations
                )

            # Determine what to restore/run based on checkpoint phase
            if checkpoint.current_phase == AutoDocsPhase.ANNOTATING:
                # Mid-annotation resume: continue annotation, then run remaining phases
                logger.info("Resuming mid-annotation phase")

                if execution_mode == ExecutionMode.MODAL:
                    await update_autodocs_status(
                        source_version_node_id=source_version_node_id,
                        status_kind=AutoDocStatusMessageKind.EVALUATING_SOURCES,
                        content="Resuming source evaluation...",
                        hatchet_id=hatchet_id,
                    )

                # Build topo for annotation (needed for _annotate_nodes)
                subgraphs = []
                for dd, code_cfg in zip(driver_docs, self.scope.code):
                    subgraph = build_subgraph(dag=dd.dag, start=code_cfg.node_path)
                    if subgraph:
                        subgraphs.append(subgraph)
                toposorts = [
                    list(TopologicalSorter(sg).static_order()) for sg in subgraphs
                ]
                topos = [
                    [(p, dd.content[p]) for p in ts]
                    for ts, dd in zip(toposorts, driver_docs)
                ]
                appended_topo = [tup for t in topos for tup in t]

                # Continue annotation from checkpoint
                annotations, pdf_annotations = await self._annotate_nodes(
                    llm=llm_tagging,
                    topo=appended_topo,
                    graph=joined_graph,
                    execution_mode=execution_mode,
                    pdf_pages_dict=pdf_pages_dict,
                    checkpoint=checkpoint,
                    bucket=bucket,
                    existing_annotations=checkpoint.annotations,
                    existing_pdf_annotations=checkpoint.pdf_annotations,
                    start_batch_idx=checkpoint.phase_current,
                )
                self._source_list = self._generate_sources_list(
                    annotations=annotations, pdf_annotations=pdf_annotations
                )

                # Now run section initialization (fresh, since we just completed annotation)
                if execution_mode == ExecutionMode.MODAL:
                    await update_autodocs_status(
                        source_version_node_id=source_version_node_id,
                        status_kind=AutoDocStatusMessageKind.GENERATING_SECTION_DRAFTS,
                        content="Generating initial section drafts...",
                        hatchet_id=hatchet_id,
                    )

                # Build reverse_topos for section init
                reverse_topos = [
                    [(p, dd.content[p]) for p in ts]
                    for ts, dd in zip(toposorts, driver_docs)
                ]
                _ = [rt.reverse() for rt in reverse_topos]

                init_node_set, sections_init = await self._initialize_sections(
                    llm=llm_section_init,
                    reverse_topos_from_start=reverse_topos,
                    driver_docs=driver_docs,
                    annotations=annotations,
                    pdf_annotations=pdf_annotations,
                    execution_mode=execution_mode,
                    pdf_pages_dict=pdf_pages_dict,
                    checkpoint=checkpoint,
                    bucket=bucket,
                    existing_scatter_state=None,
                )

                pidx = 1
                section_state = {"sections": sections_init, "_index": pidx}
                revisions = [section_state]

                # Checkpoint after section init
                checkpoint.update_phase(
                    AutoDocsPhase.SECTION_UPDATE,
                    current=pidx,
                    total=len(appended_reverse_topo),
                )
                checkpoint.update_annotations(annotations, pdf_annotations)
                checkpoint.update_sections(section_state["sections"], init_node_set)
                checkpoint.update_topo_index(pidx)
                await checkpoint.save(bucket)

            elif checkpoint.current_phase == AutoDocsPhase.SECTION_UPDATE:
                # Could be mid-scatter or mid-sequential-update
                if checkpoint.scatter_state:
                    # Mid-scatter resume: run _initialize_sections with scatter_state
                    logger.info("Resuming mid-scatter phase")

                    if execution_mode == ExecutionMode.MODAL:
                        await update_autodocs_status(
                            source_version_node_id=source_version_node_id,
                            status_kind=AutoDocStatusMessageKind.GENERATING_SECTION_DRAFTS,
                            content="Resuming section draft generation...",
                            hatchet_id=hatchet_id,
                        )

                    # Build reverse_topos for section init
                    subgraphs = []
                    for dd, code_cfg in zip(driver_docs, self.scope.code):
                        subgraph = build_subgraph(dag=dd.dag, start=code_cfg.node_path)
                        if subgraph:
                            subgraphs.append(subgraph)
                    toposorts = [
                        list(TopologicalSorter(sg).static_order()) for sg in subgraphs
                    ]
                    reverse_topos = [
                        [(p, dd.content[p]) for p in ts]
                        for ts, dd in zip(toposorts, driver_docs)
                    ]
                    _ = [rt.reverse() for rt in reverse_topos]

                    init_node_set, sections_init = await self._initialize_sections(
                        llm=llm_section_init,
                        reverse_topos_from_start=reverse_topos,
                        driver_docs=driver_docs,
                        annotations=annotations,
                        pdf_annotations=pdf_annotations,
                        execution_mode=execution_mode,
                        pdf_pages_dict=pdf_pages_dict,
                        checkpoint=checkpoint,
                        bucket=bucket,
                        existing_scatter_state=checkpoint.scatter_state,
                    )

                    pidx = 1
                    section_state = {"sections": sections_init, "_index": pidx}
                    revisions = [section_state]

                    # Checkpoint after section init
                    checkpoint.update_phase(
                        AutoDocsPhase.SECTION_UPDATE,
                        current=pidx,
                        total=len(appended_reverse_topo),
                    )
                    if annotations is not None:
                        checkpoint.update_annotations(annotations, pdf_annotations)
                    checkpoint.update_sections(section_state["sections"], init_node_set)
                    checkpoint.update_topo_index(pidx)
                    await checkpoint.save(bucket)

                else:
                    # Post-scatter, mid-sequential-update resume
                    logger.info(
                        f"Resuming mid-sequential-update: index={checkpoint.current_topo_index}"
                    )
                    init_node_set = (
                        set(checkpoint.init_node_set)
                        if checkpoint.init_node_set
                        else set()
                    )
                    section_state = {
                        "sections": checkpoint.sections_content,
                        "_index": checkpoint.current_topo_index,
                    }
                    pidx = checkpoint.current_topo_index
                    revisions = [section_state]

            elif checkpoint.current_phase == AutoDocsPhase.UPDATING_PDFS:
                # Resume mid-PDF-processing
                logger.info(
                    f"Resuming mid-PDF-update: index={checkpoint.current_pdf_index}"
                )
                init_node_set = (
                    set(checkpoint.init_node_set) if checkpoint.init_node_set else set()
                )
                section_state = {"sections": checkpoint.sections_content, "_index": 0}
                # Set pidx to skip sequential updates and resume PDF processing
                # The PDF loop uses: pdf_paths[pidx - total - 1 :]
                # To start at current_pdf_index, we need: pidx - total - 1 = current_pdf_index
                # So: pidx = current_pdf_index + total + 1
                total = len(appended_reverse_topo)
                pidx = checkpoint.current_pdf_index + total + 1
                revisions = [section_state]

            elif checkpoint.current_phase in (
                AutoDocsPhase.FORMATTING,
                AutoDocsPhase.BEFORE_ASSEMBLY,
            ):
                # Skip to formatting or assembly - restore sections and skip prior phases
                logger.info(f"Resuming from phase: {checkpoint.current_phase}")
                init_node_set = (
                    set(checkpoint.init_node_set) if checkpoint.init_node_set else set()
                )
                section_state = {"sections": checkpoint.sections_content, "_index": 0}
                revisions = [section_state]
                # Mark that we should skip sequential updates and PDF processing
                # We'll use a flag that the later code checks
                # For BEFORE_ASSEMBLY, also skip formatting
                # Set pidx very high to skip all sequential/PDF processing
                pidx = len(appended_reverse_topo) + len(self.scope.pdfs) + 100

            else:
                # ASSEMBLING or COMPLETE - shouldn't normally happen, but handle gracefully
                logger.warning(
                    f"Unexpected checkpoint phase for resume: {checkpoint.current_phase}, starting fresh"
                )
                # Fall through to fresh start by clearing checkpoint
                checkpoint = None

        # Fresh start - only runs if not resuming
        if not resume and not (
            checkpoint and checkpoint.current_phase != AutoDocsPhase.INITIALIZING
        ):
            # Create path traversal state.
            targets = [
                [code_cfg.version_id, _get_codebase_name(path=code_cfg.node_path)]
                for code_cfg in self.scope.code
            ]
            match execution_mode:
                case ExecutionMode.LOCAL:
                    driver_docs = [
                        DriverDocsContent.from_disk(
                            p=_get_path_on_disk(codebase_name=t)
                        )
                        for [_, t] in targets
                    ]
                case ExecutionMode.MODAL:
                    driver_docs = self.driver_docs
                case _:
                    raise ValueError("Invalid execution mode")

            subgraphs = []
            for dd, code_cfg in zip(driver_docs, self.scope.code):
                subgraph = build_subgraph(dag=dd.dag, start=code_cfg.node_path)
                if subgraph:
                    subgraphs.append(subgraph)
            if len(subgraphs) == 0:
                raise ValueError(
                    "Subgraphs could not be built for any supplied source nodes."
                )

            toposorts = [list(TopologicalSorter(sg).static_order()) for sg in subgraphs]
            topos = [
                [(p, dd.content[p]) for p in ts]
                for ts, dd in zip(toposorts, driver_docs)
            ]
            reverse_topos = [
                [(p, dd.content[p]) for p in ts]
                for ts, dd in zip(toposorts, driver_docs)
            ]
            _ = [rt.reverse() for rt in reverse_topos]
            appended_topo = [tup for t in topos for tup in t]
            appended_reverse_topo = [tup for rt in reverse_topos for tup in rt]
            joined_graph = {k: v for dd in driver_docs for k, v in dd.dag.items()}

            # Store paths on checkpoint for reconstruction on resume
            checkpoint.appended_reverse_topo_paths = [
                p for p, _ in appended_reverse_topo
            ]

            if execution_mode == ExecutionMode.MODAL:
                await update_autodocs_status(
                    source_version_node_id=source_version_node_id,
                    status_kind=AutoDocStatusMessageKind.EVALUATING_SOURCES,
                    content="Evaluating sources for relevance...",
                    hatchet_id=hatchet_id,
                )

            # Annotate nodes with tags, if applicable.
            if self.document.use_tagging:
                annotations, pdf_annotations = await self._annotate_nodes(
                    llm=llm_tagging,
                    topo=appended_topo,
                    graph=joined_graph,
                    execution_mode=execution_mode,
                    pdf_pages_dict=pdf_pages_dict,
                    checkpoint=checkpoint,
                    bucket=bucket,
                )
                self._source_list = self._generate_sources_list(
                    annotations=annotations, pdf_annotations=pdf_annotations
                )
                logger.info(
                    f"Annotation phase complete: {len(annotations)} nodes annotated"
                )
            else:
                annotations = None
                pdf_annotations = None

            if execution_mode == ExecutionMode.MODAL:
                await update_autodocs_status(
                    source_version_node_id=source_version_node_id,
                    status_kind=AutoDocStatusMessageKind.GENERATING_SECTION_DRAFTS,
                    content="Generating initial section drafts...",
                    hatchet_id=hatchet_id,
                )

            pidx = 1
            section_state = dict()
            revisions = []

            # Initial draft creation
            print(
                f"\n({BLUE}{self.llm.section_init_model}{RESET}) Building initial section drafts for target scope in `{self.scope.code}`..."
            )

            # Get existing scatter state from checkpoint for resume
            existing_scatter_state = checkpoint.scatter_state

            # Save annotations to checkpoint BEFORE scatter starts, so scatter
            # checkpoints preserve them (scatter saves don't call update_annotations)
            if annotations is not None:
                checkpoint.update_annotations(annotations, pdf_annotations)

            init_node_set, sections_init = await self._initialize_sections(
                llm=llm_section_init,
                reverse_topos_from_start=reverse_topos,
                driver_docs=driver_docs,
                annotations=annotations,
                pdf_annotations=pdf_annotations,
                execution_mode=execution_mode,
                pdf_pages_dict=pdf_pages_dict,
                checkpoint=checkpoint,
                bucket=bucket,
                existing_scatter_state=existing_scatter_state,
            )
            section_state["sections"] = sections_init
            section_state["_index"] = pidx
            revisions.append(section_state)

            init_state = {
                "annotations": annotations,
                "appended_reverse_topo": appended_reverse_topo,
                "init_node_set": init_node_set,
            }

            # Checkpoint after initialization - ready for section updates
            checkpoint.update_phase(
                AutoDocsPhase.SECTION_UPDATE,
                current=pidx,
                total=len(appended_reverse_topo),
            )
            if annotations is not None:
                checkpoint.update_annotations(annotations, pdf_annotations)
            checkpoint.update_sections(section_state["sections"], init_node_set)
            checkpoint.update_topo_index(pidx)
            await checkpoint.save(bucket)
            logger.info(
                f"Section initialization complete: {len(sections_init)} sections created"
            )

        # Exhaustive updates
        if any(
            s.section_creation_method == SectionCreationMethod.SEQUENTIAL_EDIT
            for s in self.sections
        ):
            total = len(appended_reverse_topo)
            logger.info(f"Starting sequential updates: {total} nodes to process")
            print(
                f"\n({BLUE}{self.llm.section_update_model}{RESET}) Iteratively improving the initial state of the documents..."
            )
            for p, tech_docs in appended_reverse_topo[pidx - 1 :]:
                if p in init_node_set and tech_docs.source is None:
                    print(
                        f"[{pidx} / {total}] Already assessed folder `{GREEN}{p}{RESET}` in building initial sections..."
                    )
                    pidx += 1
                    continue
                print(
                    f"[{pidx} / {total}] Updating sections with content from `{GREEN}{p}{RESET}`..."
                )
                new_section_state = dict()
                section_update = await self._update_sections(
                    llm=llm_section_update,
                    node_name=p,
                    previous_state=section_state["sections"],
                    tech_docs=tech_docs,
                    annotations=annotations[p] if annotations is not None else None,
                )
                pidx += 1

                new_section_state["sections"] = section_update
                new_section_state["_index"] = pidx
                revisions.append(new_section_state)
                section_state = new_section_state

                # Checkpoint after each sequential update (mid-phase)
                checkpoint.update_phase(
                    AutoDocsPhase.SECTION_UPDATE, current=pidx, total=total
                )
                checkpoint.update_sections(section_state["sections"], init_node_set)
                checkpoint.update_topo_index(pidx)
                await checkpoint.save(bucket)

            logger.info(f"Sequential updates complete: processed {pidx - 1} nodes")

            if len(self.scope.pdfs) > 0:
                pdf_paths = _get_pdf_paths(
                    [pdf_cfg.pdf_name for pdf_cfg in self.scope.pdfs],
                    execution_mode=execution_mode,
                )
                logger.info(f"Starting PDF updates: {len(pdf_paths)} PDFs to process")
                for pdf_path in pdf_paths[pidx - total - 1 :]:
                    print(f"Updating sections with content from {pdf_path}...")
                    new_section_state = dict()
                    section_update = await self._update_sections_with_pdf(
                        llm=llm_section_update,
                        previous_state=section_state["sections"],
                        pdf_path=pdf_path,
                        pdf_pages=pdf_pages_dict[pdf_path],
                        pdf_annotations=pdf_annotations[pdf_path]
                        if pdf_annotations
                        else None,
                    )
                    pidx += 1
                    new_section_state["sections"] = section_update
                    new_section_state["_index"] = pidx
                    revisions.append(new_section_state)
                    section_state = new_section_state

                    # Checkpoint after each PDF update (UPDATING_PDFS phase)
                    checkpoint.update_phase(
                        AutoDocsPhase.UPDATING_PDFS,
                        current=pidx - total,
                        total=len(pdf_paths),
                    )
                    checkpoint.update_sections(section_state["sections"], init_node_set)
                    checkpoint.update_topo_index(total)
                    checkpoint.update_pdf_index(pidx - total)
                    await checkpoint.save(bucket)
                logger.info(f"PDF updates complete: processed {len(pdf_paths)} PDFs")

        # Skip formatting if resuming from BEFORE_ASSEMBLY (formatting already done)
        skip_formatting = (
            checkpoint is not None
            and checkpoint.current_phase == AutoDocsPhase.BEFORE_ASSEMBLY
        )

        if not skip_formatting:
            logger.info("Starting formatting phase")
            if execution_mode == ExecutionMode.MODAL:
                await update_autodocs_status(
                    source_version_node_id=source_version_node_id,
                    status_kind=AutoDocStatusMessageKind.OPTIMIZING_SECTION_STRUCTURE,
                    content="Optimizing content structure for each section...",
                    hatchet_id=hatchet_id,
                )
            # Final output format enforcement
            print(
                f"\n({BLUE}{self.llm.section_format_model}{RESET}) Final section output structure pass..."
            )
            new_section_state = dict()
            final_section_update = await self._final_section_format(
                llm=llm_section_format, previous_state=section_state["sections"]
            )
            pidx += 1
            new_section_state["sections"] = final_section_update
            new_section_state["_index"] = pidx
            revisions.append(new_section_state)
            section_state = new_section_state

            # Checkpoint before assembly - formatting complete
            checkpoint.update_phase(AutoDocsPhase.BEFORE_ASSEMBLY)
            checkpoint.update_sections(section_state["sections"])
            await checkpoint.save(bucket)
            logger.info("Formatting phase complete")
        else:
            logger.info("Skipping formatting phase (resuming from BEFORE_ASSEMBLY)")

        if execution_mode == ExecutionMode.MODAL:
            await update_autodocs_status(
                source_version_node_id=source_version_node_id,
                status_kind=AutoDocStatusMessageKind.ASSEMBLING_FINAL_DOCUMENT,
                content="Assembling all sections into a single document...",
                hatchet_id=hatchet_id,
            )

        # Final document assembly
        logger.info("Starting assembly phase")
        final_doc_revisions = []
        print(
            f"\n({BLUE}{self.llm.assembly_model}{RESET}) Full document assembly pass..."
        )
        assembly_user_prompt = "Document draft content:"
        for section in section_state["sections"]:
            assembly_user_prompt += f"\n\nDRAFT content for section {section['title']}:\n\n{section['content']}"
        full_document = await llm_assembly.generate_response(
            system_prompt=self.assembly_system_prompt(),
            user_prompt=assembly_user_prompt,
        )
        final_doc_revisions.append(full_document)

        if execution_mode == ExecutionMode.MODAL:
            await update_autodocs_status(
                source_version_node_id=source_version_node_id,
                status_kind=AutoDocStatusMessageKind.COPY_EDITING,
                content="Copy editing and finalizing document...",
                hatchet_id=hatchet_id,
            )
        # Final copy editor pass
        print(
            f"\n({BLUE}{self.llm.copy_editor_model}{RESET}) Final copy editor pass..."
        )
        copy_editor_user_prompt = f"Document draft:\n\n{full_document}"
        final_document = await llm_copy_editor.generate_response(
            system_prompt=self.final_copy_editor_system_prompt(),
            user_prompt=copy_editor_user_prompt,
        )

        final_document += "\n\nMade with ❤️ by [Driver](https://www.driver.ai/)"

        final_doc_revisions.append(final_document)
        final_doc_revisions.append(fix_mermaid_syntax_in_response(text=final_document))

        logger.info("Document generation complete")
        return final_doc_revisions[-1]


async def main(args: argparse.Namespace) -> None:
    if args.validate:
        validated_cfg = AutoDocCfg.from_file(toml_file=args.validate)
        if not args.quiet:
            print(validated_cfg.model_dump_json(indent=2))
        print(f"\n\nFile `{args.validate}` is valid and ready for processing!")
    elif args.execute:
        init_state = await AutoDocInitState.from_cfg(
            cfg=AutoDocCfg.from_file(toml_file=args.execute),
            execution_mode=ExecutionMode.LOCAL,
        )
        doc = await init_state.generate(execution_mode=ExecutionMode.LOCAL)
        if not args.quiet:
            console = Console()
            console.print(Markdown(doc))
    elif args.resume:
        cfg = AutoDocCfg.from_file(toml_file=args.resume)
        state_filename = build_state_filename(
            scope_roots=cfg.scope.code, fmt=cfg.document.fmt
        )
        state = AutoDocInitState.from_disk(json_p=state_filename)
        doc = await state.generate(resume=True, execution_mode=ExecutionMode.LOCAL)
        if not args.quiet:
            console = Console()
            console.print(Markdown(doc))
    # elif args.remote:
    #     import modal

    #     run_autodoc_cli = modal.Function.from_name(
    #         app_name="autodocs", name="run_autodoc_cli", environment_name=args.env
    #     )
    #     with open(args.config) as f:
    #         toml_content = f.read()
    #     doc = run_autodoc_cli.remote(
    #         toml_content=toml_content, page_node_id=UUID(args.page_id)
    #     )
    #     with open(args.output, "w") as f:
    #         f.write(doc)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument(
        "-v",
        "--validate",
        help="validate the given configuration file",
        metavar="TOML FILE",
        type=str,
    )
    mutex_group.add_argument(
        "-e",
        "--execute",
        help="execute content generation with the given configuration file",
        metavar="TOML FILE",
        type=str,
    )
    mutex_group.add_argument(
        "-r",
        "--resume",
        help="resume execution",
        metavar="TOML FILE",
        type=str,
    )
    mutex_group.add_argument(
        "--remote",
        help="execute remotely with Modal",
        action="store_true",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        help="do not render final output",
        action="store_true",
    )

    parser.add_argument(
        "--env",
        help="execution environment (required for --remote)",
        choices=["dev", "prod"],
        type=str,
    )
    parser.add_argument(
        "--config",
        help="path to configuration file (required for --remote)",
        type=str,
    )
    parser.add_argument(
        "--page-id",
        help="page ID (required for --remote)",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="output document path (required for --remote)",
        type=str,
    )

    args = parser.parse_args()

    asyncio.run(main(args))

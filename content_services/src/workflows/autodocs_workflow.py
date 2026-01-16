from datetime import timedelta

from autodocs.src.main import run_autodoc
from hatchet_client import hatchet
from hatchet_sdk import Context
from hatchet_sdk.runnables.types import ConcurrencyExpression, ConcurrencyLimitStrategy
from shared.interfaces.hatchet_interfaces import AutodocInput


@hatchet.task(
    name="autodocs-workflow",
    execution_timeout=timedelta(minutes=480),
    concurrency=ConcurrencyExpression(
        max_runs=5,
        expression="'autodocs-workflow'",  # NOTE: must be a string literal to be evaluated as a constant task name
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
async def autodocs_task(input: AutodocInput, ctx: Context) -> dict[str, str]:
    print("starting autodocs task")
    await run_autodoc(
        version_node_id=input.version_node_id,
        config_kind=input.config_kind,
        document_goal=input.document_goal,
        user_context=input.user_context,
        content_kind=input.content_kind,
        hatchet_id=ctx.workflow_run_id,
        organization_id=input.organization_id,
    )
    print("executed autodocs task")
    return {"result": "autodoc completed"}

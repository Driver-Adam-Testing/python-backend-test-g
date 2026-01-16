from database.models_enums import AutoDocConfigKind, ContentKind
from pydantic import BaseModel


class InspectorInput(BaseModel):
    version_id: str


class Auth0SyncInput(BaseModel):
    dry_run: bool = False
    verbose: bool = False
    initial_run: bool = False


class ProcessAuth0EventInput(BaseModel):
    event: dict


class AutodocInput(BaseModel):
    version_node_id: str
    config_kind: AutoDocConfigKind
    document_goal: str | None
    user_context: str | None
    content_kind: ContentKind | None
    organization_id: str | None = None  # For checkpoint bucket computation


class HandleGithubEventsInput(BaseModel):
    installation_id: str | None
    org_id: str
    repos_added: list[dict]
    repos_deleted: list[dict]
    repos_pushed: list[dict]


class HandleGitlabEventsInput(BaseModel):
    installation_id: str | None
    org_id: str
    repos_added: list[dict]
    repos_deleted: list[dict]
    repos_pushed: list[dict]


class HandleBitbucketEventsInput(BaseModel):
    installation_id: str | None
    org_id: str
    repos_added: list[dict]
    repos_deleted: list[dict]
    repos_pushed: list[dict]


class HandleAzureDevopsEventsInput(BaseModel):
    installation_id: str | None
    org_id: str
    repos_added: list[dict]
    repos_deleted: list[dict]
    repos_pushed: list[dict]


class HandleBitbucketDCEventsInput(BaseModel):
    installation_id: str | None
    org_id: str
    repos_added: list[dict]
    repos_deleted: list[dict]
    repos_pushed: list[dict]


class ConnectReposForInstallationInput(BaseModel):
    github_installation_id: str


class RunCodebaseConnectionInput(BaseModel):
    presigned_url: str
    provisional_codebase_name: str
    org_id: str
    version_id: str
    provider: str = "manual"


class PDFProcessingInput(BaseModel):
    presigned_url: str
    version_id: str
    asset_name: str
    org_id: str

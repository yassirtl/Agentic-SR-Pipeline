"""Run configuration: defaults in code < config.yaml < .env < environment variables.

Secrets never go in the YAML. They live in the git-ignored ``.env`` file at the
project root (``OPENAI_API_KEY=...``), which ``load_config`` loads into the
environment; a variable already set in the shell wins over ``.env``. ChatOpenAI
then reads ``OPENAI_API_KEY`` from the environment itself.

Paths (LLM cache, HTTP cache, runs, protocol template) are resolved to absolute
paths under the project root, so a run started from any working directory finds
the same caches.

Environment overrides:
  SR_PROJECT_ROOT, SR_CACHE_DIR, SR_HTTP_CACHE_DIR, SR_RUNS_DIR,
  SR_PROTOCOL_TEMPLATE, SR_DEFAULT_MODEL, SR_MAX_RECORDS, SR_MAX_STEPS,
  SR_CONTACT_EMAIL, SR_MIN_HITS, SR_MAX_REFINEMENTS, SR_EMBEDDING_MODEL,
  SR_EMBEDDING_CACHE, SR_SCREENING_WORKERS, SR_SCREEN_LIMIT,
  SR_MODEL_<AGENT> (e.g. SR_MODEL_EXTRACTION=gpt-4o).
Optional database keys (read from the environment, never stored in the config):
  NCBI_API_KEY (PubMed: 10 requests/s instead of 3), OPENALEX_API_KEY.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

CONFIG_FILE = "config.yaml"


def find_project_root() -> Path:
    """Folder that holds pyproject.toml.

    ``SR_PROJECT_ROOT`` wins. Otherwise the package source tree is used: with an
    editable install (``pip install -e .``) this file lives in
    <root>/src/sr_pipeline/. If that fails (non-editable install), fall back to
    the current working directory.
    """
    env = os.environ.get("SR_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd().resolve()


class Config(BaseModel):
    default_model: str = "gpt-4o-mini"
    models: dict[str, str] = Field(default_factory=dict,
                                   description="Per-agent model, e.g. {'extraction': 'gpt-4o'}.")
    max_records: int = Field(
        3000, ge=1, le=10_000,
        description="Download ceiling per database, and upper end of the search's 'ok' hit "
                    "range: hits above it are reported as not retrieved. 10,000 = OpenAlex "
                    "page-paging and PubMed retmax limit.")
    screen_limit: int | None = Field(
        100, ge=0,
        description="Part 9b: title/abstract screening reads only the N best-ranked unique "
                    "records (relevance order of each database); the rest are reported in "
                    "PRISMA as 'not screened (limit N)'. 0 or null = screen every record. "
                    "Separate from max_records (the search's download ceiling).")
    max_steps: int = Field(40, ge=1, description="Coordinator step budget.")
    min_hits: int = Field(20, ge=0, description="Search too narrow below this many hits.")
    max_refinements: int = Field(2, ge=0, description="Search refinements after the first plan.")
    contact_email: str | None = Field(
        None, description="Sent to OpenAlex (mailto) and NCBI (email) as a polite contact.")

    # RAG (Part 5): chunks of ~chunk_words words, overlapping by chunk_overlap;
    # the LLM re-ranker reads the rerank_top best fused passages.
    embedding_model: str = "text-embedding-3-small"
    chunk_words: int = Field(250, ge=50)
    chunk_overlap: int = Field(40, ge=0)
    rerank_top: int = Field(20, ge=1)

    # Screening (Part 6): records screened in parallel threads; at full text, the
    # hybrid index gives ft_candidates passages per (paper, criterion), the LLM
    # re-ranks them and the screeners read the best ft_passages.
    screening_workers: int = Field(8, ge=1)
    ft_candidates: int = Field(10, ge=1)
    ft_passages: int = Field(5, ge=1)
    ft_rerank: bool = Field(True, description="LLM re-ranking of each criterion's passages "
                            "(most of the full-text cost; off = fused hybrid order).")

    project_root: Path = Field(default_factory=find_project_root)
    cache_dir: Path = Path(".cache/llm")
    http_cache_dir: Path = Path(".cache/http")
    embedding_cache: Path = Path(".cache/embeddings.sqlite")
    runs_dir: Path = Path("runs")
    protocol_template: Path = Path("protocols/template.yaml")

    def model_post_init(self, _ctx) -> None:
        root = self.project_root.expanduser().resolve()
        self.project_root = root
        for name in ("cache_dir", "http_cache_dir", "embedding_cache", "runs_dir",
                     "protocol_template"):
            p = getattr(self, name).expanduser()
            setattr(self, name, p if p.is_absolute() else root / p)

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default_model)

    def make_llm(self, **kwargs):
        """Real LLM wired to this config (per-agent models, absolute cache dir)."""
        from .llm import OpenAILLM

        return OpenAILLM(models=self.models, default=self.default_model,
                         cache_dir=self.cache_dir, **kwargs)

    def make_http(self, **kwargs):
        """HTTP client for the database connectors (disk cache under http_cache_dir)."""
        from .sources.http import HttpClient

        return HttpClient(cache_dir=self.http_cache_dir, **kwargs)

    def make_embedder(self, **kwargs):
        """OpenAI embeddings with the on-disk cache (re-embedding a chunk is free)."""
        from .rag.embed import OpenAIEmbedder

        return OpenAIEmbedder(self.embedding_model, cache_path=self.embedding_cache, **kwargs)


_ENV_FIELDS = {
    "SR_PROJECT_ROOT": "project_root",
    "SR_CACHE_DIR": "cache_dir",
    "SR_HTTP_CACHE_DIR": "http_cache_dir",
    "SR_RUNS_DIR": "runs_dir",
    "SR_PROTOCOL_TEMPLATE": "protocol_template",
    "SR_DEFAULT_MODEL": "default_model",
    "SR_MAX_RECORDS": "max_records",
    "SR_MAX_STEPS": "max_steps",
    "SR_CONTACT_EMAIL": "contact_email",
    "SR_MIN_HITS": "min_hits",
    "SR_MAX_REFINEMENTS": "max_refinements",
    "SR_EMBEDDING_MODEL": "embedding_model",
    "SR_EMBEDDING_CACHE": "embedding_cache",
    "SR_SCREENING_WORKERS": "screening_workers",
    "SR_SCREEN_LIMIT": "screen_limit",
}


def _env_overrides(env: dict[str, str]) -> dict:
    out: dict = {field: env[var] for var, field in _ENV_FIELDS.items() if env.get(var)}
    models = {k[len("SR_MODEL_"):].lower(): v for k, v in env.items()
              if k.startswith("SR_MODEL_") and v}
    if models:
        out["models"] = models
    return out


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Build the config. ``path`` defaults to <project root>/config.yaml (optional file).

    With ``env=None`` the real environment is used, after loading
    <project root>/.env into it (shell variables are not overwritten). Tests pass
    ``env`` explicitly, so they never read the real ``.env``.
    """
    if env is None:
        load_dotenv(find_project_root() / ".env", override=False)
        env = dict(os.environ)
    data: dict = {}

    if path is None:
        root = Path(env["SR_PROJECT_ROOT"]) if env.get("SR_PROJECT_ROOT") else find_project_root()
        path = root / CONFIG_FILE
        if not path.exists():
            path = None
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}

    overrides = _env_overrides(env)
    if "models" in overrides:  # merge per-agent models instead of replacing the dict
        overrides["models"] = {**data.get("models", {}), **overrides["models"]}
    return Config(**{**data, **overrides})

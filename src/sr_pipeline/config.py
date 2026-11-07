"""Run configuration: defaults in code < config.yaml < environment variables.

Secrets never go in the YAML: ChatOpenAI reads ``OPENAI_API_KEY`` from the
environment itself (Colab Secrets / getpass in the notebook).

Paths (LLM cache, runs, protocol template) are resolved to absolute paths under
the project root, so a run started from any working directory (a notebook in a
sub-folder, Colab's /content) finds the same cache.

Environment overrides:
  SR_PROJECT_ROOT, SR_CACHE_DIR, SR_RUNS_DIR, SR_PROTOCOL_TEMPLATE,
  SR_DEFAULT_MODEL, SR_MAX_RECORDS, SR_MAX_STEPS,
  SR_MODEL_<AGENT> (e.g. SR_MODEL_EXTRACTION=gpt-4o).
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_FILE = "config.yaml"


def find_project_root() -> Path:
    """Folder that holds pyproject.toml.

    ``SR_PROJECT_ROOT`` wins. Otherwise the package source tree is used: with an
    editable install (``pip install -e .``, what we do locally and in Colab) this
    file lives in <root>/src/sr_pipeline/. If that fails (non-editable install),
    fall back to the current working directory.
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
    max_records: int = Field(500, ge=1, description="Cap on records fetched per database.")
    max_steps: int = Field(40, ge=1, description="Coordinator step budget.")

    project_root: Path = Field(default_factory=find_project_root)
    cache_dir: Path = Path(".cache/llm")
    runs_dir: Path = Path("runs")
    protocol_template: Path = Path("protocols/template.yaml")

    def model_post_init(self, _ctx) -> None:
        root = self.project_root.expanduser().resolve()
        self.project_root = root
        for name in ("cache_dir", "runs_dir", "protocol_template"):
            p = getattr(self, name).expanduser()
            setattr(self, name, p if p.is_absolute() else root / p)

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default_model)

    def make_llm(self, **kwargs):
        """Real LLM wired to this config (per-agent models, absolute cache dir)."""
        from .llm import OpenAILLM

        return OpenAILLM(models=self.models, default=self.default_model,
                         cache_dir=self.cache_dir, **kwargs)


_ENV_FIELDS = {
    "SR_PROJECT_ROOT": "project_root",
    "SR_CACHE_DIR": "cache_dir",
    "SR_RUNS_DIR": "runs_dir",
    "SR_PROTOCOL_TEMPLATE": "protocol_template",
    "SR_DEFAULT_MODEL": "default_model",
    "SR_MAX_RECORDS": "max_records",
    "SR_MAX_STEPS": "max_steps",
}


def _env_overrides(env: dict[str, str]) -> dict:
    out: dict = {field: env[var] for var, field in _ENV_FIELDS.items() if env.get(var)}
    models = {k[len("SR_MODEL_"):].lower(): v for k, v in env.items()
              if k.startswith("SR_MODEL_") and v}
    if models:
        out["models"] = models
    return out


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Build the config. ``path`` defaults to <project root>/config.yaml (optional file)."""
    env = dict(os.environ) if env is None else env
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

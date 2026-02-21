"""Provider registry and OpenAI-compatible client factory."""
import os
from pathlib import Path

import openai
import yaml

PROVIDERS_DIR = Path(__file__).parent.parent / "configs" / "providers"


def list_providers() -> list[str]:
    """List available provider names from configs/providers/."""
    return sorted(p.stem for p in PROVIDERS_DIR.glob("*.yaml"))


def load_provider_config(provider_name: str) -> dict:
    """Load a provider YAML config by name."""
    config_path = PROVIDERS_DIR / f"{provider_name}.yaml"
    if not config_path.exists():
        available = ", ".join(list_providers())
        raise ValueError(
            f"Unknown provider: '{provider_name}'. Available: {available}"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_client(
    provider_name: str,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[openai.OpenAI, str, dict]:
    """
    Build an OpenAI-compatible client for any provider.

    Returns:
        (client, model_name, extra_params)
    """
    config = load_provider_config(provider_name)

    # Resolve API key: explicit arg > env var > default value
    if api_key is None:
        env_var = config.get("api_key_env")
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(
                    f"Provider '{provider_name}' requires {env_var} to be set. "
                    f"Export it or pass --api-key."
                )
        else:
            api_key = config.get("api_key_default", "no-key")

    model_name = model or config["default_model"]
    extra_params = config.get("extra_params", {})
    rate_limit_delay = config.get("rate_limit_delay", 0)

    client = openai.OpenAI(base_url=config["base_url"], api_key=api_key)
    return client, model_name, {**extra_params, "_rate_limit_delay": rate_limit_delay}

"""CLI entry point for the Synthetic Dataset Generation Suite."""
import json
from pathlib import Path

import click
import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
TASKS_DIR = CONFIGS_DIR / "tasks"


def _load_task_config(task_name: str) -> dict:
    """Load a task YAML config by name."""
    config_path = TASKS_DIR / f"{task_name}.yaml"
    if not config_path.exists():
        available = ", ".join(sorted(p.stem for p in TASKS_DIR.glob("*.yaml")))
        raise click.ClickException(
            f"Unknown task: '{task_name}'. Available: {available}"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.version_option(package_name="synthetic-dataset-generation-suite")
def cli():
    """Synthetic Dataset Generation Suite — multi-provider reasoning dataset pipeline."""


@cli.command()
@click.option("--task", required=True, help="Task config name (e.g. quantum_reasoning)")
@click.option("--output", "-o", required=True, help="Output JSON file path")
@click.option("--sample", "-n", type=int, default=None, help="Only extract first N examples")
def extract(task, output, sample):
    """Extract Q&A data from the configured source."""
    from .extract import extract_data

    task_config = _load_task_config(task)
    count = extract_data(task_config, output, sample_size=sample)
    click.echo(f"Done. {count} examples written to {output}")


@cli.command()
@click.option("--task", required=True, help="Task config name")
@click.option("--provider", required=True, help="Provider name (e.g. ollama, openai, anthropic)")
@click.option("--model", default=None, help="Override default model for the provider")
@click.option("--api-key", default=None, help="API key (overrides env var)")
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file (from extract)")
@click.option("--output", "-o", default=None, help="Output JSONL file (default: data/<task>_output.jsonl)")
@click.option("--test", type=int, default=None, help="Test mode: generate N samples with detailed validation")
@click.option("--no-resume", is_flag=True, help="Start fresh, don't resume from existing output")
def generate(task, provider, model, api_key, input_file, output, test, no_resume):
    """Generate reasoning dataset using any LLM provider."""
    from .generate import run_generation, run_test
    from .providers import get_client

    task_config = _load_task_config(task)
    client, model_name, extra_params = get_client(provider, model=model, api_key=api_key)

    click.echo(f"Provider: {provider} | Model: {model_name}")

    with open(input_file) as f:
        input_data = json.load(f)

    if test is not None:
        run_test(client, model_name, extra_params, task_config, input_data, num_samples=test)
    else:
        if output is None:
            output = f"data/{task}_output.jsonl"
        run_generation(
            client, model_name, extra_params, task_config,
            input_data, output, resume=not no_resume,
        )


@cli.command("filter")
@click.argument("input_file")
@click.option("--output", "-o", default=None, help="Output JSONL file")
@click.option("--lenient", is_flag=True, help="Only reject critical failures (missing answer tags)")
@click.option("--no-heal", is_flag=True, help="Disable healing of broken samples")
@click.option("--task", default=None, help="Task config name (for domain-specific validation rules)")
def filter_cmd(input_file, output, lenient, no_heal, task):
    """Filter and validate a generated JSONL dataset."""
    from .filter import filter_dataset

    validation_rules = None
    if task:
        task_config = _load_task_config(task)
        validation_rules = task_config.get("validation", {})

    filter_dataset(
        input_file,
        output_file=output,
        strict=not lenient,
        heal=not no_heal,
        validation_rules=validation_rules,
    )


@cli.command()
@click.argument("dataset")
@click.option("--samples", "-n", type=int, default=5, help="Number of samples to show")
@click.option("--random", "-r", "use_random", is_flag=True, help="Random sampling")
@click.option("--stats", "-s", "stats_only", is_flag=True, help="Show statistics only")
@click.option("--offset", type=int, default=0, help="Start from this sample index")
@click.option("--task", default=None, help="Task config name (for topic keywords)")
def qa(dataset, samples, use_random, stats_only, offset, task):
    """Inspect and analyze a reasoning dataset."""
    from .qa import run_qa

    topics = None
    if task:
        task_config = _load_task_config(task)
        topics = task_config.get("validation", {}).get("topics", [])

    run_qa(
        dataset,
        num_samples=samples,
        use_random=use_random,
        stats_only=stats_only,
        offset=offset,
        topics=topics,
    )


@cli.command()
def providers():
    """List available LLM providers."""
    from .providers import list_providers, load_provider_config

    for name in list_providers():
        config = load_provider_config(name)
        key_info = config.get("api_key_env", "none (local)")
        click.echo(f"  {name:15s}  model={config['default_model']:30s}  key={key_info}")


@cli.command()
def tasks():
    """List available task configs."""
    for p in sorted(TASKS_DIR.glob("*.yaml")):
        with open(p) as f:
            config = yaml.safe_load(f)
        click.echo(f"  {p.stem:25s}  {config.get('name', '')}")

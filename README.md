# Synthetic Dataset Generation Suite (SDGS)

Multi-provider pipeline for generating synthetic reasoning datasets for LLM fine-tuning. Plug in any OpenAI-compatible LLM provider — local or cloud — and generate structured reasoning datasets from any domain.

## Supported Providers

| Provider | Model Examples | Key Required |
|----------|---------------|--------------|
| **Ollama** (local) | `qwen3:32b`, `gpt-oss:120b`, `llama3` | No |
| **OpenAI** | `gpt-4o`, `o1-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini-2.0-flash`, `gemini-2.5-pro` | `GEMINI_API_KEY` |
| **Perplexity** | `sonar-pro`, `sonar-deep-research` | `PERPLEXITY_API_KEY` |

## Installation

```bash
cd Synthetic-Dataset-Generation-Suite
pip install -e .
```

## Quick Start

```bash
# 1. Set your API key (skip for Ollama)
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Extract source data
sdgs extract --task quantum_reasoning --output data/quantum.json

# 3. Test with a few samples first
sdgs generate --task quantum_reasoning --provider anthropic -i data/quantum.json --test 3

# 4. Run full generation
sdgs generate --task quantum_reasoning --provider anthropic -i data/quantum.json -o data/output.jsonl

# 5. Filter and validate
sdgs filter data/output.jsonl --task quantum_reasoning

# 6. Inspect results
sdgs qa data/output_filtered.jsonl --task quantum_reasoning --stats
```

## Pipeline

```
Extract → Test → Generate → Filter → QA
```

| Stage | Command | Purpose |
|-------|---------|---------|
| **Extract** | `sdgs extract` | Pull Q&A data from HuggingFace or local files |
| **Test** | `sdgs generate --test N` | Validate N samples before committing to full run |
| **Generate** | `sdgs generate` | Generate reasoning datasets with any provider |
| **Filter** | `sdgs filter` | Heal broken outputs + validate quality |
| **QA** | `sdgs qa` | Inspect samples and view statistics |

## CLI Reference

```bash
# List available providers and tasks
sdgs providers
sdgs tasks

# Extract data
sdgs extract --task <name> --output <path> [--sample N]

# Generate (full or test mode)
sdgs generate --task <name> --provider <name> -i <input.json> [-o <output.jsonl>] [--model <model>] [--test N] [--no-resume]

# Filter
sdgs filter <input.jsonl> [-o <output.jsonl>] [--lenient] [--no-heal] [--task <name>]

# QA inspection
sdgs qa <dataset.jsonl> [-n N] [-r] [-s] [--offset N] [--task <name>]
```

## Adding a New Provider

Create a YAML file in `configs/providers/`:

```yaml
name: my_provider
base_url: "https://api.myprovider.com/v1/"
api_key_env: "MY_PROVIDER_API_KEY"
default_model: "my-model-name"
rate_limit_delay: 0.5
extra_params: {}
```

Any provider that exposes an OpenAI-compatible chat completions endpoint works out of the box.

## Adding a New Domain

Create a YAML file in `configs/tasks/`:

```yaml
name: "My Domain Task"
domain: "my_domain"

source:
  type: local_json          # or huggingface, local_jsonl
  path: "./data/my_data.json"
  fields:
    question: "question"
    answer: "answer"
    metadata: []

generation:
  temperature: 0.3
  max_retries: 3
  system_prompt: |
    Your custom system prompt here.
    Use <think> and <answer> tags.
  user_prompt_template: |
    Question: {question}
    Reference: {answer}
    Solve step by step.

validation:
  require_tags:
    - ["<think>", "</think>"]
    - ["<answer>", "</answer>"]
  require_latex: false
  min_reasoning_length: 50
  min_answer_length: 10
  check_latex_consistency: false
  topics: []

output:
  format: jsonl
```

See `configs/tasks/example_task.yaml` for a complete template.

## Project Structure

```
sdgs/
  providers.py     # Provider registry + client factory
  generate.py      # Core generation logic
  extract.py       # Data extraction (HF, JSON, JSONL)
  filter.py        # Post-processing filter + healer
  qa.py            # Dataset inspection + statistics
  validate.py      # Shared validation + healing utilities
  cli.py           # Click CLI entry point

configs/
  providers/       # One YAML per LLM provider
  tasks/           # One YAML per domain/task
```

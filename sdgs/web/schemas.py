"""Pydantic request/response models for the SDGS web API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# --- Auth schemas ---

class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# --- Dataset schemas ---

class CreateDatasetRequest(BaseModel):
    topic: str
    provider: str | None = None
    model: str | None = None
    target_size: int = 100
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class CreateFromPapersRequest(BaseModel):
    paper_ids: list[int]
    provider: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class ImportHFRequest(BaseModel):
    repo_id: str  # e.g. "tatsu-lab/alpaca"
    split: str | None = None  # e.g. "train"


class BatchCreateRequest(BaseModel):
    datasets: list[CreateDatasetRequest]


class DatasetResponse(BaseModel):
    id: int
    name: str
    topic: str
    status: str
    provider: str | None = None
    model: str | None = None
    target_size: int = 100
    actual_size: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    healed_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    gpu_kwh: float = 0.0
    duration_seconds: float = 0.0
    output_path: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    hf_repo: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int
    page: int
    per_page: int


class QAPairResponse(BaseModel):
    id: int | None = None
    instruction: str
    output: str
    is_valid: bool = True
    was_healed: bool = False
    source_title: str | None = None
    think_text: str | None = None
    answer_text: str | None = None


class DatasetSamplesResponse(BaseModel):
    samples: list[QAPairResponse]
    total: int
    page: int
    per_page: int


# --- Paper schemas ---

class PaperResponse(BaseModel):
    id: int
    paper_id: str | None = None
    title: str
    authors: list[str] = []
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    source: str | None = None
    citation_count: int = 0
    qa_pair_count: int = 0
    pdf_path: str | None = None
    dataset_id: int | None = None

    class Config:
        from_attributes = True


class PaperListResponse(BaseModel):
    papers: list[PaperResponse]
    total: int
    page: int
    per_page: int


# --- Settings schemas ---

class ApiKeyInfo(BaseModel):
    provider_name: str
    masked_key: str
    updated_at: datetime | None = None


class SaveApiKeyRequest(BaseModel):
    api_key: str


class HFTokenStatus(BaseModel):
    configured: bool


class SaveHFTokenRequest(BaseModel):
    token: str


class S2TokenStatus(BaseModel):
    configured: bool


class SaveS2TokenRequest(BaseModel):
    token: str


# --- HuggingFace push ---

class HFPushRequest(BaseModel):
    repo_name: str
    description: str = ""
    private: bool = True


class HFPushResponse(BaseModel):
    repo_url: str
    hf_repo: str


# --- Provider/Task schemas ---

class ProviderInfo(BaseModel):
    name: str
    default_model: str
    api_key_env: str | None = None
    has_key: bool = False


class TaskInfo(BaseModel):
    name: str
    display_name: str
    domain: str | None = None


# --- Galaxy schemas ---

class GalaxyNode(BaseModel):
    id: str
    type: str  # "paper" or "qa"
    label: str
    size: float
    color: str
    cluster: int = 0
    # Paper-specific
    year: int | None = None
    citation_count: int | None = None
    abstract: str | None = None
    authors: list[str] | None = None
    url: str | None = None
    # QA-specific
    instruction: str | None = None
    output_preview: str | None = None


class GalaxyLink(BaseModel):
    source: str
    target: str
    weight: float = 1.0
    type: str = "paper_qa"  # paper_qa, keyword, co_topic


class ClusterInfo(BaseModel):
    id: int
    label: str
    color: str
    paper_count: int


class GalaxyData(BaseModel):
    nodes: list[GalaxyNode]
    links: list[GalaxyLink]
    clusters: list[ClusterInfo]


class PaperDetail(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    abstract: str | None = None
    year: int | None = None
    citation_count: int = 0
    url: str | None = None
    qa_pairs: list[QAPairResponse] = []


# --- Training schemas ---

class StartTrainingRequest(BaseModel):
    dataset_id: int | None = None
    dataset_path: str | None = None
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    model_size: str = "14B"
    lora_rank: int = 16
    lora_alpha: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = -1


class TrainingRunResponse(BaseModel):
    id: int
    run_name: str
    status: str
    dataset_id: int | None = None
    base_model: str | None = None
    model_size: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = -1
    dataset_path: str | None = None
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    adapter_path: str | None = None
    output_dir: str | None = None
    final_loss: float | None = None
    total_steps: int | None = None
    training_runtime_seconds: float | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None

    class Config:
        from_attributes = True


class TrainingRunListResponse(BaseModel):
    runs: list[TrainingRunResponse]
    total: int
    page: int
    per_page: int


class StartEvaluationRequest(BaseModel):
    training_run_id: int | None = None
    model_path: str | None = None
    test_dataset_path: str | None = None
    judge_model: str = "gpt-oss:120b"
    max_samples: int = 50


class EvaluationRunResponse(BaseModel):
    id: int
    run_name: str
    status: str
    training_run_id: int | None = None
    model_path: str | None = None
    test_dataset_path: str | None = None
    judge_model: str | None = None
    max_samples: int = 50
    factual_accuracy: float | None = None
    completeness: float | None = None
    technical_precision: float | None = None
    overall_accuracy: float | None = None
    purity: float | None = None
    entropy: float | None = None
    samples_scored: int = 0
    samples_skipped: int = 0
    samples_failed: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None

    class Config:
        from_attributes = True


class EvaluationRunListResponse(BaseModel):
    evaluations: list[EvaluationRunResponse]
    total: int
    page: int
    per_page: int


class EvaluationDetailResponse(EvaluationRunResponse):
    per_sample_results: list[dict] | None = None
    articles_log: list[dict] | None = None
    correction_results: dict | None = None


# --- Knobs / Correction schemas ---

class KnobsRequest(BaseModel):
    learning_rate: float | None = None


class CorrectionRequest(BaseModel):
    api_key: str | None = None
    score_threshold: float = 50.0
    model: str = "claude-opus-4-20250916"

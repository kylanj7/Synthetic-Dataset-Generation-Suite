"""LoRA Merge + GGUF Conversion Pipeline.

Ported from Qwen-Fine-Tuning-Lab/merge_and_convert_gguff.py.
All interactive selection stripped; programmatic API only.

Usage:
    from sdgs.web.engine.merge_convert import merge_and_convert
    gguf_path = merge_and_convert(
        adapter_path="outputs/run-name/final_adapter",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).parent
GGUF_OUTPUT_DIR = SCRIPT_DIR.parent.parent.parent / "models" / "gguf"

QUANT_OPTIONS = {
    "q4_k_m": "Best balance of quality/size (recommended)",
    "q5_k_m": "Higher quality, ~20% larger",
    "q6_k": "Near-lossless, ~40% larger",
    "q8_0": "Highest quality, largest size",
    "q4_k_s": "Smaller than q4_k_m, slightly lower quality",
    "q3_k_m": "Smallest usable size, lower quality",
    "q2_k": "Aggressive compression, lowest quality",
}


def load_run_metadata(adapter_path: str) -> Dict:
    """Load run_metadata.json from the training output directory."""
    adapter_dir = Path(adapter_path)
    for candidate in [adapter_dir.parent / "run_metadata.json",
                      adapter_dir / "run_metadata.json"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)
    return {}


def merge_lora(
    adapter_path: str,
    base_model: str,
    output_dir: str,
) -> str:
    """Merge LoRA adapter with base model to create a standalone model.

    Args:
        adapter_path: Path to the saved LoRA adapter
        base_model: HuggingFace model name (e.g. "Qwen/Qwen2.5-14B-Instruct")
        output_dir: Where to save the merged model

    Returns:
        Path to the merged model directory
    """
    print("=" * 80)
    print("STEP 1: MERGING LORA ADAPTER WITH BASE MODEL")
    print("=" * 80)
    print(f"Base Model: {base_model}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Output Path: {output_dir}")

    # Load base model in FP16 (no quantization for merging)
    print("\n[1/4] Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded!")

    # Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    print("Tokenizer loaded!")

    # Load and merge LoRA adapter
    print("\n[3/4] Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    print("LoRA adapter merged successfully!")

    # Save merged model
    print("\n[4/4] Saving merged model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"Merged model saved to: {output_dir}")
    print(f"Total size: {total_size / 1024**3:.2f} GB")

    return output_dir


def convert_to_gguf(
    model_dir: str,
    output_path: str,
    quant_method: str = "q4_k_m",
) -> str:
    """Convert a merged HuggingFace model to quantized GGUF format.

    Requires llama.cpp to be cloned/built (auto-handled).

    Args:
        model_dir: Path to the merged HuggingFace model
        output_path: Final output path for the quantized GGUF file
        quant_method: Quantization method (default: q4_k_m)

    Returns:
        Path to the final quantized GGUF file
    """
    print("\n" + "=" * 80)
    print("STEP 2: CONVERTING TO GGUF FORMAT")
    print("=" * 80)
    print(f"Model Path: {model_dir}")
    print(f"Quantization: {quant_method}")

    # Check/clone llama.cpp
    if not os.path.exists("llama.cpp"):
        print("\n[1/5] Cloning llama.cpp repository...")
        result = os.system("git clone https://github.com/ggerganov/llama.cpp")
        if result != 0:
            raise RuntimeError("Failed to clone llama.cpp repository")
        print("llama.cpp cloned successfully!")
    else:
        print("\n[1/5] llama.cpp repository already exists")

    # Build llama.cpp
    quantize_bin = "llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        print("\n[2/5] Building llama.cpp with CMake...")
        os.makedirs("llama.cpp/build", exist_ok=True)
        build_cmds = [
            "cd llama.cpp/build",
            "cmake .. -DLLAMA_CUBLAS=ON",
            "cmake --build . --config Release -j",
        ]
        result = os.system(" && ".join(build_cmds))
        if result != 0:
            print("WARNING: CUDA build failed, trying CPU-only...")
            os.system("rm -rf llama.cpp/build")
            os.makedirs("llama.cpp/build", exist_ok=True)
            build_cmds = [
                "cd llama.cpp/build",
                "cmake ..",
                "cmake --build . --config Release -j",
            ]
            result = os.system(" && ".join(build_cmds))
            if result != 0:
                raise RuntimeError("Failed to build llama.cpp")
        print("llama.cpp built successfully!")
    else:
        print("\n[2/5] llama.cpp already built")

    # Install GGUF dependencies
    print("\n[3/5] Installing GGUF dependencies...")
    os.system("pip install gguf protobuf -q")

    # Convert to FP16 GGUF
    print("\n[4/5] Converting to FP16 GGUF format...")
    fp16_output = f"{model_dir}.fp16.gguf"
    convert_script = (
        "llama.cpp/convert_hf_to_gguf.py"
        if os.path.exists("llama.cpp/convert_hf_to_gguf.py")
        else "llama.cpp/convert.py"
    )
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")

    result = os.system(
        f"python {convert_script} {model_dir} "
        f"--outfile {fp16_output} --outtype f16"
    )
    if result != 0:
        raise RuntimeError("FP16 GGUF conversion failed!")
    print(f"FP16 GGUF created: {fp16_output}")

    # Quantize
    print(f"\n[5/5] Quantizing to {quant_method}...")
    result = os.system(
        f"./llama.cpp/build/bin/llama-quantize "
        f"{fp16_output} {output_path} {quant_method}"
    )
    if result != 0:
        raise RuntimeError("Quantization failed!")

    # Clean up intermediate FP16 file
    if os.path.exists(fp16_output):
        os.remove(fp16_output)
        print(f"Cleaned up intermediate file: {fp16_output}")

    final_size = os.path.getsize(output_path)
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"Output File: {output_path}")
    print(f"Final Size: {final_size / 1024**3:.2f} GB")
    print(f"Quantization: {quant_method}")

    return output_path


def merge_and_convert(
    adapter_path: str,
    base_model: Optional[str] = None,
    quant_method: str = "q4_k_m",
    output_name: Optional[str] = None,
    keep_merged: bool = False,
) -> str:
    """Full pipeline: merge LoRA + convert to GGUF.

    Args:
        adapter_path: Path to the LoRA adapter
        base_model: HuggingFace model name (auto-detected from run_metadata if None)
        quant_method: GGUF quantization method
        output_name: Name for the output GGUF file (auto-generated if None)
        keep_merged: If True, keep the merged model directory

    Returns:
        Path to the final GGUF file
    """
    # Load run metadata for auto-detection
    run_metadata = load_run_metadata(adapter_path)

    if not base_model:
        base_model = run_metadata.get("base_model", "Qwen/Qwen2.5-14B-Instruct")
        print(f"Auto-detected base model: {base_model}")

    if not output_name:
        output_name = run_metadata.get("run_name", Path(adapter_path).name)

    # Create temp merged path
    merged_path = f"_temp_merged_{output_name}"
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Merge
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        merge_lora(adapter_path, base_model, merged_path)

        # Step 2: Convert to GGUF
        final_gguf_name = f"{output_name}-{quant_method}.gguf"
        final_gguf_path = str(GGUF_OUTPUT_DIR / final_gguf_name)

        temp_gguf = convert_to_gguf(merged_path, final_gguf_path, quant_method)

        # Step 3: Clean up
        if not keep_merged and os.path.exists(merged_path):
            print(f"\nCleaning up merged model directory: {merged_path}")
            shutil.rmtree(merged_path)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"GGUF Model: {final_gguf_path}")

        return final_gguf_path

    except Exception as e:
        # Clean up on failure
        if not keep_merged and os.path.exists(merged_path):
            shutil.rmtree(merged_path, ignore_errors=True)
        raise

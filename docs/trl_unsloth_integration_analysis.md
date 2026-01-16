# TRL and Unsloth Integration Analysis for OpenAdapt-ML

**Date**: January 2026
**Status**: Technical Analysis and Recommendation
**Scope**: Integrating TRL's SFTTrainer and Unsloth's memory optimizations with OpenAdapt-ML's VLM fine-tuning pipeline

---

## Executive Summary

This document analyzes how OpenAdapt-ML can integrate with TRL (Transformer Reinforcement Learning) and Unsloth to leverage production-grade training infrastructure while preserving OpenAdapt-ML's unique value propositions.

**Key Recommendation**: Adopt a **composition strategy** where OpenAdapt-ML provides the data pipeline, action DSL, and domain-specific features (demo conditioning, Set-of-Marks, grounding), while delegating the actual training loop to TRL's SFTTrainer with Unsloth's memory optimizations.

### Benefits
- 2x faster training with 70% less VRAM via Unsloth
- Production-tested training infrastructure via TRL
- Preserved OpenAdapt-ML differentiators (demo conditioning, SoM, action DSL)
- Maintained compatibility with Azure/Lambda Labs cloud training

---

## 1. What OpenAdapt-ML Offers That Is Unique

### 1.1 Demo-Conditioned Prompting (Validated)

OpenAdapt-ML's core value proposition is **trajectory-conditioned disambiguation of UI affordances**:

| Condition | First-Action Accuracy | Notes |
|-----------|----------------------|-------|
| Zero-shot | 46.7% (21/45) | Systematic spatial bias |
| **Demo-conditioned** | **100.0% (45/45)** | +53.3 pp improvement |
| Length-matched control | 57.8% (26/45) | Rules out verbosity effect |

This demonstrates that providing human demonstrations in prompts dramatically improves action selection. This is a **semantic** improvement, not just more tokens.

**Implementation location**: `openadapt_ml/experiments/demo_prompt/`

### 1.2 Set-of-Marks (SoM) DSL

OpenAdapt-ML implements a dual DSL system:

**Coordinate-based DSL** (for fine-tuned models):
```
CLICK(x=0.42, y=0.73)
TYPE(text="alice")
DONE()
```

**Index-based DSL** (for API models):
```
CLICK([1])
TYPE([2], "alice")
DONE()
```

The SoM approach achieved **100% accuracy on synthetic benchmarks** by eliminating coordinate prediction errors. This aligns with industry practice (OSWorld, Claude Computer Use, OpenAI Operator).

**Implementation location**: `openadapt_ml/datasets/next_action.py` (format_action, parse_action_som)

### 1.3 Action DSL with Thought Supervision

The system prompt enforces a structured response format:
```
RESPONSE FORMAT (required):
Thought: [Brief reasoning: what element to interact with and why]
Action: [Exactly one action, e.g., CLICK(x=0.35, y=0.42)]
```

This ReAct-style format enables:
- Interpretable agent reasoning
- Deterministic thought generation during training
- Future potential for thought supervision

**Implementation location**: `openadapt_ml/datasets/next_action.py` (SYSTEM_PROMPT, _generate_thought_for_step)

### 1.4 Gemini-Based Grounding Module

Real-time UI element detection using Google's Gemini vision API:
- Zero-shot detection on any UI
- Set-of-Marks overlay generation
- Natural language grounding ("find the login button")

**Implementation location**: `openadapt_ml/grounding/detector.py` (GeminiGrounder)

### 1.5 Windows Agent Arena (WAA) Benchmark Integration

Full integration with Microsoft's WAA benchmark:
- 154 tasks across 11 Windows domains
- Azure VM orchestration for parallel evaluation
- Accessibility tree + screenshot observation modes

**Implementation location**: `openadapt_ml/benchmarks/waa.py`, `openadapt_ml/benchmarks/azure.py`

### 1.6 Cloud GPU Training Infrastructure

Automated cloud training pipelines:
- Lambda Labs GPU rental integration
- Azure ML integration with ACR
- Dashboard with live training visualization
- SSE-based progress streaming

**Implementation location**: `openadapt_ml/cloud/lambda_labs.py`, `openadapt_ml/cloud/local.py`

### 1.7 Capture Ingestion Pipeline

Converts real-world screen recordings into training data:
- openadapt-capture format ingestion
- Video frame extraction
- Event-to-action mapping

**Implementation location**: `openadapt_ml/ingest/capture.py`

---

## 2. TRL Library Analysis

### 2.1 What TRL Provides

TRL is Hugging Face's production-grade library for post-training language models:

**Trainers Available**:
- `SFTTrainer` - Supervised Fine-Tuning (our primary interest)
- `DPOTrainer` - Direct Preference Optimization
- `GRPOTrainer` - Group Relative Policy Optimization (with vLLM support)
- `RewardTrainer` - Reward model training
- `PPOTrainer` - Proximal Policy Optimization

**Key Features**:
1. **VLM Support**: Full support for vision-language models including Qwen2.5-VL
2. **PEFT Integration**: Native LoRA/QLoRA support via `peft_config` parameter
3. **Dataset Flexibility**: Supports conversational, prompt-completion, and vision formats
4. **Assistant-Only Loss**: Compute loss only on assistant responses
5. **Packing**: Multiple examples in single sequence for efficiency
6. **Multi-GPU**: DeepSpeed, FSDP integration

### 2.2 TRL Dataset Format for VLMs

TRL's `DataCollatorForVisionLanguageModeling` expects:

```python
{
    "images": [Image.open("screenshot.png")],
    "messages": [
        {"role": "user", "content": "What is this?"},
        {"role": "assistant", "content": "A login screen."}
    ]
}
```

**Critical**: For VLMs, set `max_length=None` to avoid truncating image tokens.

### 2.3 TRL SFTTrainer VLM Example

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32),
    args=SFTConfig(
        max_length=None,  # Critical for VLMs
        assistant_only_loss=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
    ),
)
trainer.train()
```

---

## 3. Unsloth Library Analysis

### 3.1 What Unsloth Provides

Unsloth is an optimization framework that makes VLM fine-tuning:
- **2x faster** training
- **70% less VRAM** usage
- **8x longer context** support
- **No accuracy loss**

### 3.2 Supported Models

Unsloth supports:
- Qwen3-VL (2B, 4B, 8B, 32B) - including vision RL
- Qwen2.5-VL
- Gemma 3 (vision)
- Llama 3.2 Vision
- And many text-only models

### 3.3 Key Optimizations

1. **4-bit Quantization (QLoRA)**:
   ```python
   model, tokenizer = FastVisionModel.from_pretrained(
       'unsloth/Qwen3-VL-2B-Instruct',
       load_in_4bit=True,
       use_gradient_checkpointing='unsloth'
   )
   ```

2. **Selective Layer Fine-tuning**:
   ```python
   model = FastVisionModel.get_peft_model(
       model,
       finetune_vision_layers=True,    # Can disable to save memory
       finetune_language_layers=True,
       finetune_attention_modules=True,
       finetune_mlp_modules=True,
       r=16,
       lora_alpha=32,
   )
   ```

3. **Custom CUDA Kernels**: Unsloth's proprietary kernels optimize memory access patterns

4. **Gradient Checkpointing**: Extended checkpointing for long contexts

### 3.4 Unsloth + TRL Integration

Unsloth integrates seamlessly with TRL:

```python
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Load with Unsloth optimizations
model, tokenizer = FastVisionModel.from_pretrained(
    'unsloth/Qwen3-VL-2B-Instruct',
    load_in_4bit=True,
    use_gradient_checkpointing='unsloth'
)

# Apply LoRA
model = FastVisionModel.get_peft_model(model, r=16, lora_alpha=32)

# Train with TRL's SFTTrainer
FastVisionModel.for_training(model)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        output_dir='outputs',
        # Unsloth-specific settings
        remove_unused_columns=False,
        dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True},
    ),
)
trainer.train()
```

### 3.5 Memory Requirements

| Model | Standard | With Unsloth 4-bit |
|-------|----------|-------------------|
| Qwen3-VL-2B | ~16GB VRAM | ~8GB VRAM |
| Qwen3-VL-8B | ~48GB VRAM | ~16GB VRAM |
| Qwen3-VL-32B | ~128GB VRAM | ~40GB VRAM |

**Unsloth can train Qwen3-VL-8B on a free Google Colab T4 GPU (16GB).**

---

## 4. Specific Changes to Use TRL's SFTTrainer

### 4.1 Current OpenAdapt-ML Data Pipeline

**Current flow** (`openadapt_ml/datasets/next_action.py`):
```
Episodes -> build_next_action_sft_samples() -> List[Dict]
                                                  |
                                                  v
                                        {
                                          "images": [path],
                                          "messages": [...]
                                        }
```

**Current training** (`openadapt_ml/training/trainer.py`):
```python
# Custom training loop using QwenVLAdapter
adapter.prepare_inputs(batch)  # Custom preprocessing
adapter.compute_loss(inputs)   # Custom loss computation
optimizer.step()               # Manual optimization
```

### 4.2 Required Changes for TRL Integration

**Change 1: Dataset Format Conversion**

TRL expects PIL Images, not paths. Add a conversion layer:

```python
# openadapt_ml/datasets/trl_adapter.py (new file)

from PIL import Image
from datasets import Dataset

def convert_to_trl_format(samples: List[Dict]) -> Dataset:
    """Convert OpenAdapt-ML samples to TRL-compatible HuggingFace Dataset."""

    def load_images(sample):
        # Convert paths to PIL Images
        sample["images"] = [Image.open(p) for p in sample["images"]]
        return sample

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(load_images)
    return dataset
```

**Change 2: Message Format Alignment**

OpenAdapt-ML's format already matches TRL's expected structure:
```python
{
    "images": [Image],  # Changed from path to PIL Image
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

No changes needed to message structure.

**Change 3: Training Script Refactor**

Replace custom training loop with TRL's SFTTrainer:

```python
# openadapt_ml/training/trl_trainer.py (new file)

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from openadapt_ml.datasets.next_action import build_next_action_sft_samples
from openadapt_ml.datasets.trl_adapter import convert_to_trl_format

def train_with_trl(
    episodes: List[Episode],
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    output_dir: str = "checkpoints",
    use_som: bool = False,
    **training_kwargs
):
    """Train VLM using TRL's SFTTrainer."""

    # Build samples using OpenAdapt-ML's data pipeline
    samples = build_next_action_sft_samples(episodes, use_som=use_som)
    dataset = convert_to_trl_format(samples)

    # Configure LoRA
    peft_config = LoraConfig(
        r=training_kwargs.get("lora_r", 16),
        lora_alpha=training_kwargs.get("lora_alpha", 32),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    )

    # Configure training
    config = SFTConfig(
        output_dir=output_dir,
        max_length=None,  # Critical for VLMs
        assistant_only_loss=True,
        per_device_train_batch_size=training_kwargs.get("batch_size", 1),
        gradient_accumulation_steps=training_kwargs.get("grad_accum", 4),
        learning_rate=training_kwargs.get("learning_rate", 2e-4),
        num_train_epochs=training_kwargs.get("num_epochs", 3),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model_name,
        train_dataset=dataset,
        peft_config=peft_config,
        args=config,
    )

    trainer.train()
    return trainer
```

### 4.3 Backward Compatibility

Maintain the existing `QwenVLAdapter` for:
- Inference (generate method)
- API-backed evaluation
- Legacy training configs

The new TRL-based training becomes an **alternative**, not a replacement.

---

## 5. Integrating Unsloth While Keeping Unique Features

### 5.1 Unsloth Integration Layer

Create an optional Unsloth wrapper:

```python
# openadapt_ml/training/unsloth_adapter.py (new file)

from typing import Optional
import torch

def get_unsloth_model(
    model_name: str,
    load_in_4bit: bool = True,
    max_seq_length: int = 4096,
) -> tuple:
    """Load model with Unsloth optimizations if available."""

    try:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
            max_seq_length=max_seq_length,
        )

        return model, tokenizer, True  # Using Unsloth

    except ImportError:
        # Fallback to standard transformers
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)

        return model, processor, False  # Not using Unsloth


def apply_unsloth_peft(
    model,
    use_unsloth: bool,
    r: int = 16,
    lora_alpha: int = 32,
    finetune_vision: bool = True,
):
    """Apply LoRA with Unsloth optimizations if available."""

    if use_unsloth:
        from unsloth import FastVisionModel

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=finetune_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            random_state=3407,
        )

        FastVisionModel.for_training(model)
    else:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, config)

    return model
```

### 5.2 Unified Training Entry Point

```python
# openadapt_ml/training/unified_trainer.py (new file)

def train(
    episodes: List[Episode],
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    use_unsloth: bool = True,
    use_trl: bool = True,
    use_som: bool = False,
    output_dir: str = "checkpoints",
    **kwargs
):
    """Unified training entry point.

    Automatically uses the best available optimizations:
    - Unsloth for memory efficiency (if installed)
    - TRL for production training infrastructure
    - Falls back gracefully if dependencies missing

    Args:
        episodes: Training data as OpenAdapt-ML Episodes
        model_name: HuggingFace model identifier
        use_unsloth: Enable Unsloth optimizations
        use_trl: Use TRL's SFTTrainer
        use_som: Use Set-of-Marks DSL instead of coordinates
        output_dir: Where to save checkpoints
    """

    # Build OpenAdapt-ML samples (preserves our DSL, thoughts, etc.)
    samples = build_next_action_sft_samples(episodes, use_som=use_som)

    if use_trl:
        dataset = convert_to_trl_format(samples)

        if use_unsloth:
            # Unsloth + TRL path (optimal)
            model, tokenizer, is_unsloth = get_unsloth_model(
                model_name,
                load_in_4bit=True
            )
            model = apply_unsloth_peft(model, is_unsloth)

            from unsloth.trainer import UnslothVisionDataCollator
            from trl import SFTTrainer, SFTConfig

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model, tokenizer),
                train_dataset=dataset,
                args=SFTConfig(
                    output_dir=output_dir,
                    remove_unused_columns=False,
                    dataset_text_field='',
                    dataset_kwargs={'skip_prepare_dataset': True},
                    **kwargs
                ),
            )
        else:
            # TRL without Unsloth
            from trl import SFTTrainer, SFTConfig
            from peft import LoraConfig

            trainer = SFTTrainer(
                model=model_name,
                train_dataset=dataset,
                peft_config=LoraConfig(r=16, lora_alpha=32),
                args=SFTConfig(
                    output_dir=output_dir,
                    max_length=None,
                    assistant_only_loss=True,
                    **kwargs
                ),
            )

        trainer.train()
        return trainer

    else:
        # Fallback to existing OpenAdapt-ML training loop
        from openadapt_ml.training.trainer import train_supervised
        return train_supervised(episodes, model_name, output_dir, **kwargs)
```

### 5.3 Preserving OpenAdapt-ML Features

The integration preserves all unique features:

| Feature | Where It's Preserved |
|---------|---------------------|
| Demo conditioning | Data pipeline (demo_prompt module) |
| Set-of-Marks DSL | build_next_action_sft_samples(use_som=True) |
| Thought generation | _generate_thought_for_step() |
| Action DSL | format_action(), parse_action_som() |
| Grounding module | Unchanged (inference only) |
| WAA benchmarks | Unchanged (evaluation only) |
| Cloud training | Wrap new trainer in existing Lambda/Azure pipelines |
| Dashboard | Adapt to TRL's logging format |

---

## 6. Recommended Architecture

### 6.1 High-Level Architecture

```
                    OpenAdapt-ML (Domain Layer)
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  ┌─────────────┐   ┌──────────────┐   ┌──────────┐  │
    │  │   Capture   │   │    Demo      │   │   SoM    │  │
    │  │   Ingest    │   │  Retrieval   │   │ Grounding│  │
    │  └─────────────┘   └──────────────┘   └──────────┘  │
    │         │                │                  │        │
    │         └────────┬───────┴──────────────────┘        │
    │                  │                                   │
    │         ┌────────▼────────┐                         │
    │         │  Episode/Step   │                         │
    │         │    Schema       │                         │
    │         └────────┬────────┘                         │
    │                  │                                   │
    │         ┌────────▼────────┐                         │
    │         │   Action DSL    │                         │
    │         │ CLICK/TYPE/DONE │                         │
    │         └────────┬────────┘                         │
    │                  │                                   │
    │         ┌────────▼────────────────┐                 │
    │         │ build_next_action_sft_  │                 │
    │         │ samples() + Thought Gen │                 │
    │         └────────┬────────────────┘                 │
    │                  │                                   │
    └──────────────────┼───────────────────────────────────┘
                       │
    ┌──────────────────▼────────────────────────────────────┐
    │              Training Infrastructure Layer            │
    │                                                       │
    │  ┌─────────────────────────────────────────────────┐ │
    │  │              Unsloth (Optional)                  │ │
    │  │  - FastVisionModel                              │ │
    │  │  - 4-bit Quantization                           │ │
    │  │  - Custom CUDA Kernels                          │ │
    │  │  - Gradient Checkpointing                       │ │
    │  └──────────────────────┬──────────────────────────┘ │
    │                         │                             │
    │  ┌──────────────────────▼──────────────────────────┐ │
    │  │              TRL SFTTrainer                      │ │
    │  │  - DataCollatorForVisionLanguageModeling        │ │
    │  │  - Assistant-only loss                          │ │
    │  │  - Multi-GPU / DeepSpeed                        │ │
    │  │  - PEFT integration                             │ │
    │  └──────────────────────┬──────────────────────────┘ │
    │                         │                             │
    └─────────────────────────┼─────────────────────────────┘
                              │
    ┌─────────────────────────▼─────────────────────────────┐
    │                  Compute Layer                        │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
    │  │   Local     │   │   Lambda    │   │   Azure     │ │
    │  │   GPU       │   │   Labs      │   │   ML        │ │
    │  └─────────────┘   └─────────────┘   └─────────────┘ │
    └───────────────────────────────────────────────────────┘
```

### 6.2 Data Flow

```
1. Capture/Synthetic → Episode objects
                           │
2. Demo Retrieval ─────────┤ (optional conditioning)
                           │
3. build_next_action_sft_samples()
   - Apply SoM if enabled
   - Generate thoughts
   - Format action DSL
                           │
                           ▼
4. convert_to_trl_format()
   - Load images as PIL
   - Create HuggingFace Dataset
                           │
                           ▼
5. Unsloth FastVisionModel.from_pretrained()
   - 4-bit quantization
   - Memory-efficient loading
                           │
                           ▼
6. TRL SFTTrainer.train()
   - Assistant-only loss
   - Gradient accumulation
   - Learning rate scheduling
                           │
                           ▼
7. Save LoRA adapter
   - Compatible with existing inference code
```

### 6.3 Migration Path

**Phase 1**: Add TRL/Unsloth as optional dependencies
- Add `trl`, `unsloth` to optional deps in pyproject.toml
- Create adapter modules (`trl_adapter.py`, `unsloth_adapter.py`)
- Maintain backward compatibility with existing trainer

**Phase 2**: Validate equivalence
- Run same training with old vs new infrastructure
- Compare loss curves, evaluation metrics
- Ensure checkpoint compatibility

**Phase 3**: Default to TRL/Unsloth
- Make TRL the default trainer (with fallback)
- Update cloud training scripts
- Update dashboard to handle TRL logging format

**Phase 4**: Deprecate custom trainer
- Keep for edge cases but recommend TRL path
- Update documentation

---

## 7. Implementation Recommendations

### 7.1 Immediate Actions

1. **Add optional dependencies**:
   ```toml
   # pyproject.toml
   [project.optional-dependencies]
   training = [
       "trl>=0.26.0",
       "unsloth>=2025.1",  # Check exact version
       "peft>=0.10.0",
   ]
   ```

2. **Create TRL adapter module** at `openadapt_ml/datasets/trl_adapter.py`

3. **Create unified trainer** at `openadapt_ml/training/unified_trainer.py`

4. **Add CLI flag** to `scripts/train.py`:
   ```python
   @click.option("--use-trl/--no-trl", default=True)
   @click.option("--use-unsloth/--no-unsloth", default=True)
   ```

### 7.2 Testing Strategy

1. **Unit tests**: Verify data format conversion
2. **Integration test**: Train small model (Qwen3-VL-2B) on synthetic data
3. **Benchmark**: Compare memory usage with/without Unsloth
4. **Evaluation**: Ensure trained models pass existing eval suite

### 7.3 Documentation Updates

- Update `docs/cloud_gpu_training.md` with TRL/Unsloth options
- Add `docs/training_backends.md` explaining options
- Update README with memory requirements

---

## 8. Risk Analysis

### 8.1 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Unsloth version incompatibility | Medium | High | Pin exact versions, add CI tests |
| TRL API changes | Low | Medium | Pin TRL version, wrap in adapter |
| Memory regression on edge cases | Low | Medium | Benchmark suite, fallback option |
| Dashboard incompatibility | Medium | Low | Adapt logging callbacks |

### 8.2 Fallback Strategy

The existing `QwenVLAdapter` and custom training loop remain as fallbacks:
- If TRL import fails, use existing trainer
- If Unsloth import fails, use TRL without Unsloth
- CLI flags allow explicit selection

---

## 9. Conclusion

OpenAdapt-ML should adopt a **composition strategy** with TRL and Unsloth:

1. **Keep OpenAdapt-ML's domain expertise**: Demo conditioning, SoM, action DSL, grounding, benchmarks
2. **Delegate training mechanics**: Use TRL's battle-tested SFTTrainer
3. **Add memory efficiency**: Integrate Unsloth for 2x speed, 70% less VRAM
4. **Maintain flexibility**: Optional deps, fallback to custom trainer

This approach:
- Reduces maintenance burden (no custom training loop to maintain)
- Improves performance (Unsloth optimizations)
- Preserves differentiation (unique features remain in OpenAdapt-ML)
- Enables future extensibility (DPO, GRPO, reward modeling via TRL)

---

## 10. OpenCUA Integration Analysis

### 10.1 What is OpenCUA?

**OpenCUA** (Open Foundations for Computer-Use Agents) is a comprehensive open-source framework from XLang AI for building computer-use agents. It consists of three main components:

1. **OpenCUA Models**: End-to-end VLM-based computer-use foundation models
2. **AgentNet Dataset**: 22.6K human-annotated computer-use tasks across Windows, macOS, and Ubuntu
3. **AgentNetTool**: Cross-platform annotation infrastructure for capturing demonstrations

**Key Models Available** (all MIT licensed):

| Model | Base | Parameters | OSWorld-Verified (100 steps) | OSWorld-G (Grounding) |
|-------|------|------------|------------------------------|----------------------|
| OpenCUA-7B | Qwen2.5-VL-7B | 8B | 26.6% | 55.3% |
| OpenCUA-32B | Qwen2.5-VL-32B | 33B | 34.8% | 59.6% |
| OpenCUA-72B | Qwen2.5-VL-72B | 72B | 45.0% (SOTA) | 60.8% |

OpenCUA-32B is notable as it **surpasses OpenAI CUA (GPT-4o)** on 100-step tasks while being fully open-source.

### 10.2 Model Architecture

OpenCUA models are based on **Qwen2.5-VL** but with critical architectural modifications:

**Key Modifications from Base Qwen2.5-VL**:
1. **Positional Embeddings**: Multimodal Rotary Position Embedding (M-RoPE) replaced with 1D RoPE
2. **Tokenizer/ChatTemplate**: Aligned with Kimi-VL (not default Qwen)

**Important**: These modifications mean OpenCUA models **cannot be loaded with standard Qwen2.5-VL loaders**. They require:
```python
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

model = AutoModel.from_pretrained(
    "xlangai/OpenCUA-7B",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True  # REQUIRED
)
tokenizer = AutoTokenizer.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
```

### 10.3 Action Format and DSL

OpenCUA uses **PyAutoGUI-style actions** with absolute coordinates:

```python
# Click action
pyautogui.click(x=1443, y=343)

# Type action
pyautogui.write("search query")

# Keyboard shortcut
pyautogui.hotkey("ctrl", "s")

# Scroll
pyautogui.scroll(-3)

# Terminate
terminate()
```

**Coordinate System**: OpenCUA outputs coordinates on smart-resized images. Conversion to screen coordinates:
```python
def qwen25_smart_resize_to_absolute(model_x, model_y, original_width, original_height):
    resized_height, resized_width = smart_resize(
        original_height, original_width,
        factor=28, min_pixels=3136, max_pixels=12845056
    )
    rel_x = model_x / resized_width
    rel_y = model_y / resized_height
    abs_x = int(rel_x * original_width)
    abs_y = int(rel_y * original_height)
    return abs_x, abs_y
```

**Comparison with OpenAdapt-ML DSL**:

| Feature | OpenCUA | OpenAdapt-ML |
|---------|---------|--------------|
| Action format | `pyautogui.click(x=100, y=200)` | `CLICK(x=0.42, y=0.73)` |
| Coordinates | Absolute pixels | Normalized (0-1) |
| SoM support | No | Yes (`CLICK([1])`) |
| Thought format | Multi-line CoT | Single-line Thought: prefix |

### 10.4 AgentNet Dataset

AgentNet provides high-quality training data with a structure similar to OpenAdapt-ML:

**Task Format**:
```json
{
  "task_id": "20240927235321_...",
  "instruction": "sort the table in ascending order...",
  "task_completed": true,
  "task_difficulty": 3,
  "traj": [
    {
      "index": 0,
      "image": "screenshot.png",
      "value": {
        "observation": "I'm looking at a spreadsheet...",
        "thought": "I need to select the data range...",
        "action": "Click on cell C2",
        "code": "pyautogui.click(x=0.1632, y=0.2711)",
        "reflection": "The action has successfully selected..."
      }
    }
  ]
}
```

**Chain-of-Thought Levels**:
- **L1**: Action + Code only (lightweight)
- **L2**: Thought + Action + Code (default for OpenCUA models)
- **L3**: Observation + Thought + Action + Code (full reasoning)

**Dataset Scale**:
- 22.6K tasks total
- 12K Windows, 5K macOS, 5K Ubuntu
- 200+ applications and websites
- Average 18.6 steps per task

### 10.5 Compatibility with Unsloth

## ⚠️ CRITICAL: OpenCUA Models Are NOT Compatible with Unsloth ⚠️

**Status: INCOMPATIBLE - DO NOT ATTEMPT**

OpenCUA models (7B, 32B, 72B) **cannot be loaded with Unsloth** due to fundamental architectural changes from the base Qwen2.5-VL:

### Architectural Differences (Verified from HuggingFace Model Card)

| Component | Qwen2.5-VL (Unsloth-compatible) | OpenCUA (NOT compatible) |
|-----------|--------------------------------|--------------------------|
| Position Embedding | **M-RoPE** (Multimodal Rotary) | **1D RoPE** (standard) |
| Tokenizer | Qwen default | **Kimi-VL** tokenizer |
| Chat Template | Qwen default | **Kimi-VL** chat template |
| Loading Method | Standard transformers | `trust_remote_code=True` REQUIRED |
| vLLM Support | Yes | Not yet (in development) |

### Why This Matters

**M-RoPE vs 1D RoPE**: Qwen2.5-VL's M-RoPE decomposes positional embeddings to capture:
- 1D textual positions
- 2D visual positions (height × width)
- 3D video positions (temporal)

OpenCUA replaced this with standard 1D RoPE (text-only style), breaking Unsloth's optimized CUDA kernels which expect M-RoPE.

### What Will NOT Work

```python
# ❌ THIS WILL FAIL - DO NOT ATTEMPT:
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "xlangai/OpenCUA-7B",  # FAILS - incompatible architecture
    load_in_4bit=True
)
# Error: Unsloth expects M-RoPE, finds 1D RoPE
```

### What WILL Work

**For training custom data**: Use standard Qwen2.5-VL or Qwen3-VL with Unsloth:
```python
# ✅ This works - train your own data on base Qwen models:
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",  # Standard Qwen - works!
    load_in_4bit=True
)
# Then fine-tune on your openadapt-capture data
```

**For evaluation baselines**: Use OpenCUA models directly via transformers (inference only):
```python
# ✅ For inference/evaluation (not training):
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained(
    "xlangai/OpenCUA-7B",
    trust_remote_code=True,  # REQUIRED
    torch_dtype="auto"
)
```

### Summary

| Use Case | Recommended Approach |
|----------|---------------------|
| **Fine-tune on custom data** | Qwen2.5-VL + Unsloth + TRL ✅ |
| **Fine-tune on custom data** | OpenCUA + Unsloth ❌ NOT POSSIBLE |
| **Zero-shot evaluation** | OpenCUA models via transformers ✅ |
| **Compare baselines** | OpenCUA (inference) vs fine-tuned Qwen ✅ |

**Bottom line**: For training on openadapt-capture or enterprise data, use base Qwen2.5-VL/Qwen3-VL with Unsloth. OpenCUA models are useful for evaluation baselines only.

### 10.6 Compatibility with TRL SFTTrainer

**Current Status: PARTIALLY COMPATIBLE with Custom Collator**

TRL's SFTTrainer can work with OpenCUA models but requires custom handling:

```python
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from trl import SFTTrainer, SFTConfig

# Load OpenCUA model with trust_remote_code
model = AutoModel.from_pretrained(
    "xlangai/OpenCUA-7B",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained("xlangai/OpenCUA-7B", trust_remote_code=True)

# Custom data collator needed (similar to UnslothVisionDataCollator but adapted)
def opencua_collate_fn(batch):
    # Handle Kimi-VL tokenization
    # Process images with OpenCUA's image processor
    # Apply custom chat template
    pass

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=opencua_collate_fn,
    args=SFTConfig(
        max_length=None,
        remove_unused_columns=False,
    ),
)
```

**Challenges**:
1. No standard `DataCollatorForVisionLanguageModeling` for OpenCUA's format
2. Custom image preprocessing required (smart resize with factor=28)
3. Kimi-VL chat template must be applied correctly

### 10.7 Recommended Approach for OpenAdapt-ML

## ⚠️ IMPORTANT: Focus on Custom Data, Not AgentNet ⚠️

**OpenAdapt-ML's primary use case is fine-tuning on YOUR data:**
- openadapt-capture recordings
- Enterprise-specific GUI workflows
- Custom domain applications

**AgentNet is NOT needed** - it's OpenCUA's pre-existing training dataset. OpenCUA models are already trained on it. The value of OpenAdapt-ML is training on YOUR custom data.

### Recommended Architecture

```
Your Data (openadapt-capture, enterprise recordings)
    ↓
Export to TRL format (PIL Images + messages)
    ↓
Unsloth FastVisionModel (Qwen2.5-VL or Qwen3-VL)
    ↓
TRL SFTTrainer
    ↓
Fine-tuned model for YOUR domain
```

### OpenCUA Models: Use for Evaluation Only

OpenCUA models are useful as **zero-shot baselines** to compare against your fine-tuned models:

```python
# For evaluation/comparison (NOT training):
from transformers import AutoModel, AutoTokenizer

class OpenCUAEvaluator:
    """Use OpenCUA for zero-shot baseline evaluation only."""

    def __init__(self, model_name: str = "xlangai/OpenCUA-7B"):
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True  # REQUIRED
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def evaluate_zero_shot(self, screenshot, instruction):
        """Compare OpenCUA zero-shot vs your fine-tuned model."""
        pass
```

### What NOT to Do

❌ **Don't import AgentNet** - Your fine-tuned model should learn YOUR workflows
❌ **Don't try to fine-tune OpenCUA** - Incompatible with Unsloth
❌ **Don't create agentnet_adapter.py** - Not needed for custom data training

### What TO Do

✅ **Export your captures to TRL format** - Modify existing export script
✅ **Fine-tune Qwen2.5-VL/Qwen3-VL** - With Unsloth for efficiency
✅ **Use OpenCUA for baselines** - Zero-shot comparison only
✅ **Keep it simple** - One training path, minimal code

### 10.9 Special Considerations

**Coordinate System Mapping**:
```python
# OpenCUA uses absolute pixels after smart resize
# OpenAdapt-ML uses normalized 0-1 coordinates
# Conversion required at data import time

def agentnet_to_normalized(x: float, y: float, width: int, height: int) -> tuple:
    """Convert AgentNet relative coords (on resized image) to normalized."""
    # AgentNet stores as fraction of resized dimensions
    # We need fraction of original dimensions
    return (x, y)  # Already normalized in AgentNet format
```

**Multi-Image Context**:
OpenCUA models process up to 3 screenshots for context. OpenAdapt-ML's demo-conditioning aligns with this:
```python
# Both approaches provide historical context
# OpenCUA: last 3 screenshots
# OpenAdapt-ML: demo trajectory + current screenshot
```

**Action Format Mapping**:

| AgentNet Action | OpenAdapt-ML Action |
|-----------------|---------------------|
| `pyautogui.click(x=100, y=200)` | `CLICK(x=0.05, y=0.1)` |
| `pyautogui.write("text")` | `TYPE(text="text")` |
| `pyautogui.hotkey("ctrl", "s")` | `KEY(key="ctrl+s")` |
| `pyautogui.scroll(-3)` | `SCROLL(direction="down")` |
| `terminate()` | `DONE()` |

### 10.10 Risk Analysis for OpenCUA Integration

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OpenCUA model changes break adapter | Medium | Low | Pin model versions, adapter abstraction |
| AgentNet format changes | Low | Medium | Version-specific importers |
| Coordinate conversion errors | Medium | High | Extensive unit tests, visual validation |
| vLLM support adds complexity | Low | Low | Keep inference-only initially |
| License changes | Very Low | High | MIT license well-established |

### 10.11 Conclusion

OpenCUA represents a significant advancement in open-source computer-use agents. For OpenAdapt-ML integration:

**Immediate Value**:
- AgentNet dataset: 22.6K trajectories for training data augmentation
- OpenCUA models: Strong zero-shot baselines for WAA evaluation

**Limitations**:
- OpenCUA models are NOT directly compatible with Unsloth's FastVisionModel
- Custom handling required for TRL SFTTrainer
- Architectural modifications (1D RoPE, Kimi-VL tokenizer) prevent drop-in replacement

**Recommended Strategy**:
1. **Use AgentNet data** with standard Qwen2.5-VL models via Unsloth + TRL
2. **Use OpenCUA models** for inference/evaluation baselines only
3. **Monitor** for official Unsloth/vLLM support for OpenCUA models

This preserves OpenAdapt-ML's optimized training pipeline while leveraging OpenCUA's high-quality data and strong baseline models.

---

## References

### TRL Documentation
- [TRL Main Docs](https://huggingface.co/docs/trl)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [VLM Alignment in TRL](https://huggingface.co/blog/trl-vlm-alignment)
- [Fine-Tuning VLMs with TRL Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)

### Unsloth Documentation
- [Unsloth Main Site](https://unsloth.ai/)
- [Vision Fine-tuning](https://docs.unsloth.ai/basics/vision-fine-tuning)
- [Qwen3-VL Guide](https://docs.unsloth.ai/models/qwen3-vl-run-and-fine-tune)
- [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl)
- [GitHub](https://github.com/unslothai/unsloth)

### OpenCUA Documentation
- [OpenCUA Website](https://opencua.xlang.ai/)
- [OpenCUA GitHub](https://github.com/xlang-ai/OpenCUA)
- [OpenCUA Paper (arXiv)](https://arxiv.org/abs/2508.09123)
- [OpenCUA-7B on HuggingFace](https://huggingface.co/xlangai/OpenCUA-7B)
- [OpenCUA-32B on HuggingFace](https://huggingface.co/xlangai/OpenCUA-32B)
- [AgentNet Dataset](https://huggingface.co/datasets/xlangai/AgentNet)
- [OpenCUA Collection](https://huggingface.co/collections/xlangai/opencua-open-foundations-for-computer-use-agents)

### OpenAdapt-ML Internal Docs
- `docs/design.md` - Core architecture
- `docs/roadmap.md` - Build priorities
- `docs/experiments/demo_conditioned_prompting_results.md` - Demo validation
- `docs/set_of_marks_implementation.md` - SoM design

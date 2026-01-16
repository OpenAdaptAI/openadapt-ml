## COMPREHENSIVE REPORT: GRPO TRAINING SUPPORT IN OPENADAPT-ML

### EXECUTIVE SUMMARY

The openadapt-ml project currently has **SFT (Supervised Fine-Tuning) training fully implemented** via TRL's `SFTTrainer` with Unsloth optimizations, but **has zero support for GRPO (Group Relative Policy Optimization)** or preference-based training. The codebase needs substantial new infrastructure to support reward modeling, preference data collection, and GRPO trainer integration.

---

## 1. CURRENT TRAINING INFRASTRUCTURE

### 1.1 What's Implemented

**Training Architecture:**
- Location: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/training/`
- Main entry: `trl_trainer.py` with `train_with_trl()` function
- Core components:
  - `TRLTrainingConfig` dataclass with SFT hyperparameters
  - Unsloth FastVisionModel loader with fallback to standard transformers
  - LoRA adapter configuration
  - TRL `SFTTrainer` integration

**Dataset & Samples:**
- Location: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/datasets/next_action.py`
- Function: `build_next_action_sft_samples()`
- Converts Episodes → SFT samples with format:
  ```python
  {
    "images": [image_path],
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Goal: ... Look at screenshot..."},
      {"role": "assistant", "content": "Action: CLICK([1])"}
    ]
  }
  ```

**Supported Models:**
- Qwen3-VL (8B/2B)
- Qwen2.5-VL (7B)
- API adapters (Claude, GPT-5.1)
- Dummy adapter for testing

**Cloud Integration:**
- Location: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/`
- Lambda Labs integration: `lambda_labs.py` (102KB)
- Azure ML support: `azure_inference.py` + CLI orchestration
- Local training: `local.py` with HTTP dashboard
- All using CLI-first patterns

**Schema Support for Rewards:**
- Location: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/schema/episode.py`
- Step class has optional `reward: Optional[float]` field
- Episode class has optional `final_reward: Optional[float]` field
- BUT: Never used in actual training (SFT only)

### 1.2 Training Config Examples

Located in `/Users/abrichr/oa/src/openadapt-ml/configs/`:
- `qwen3vl_capture.yaml` - Real capture training
- `qwen3vl_synthetic*.yaml` - Synthetic scenario training
- All use YAML with:
  ```yaml
  model:
    name: Qwen/Qwen3-VL-2B-Instruct
    load_in_4bit: true
  lora:
    r: 8
    lora_alpha: 16
  training:
    num_train_epochs: 5
    per_device_train_batch_size: 1
    learning_rate: 5.0e-5
    early_stop_loss: 1.0
  ```

**No GRPO-specific configs exist.**

---

## 2. TRL/UNSLOTH INTEGRATION STATUS

### 2.1 Current TRL Integration

**Dependency:**
```python
# pyproject.toml
"trl>=0.12.0",
"datasets>=2.18.0",
```

**Current TRL Classes Used:**
- `SFTTrainer` - Only trainer currently used
- `SFTConfig` - Training configuration

**What's NOT Used from TRL:**
- `RewardTrainer` - For training reward models
- `GRPOTrainer` - For group relative policy optimization (TRL 0.7.10+)
- `DPOTrainer` - For direct preference optimization
- `PPOTrainer` - For reinforcement learning

### 2.2 GRPOTrainer Requirements (from TRL Docs)

From TRL's documentation pattern, `GRPOTrainer` requires:

**Data Format:**
```python
{
  "prompt": "...",  # Context/observation
  "chosen": "...",  # Preferred completion/action
  "rejected": "...", # Non-preferred completion/action
  # Optional:
  "images": [...],  # For VLMs
  "chosen_images": [...],
  "rejected_images": [...]
}
```

**Key Configuration:**
```python
class GRPOConfig:
  output_dir: str
  num_train_epochs: int
  per_device_train_batch_size: int
  learning_rate: float
  
  # GRPO-specific
  max_prompt_length: Optional[int] = None
  max_completion_length: Optional[int] = None
  temperature: float = 1.0
  
  # Policy model
  model_name: str  # Base model
```

**NOT in Current Codebase:**
- Preference pair data format
- Chosen/rejected action pairs
- Comparative loss computation
- Group-wise optimization logic

---

## 3. MODEL ADAPTERS ANALYSIS

### 3.1 Current Adapter System

**Location:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/models/`

**Files:**
1. `base_adapter.py` - Abstract `BaseVLMAdapter` class
   ```python
   @abstractmethod
   def prepare_inputs(batch: List[Dict]) -> Dict[str, Any]: ...
   
   @abstractmethod
   def compute_loss(inputs: Dict) -> torch.Tensor: ...
   
   @abstractmethod
   def generate(sample: Dict) -> str: ...
   ```

2. `qwen_vl.py` - `QwenVLAdapter` implementation
   - Supports Qwen3-VL and Qwen2.5-VL
   - Handles chat templates with images
   - Implements label masking for assistant-only loss
   - **Does NOT support:**
     - Reward prediction heads
     - Preference pair processing
     - Dual input handling (chosen vs rejected)

3. `api_adapter.py` - API backends (Claude, OpenAI)
4. `dummy_adapter.py` - For testing

### 3.2 What's Missing for GRPO

**Reward Model Head:**
```python
class RewardVLMAdapter(BaseVLMAdapter):
  def __init__(self, ...):
    self.model = base_model
    self.reward_head = nn.Linear(hidden_size, 1)  # MISSING
    
  def compute_reward(self, sample) -> float:
    # Forward pass
    # Extract final hidden state
    # Pass through reward head
    # Return scalar reward
    # MISSING IMPLEMENTATION
```

**Preference Pair Processing:**
```python
# Current (SFT):
prepare_inputs({"images": [...], "messages": [...]})

# Needed for GRPO:
prepare_inputs({
  "images": [...],  # Shared observation
  "chosen_messages": [...],  # Preferred action
  "rejected_messages": [...]  # Non-preferred action
})
```

**Dual Loss Computation:**
```python
def compute_preference_loss(self, batch):
  # Current: Single forward pass per sample
  
  # Needed for GRPO:
  # 1. Forward on chosen action
  # 2. Forward on rejected action
  # 3. Compute relative loss
  # 4. Group-wise ranking loss
```

---

## 4. CONFIG INFRASTRUCTURE

### 4.1 Current Config System

**Location:** `/Users/abrichr/oa/src/openadapt-ml/configs/`

**Structure:**
- YAML files for training runs
- Python loader in `openadapt_ml/config.py` (pydantic-settings)
- No TypedDict validation for training configs

**Example Config Fields:**
```yaml
model:
  name: string
  load_in_4bit: bool
  max_pixels: int

lora:
  r: int
  lora_alpha: int
  target_modules: [list]

training:
  num_train_epochs: int
  per_device_train_batch_size: int
  learning_rate: float
  early_stop_loss: float
```

### 4.2 Missing GRPO Config Support

**Need to add:**
```yaml
# For GRPO training
grpo:
  enabled: true
  max_prompt_length: 512
  max_completion_length: 64
  temperature: 1.0
  beta: 0.5  # KL coefficient
  group_size: 4  # Group size for relative scoring

# For reward model training
reward_model:
  enabled: true
  head_type: "linear"  # or "mlp"
  hidden_size: 768
  dropout: 0.1
  
# Preference data
preference_data:
  source: "human_feedback"  # or "model_comparison", "trajectory_ranking"
  format: "chosen_rejected"  # or "ranking_list"
```

---

## 5. DATASET & PREFERENCE DATA FORMAT

### 5.1 Current Episode Schema

**Location:** `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/schema/episode.py`

```python
@dataclass
class Step:
  step_index: int
  observation: Observation
  action: Action
  reasoning: Optional[str] = None
  reward: Optional[float] = None  # EXISTS BUT UNUSED
  
@dataclass
class Episode:
  episode_id: str
  instruction: str
  steps: List[Step]
  success: bool
  final_reward: Optional[float] = None  # EXISTS BUT UNUSED
  metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Current Usage:**
- `reward` and `final_reward` are defined but never populated
- Never used in training

### 5.2 Missing: Preference Pair Format

**Need new schema:**
```python
@dataclass
class ActionPair:
  """Preference comparison between two actions."""
  observation: Observation  # Shared context
  chosen_action: Action
  rejected_action: Action
  reasoning: Optional[str] = None
  human_rating: Optional[float] = None  # 0-1 preference score
  quality_score: Optional[float] = None
  metadata: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class PreferenceDataset:
  """Collection of preference pairs for training."""
  pairs: List[ActionPair]
  source: str  # "human_feedback", "model_comparison", "trajectory_ranking"
  created_at: datetime
  metadata: Dict[str, Any]
```

**Need dataset builder:**
```python
def build_preference_pairs(
  episodes: List[Episode],
  method: str = "trajectory_comparison"  # or "human_annotation", "reward_model"
) -> List[Dict[str, Any]]:
  """
  Convert episodes to preference pairs for GRPO training.
  
  Returns TRL-compatible format:
  {
    "prompt": "observation + goal",
    "chosen": "preferred action",
    "rejected": "rejected action",
    "images": [PIL Image],
    "chosen_images": [PIL Image],
    "rejected_images": [PIL Image]
  }
  """
```

---

## 6. WHAT'S MISSING FOR GRPO TRAINING

### 6.1 Core Missing Components

| Component | Status | Location | Priority |
|-----------|--------|----------|----------|
| RewardVLMAdapter | Missing | `models/` | Critical |
| GRPOTrainer wrapper | Missing | `training/` | Critical |
| Preference pair schema | Partial | `schema/` | Critical |
| Preference data builders | Missing | `datasets/` | High |
| GRPOTrainingConfig | Missing | `training/` | High |
| Reward model checkpoint save/load | Missing | `models/` | High |
| GRPO-specific evaluator | Missing | `evals/` | Medium |
| Config file templates | Missing | `configs/` | Medium |

### 6.2 Detailed Missing Implementations

#### A. Reward Model Architecture

```python
# MISSING: openadapt_ml/models/reward_vl.py

from openadapt_ml.models.base_adapter import BaseVLMAdapter

class RewardVLMAdapter(BaseVLMAdapter):
  """Reward model for preference-based training.
  
  Adds a scalar reward head to a base VLM.
  Used for:
  1. Training on human preference data
  2. Scoring actions for GRPO ranking
  """
  
  def __init__(self, model, processor, device=None):
    super().__init__(model, processor, device)
    self.reward_head = nn.Sequential(
      nn.Linear(model.config.hidden_size, 256),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(256, 1),
      nn.Sigmoid()  # Score 0-1
    )
    
  def compute_reward(self, sample: Dict) -> float:
    """Predict reward for an action."""
    inputs = self.prepare_inputs([sample])
    with torch.no_grad():
      outputs = self.model(**inputs)
      hidden_states = outputs.hidden_states[-1]
      final_token = hidden_states[:, -1, :]  # Last token
      reward = self.reward_head(final_token)
    return reward.item()
    
  def compute_loss(self, inputs):
    """Preference loss between chosen and rejected."""
    # Forward on both
    chosen_outputs = self.model(**inputs["chosen"])
    rejected_outputs = self.model(**inputs["rejected"])
    
    chosen_reward = self.reward_head(chosen_outputs.hidden_states[-1][:, -1, :])
    rejected_reward = self.reward_head(rejected_outputs.hidden_states[-1][:, -1, :])
    
    # Margin loss: chosen should be higher
    loss = F.margin_ranking_loss(
      chosen_reward, rejected_reward, 
      torch.ones_like(chosen_reward),
      margin=0.1
    )
    return loss
```

#### B. Preference Data Builders

```python
# MISSING: openadapt_ml/datasets/preference.py

def build_preference_pairs_from_trajectories(
  episodes_base: List[Episode],
  episodes_improved: List[Episode],
) -> List[Dict[str, Any]]:
  """
  Create preference pairs from two model runs.
  
  Uses episodes from base model and improved model,
  matching on same observation, labeling improved as "chosen".
  """
  # Implementation needed

def build_preference_pairs_from_human_feedback(
  episodes: List[Episode],
  feedback_file: str,  # JSON with action scores
) -> List[Dict[str, Any]]:
  """
  Create preference pairs from human annotated feedback.
  
  Feedback format:
  {
    "episode_id": "step_index": {"action": "...", "score": 0.9}
  }
  """
  # Implementation needed

def build_preference_pairs_from_ranking(
  episodes: List[Episode],
  comparator_fn: Callable[[Action, Action], bool],
) -> List[Dict[str, Any]]:
  """
  Create preference pairs using a comparator function.
  
  Could compare:
  - Success vs failure
  - Fast vs slow trajectories
  - Correct vs incorrect actions
  """
  # Implementation needed
```

#### C. GRPO Trainer Wrapper

```python
# MISSING: openadapt_ml/training/grpo_trainer.py

@dataclass
class GRPOTrainingConfig:
  """GRPO training configuration."""
  
  # Model
  model_name: str = "unsloth/Qwen3-VL-2B-Instruct"
  load_in_4bit: bool = True
  max_seq_length: int = 4096
  
  # LoRA
  lora_r: int = 16
  lora_alpha: int = 32
  lora_dropout: float = 0.0
  
  # Training
  num_epochs: int = 3
  batch_size: int = 1
  gradient_accumulation_steps: int = 4
  learning_rate: float = 2e-4
  warmup_ratio: float = 0.03
  
  # GRPO-specific
  max_prompt_length: int = 512
  max_completion_length: int = 64
  temperature: float = 1.0
  beta: float = 0.5  # KL weight
  group_size: int = 4  # For group ranking
  
  # Output
  output_dir: str = "checkpoints"
  save_strategy: str = "epoch"

def train_with_grpo(
  preference_pairs: List[Dict],
  config: GRPOTrainingConfig,
  base_path: Optional[Path] = None,
) -> str:
  """
  Train using GRPO on preference pairs.
  
  Steps:
  1. Load base model
  2. Create reward head
  3. Prepare preference data
  4. Run GRPOTrainer
  5. Save checkpoint
  """
  # Implementation needed
```

#### D. GRPO Config Examples

```yaml
# MISSING: configs/qwen3vl_preference_training.yaml

model:
  name: Qwen/Qwen3-VL-2B-Instruct
  load_in_4bit: true
  max_pixels: 262144

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05

grpo:
  max_prompt_length: 512
  max_completion_length: 64
  temperature: 1.0
  beta: 0.5
  group_size: 4

training:
  num_train_epochs: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 5.0e-5
  warmup_ratio: 0.1
  early_stop_loss: 0.05
  logging_steps: 1
```

---

## 7. IMPLEMENTATION PLAN FOR GRPO SUPPORT

### Phase 1: Schema & Data (Weeks 1-2)

**Goals:** Enable preference data representation and handling

**Tasks:**
1. [ ] Add `ActionPair` class to `openadapt_ml/schema/episode.py`
2. [ ] Add `PreferenceDataset` class to schema
3. [ ] Create `openadapt_ml/datasets/preference.py` with:
   - `build_preference_pairs_from_trajectories()`
   - `build_preference_pairs_from_human_feedback()`
   - `build_preference_pairs_from_ranking()`
4. [ ] Add tests in `tests/test_preference_data.py`

**Deliverable:** Preference pairs can be created from episodes

### Phase 2: Model Adapters (Weeks 2-3)

**Goals:** Create reward models and dual-input adapters

**Tasks:**
1. [ ] Create `openadapt_ml/models/reward_vl.py` with:
   - `RewardVLMAdapter` class
   - Reward head architecture
   - `compute_reward()` method
   - Preference loss computation
2. [ ] Update `openadapt_ml/models/base_adapter.py`:
   - Add optional `compute_reward()` method
   - Add optional `compute_preference_loss()` method
3. [ ] Update `QwenVLAdapter`:
   - Add preference pair handling
   - Support chosen/rejected paths
4. [ ] Add tests: `tests/test_reward_adapter.py`

**Deliverable:** Models can compute rewards and preference losses

### Phase 3: Training Wrapper (Weeks 3-4)

**Goals:** Create GRPO training entry point

**Tasks:**
1. [ ] Create `openadapt_ml/training/grpo_trainer.py` with:
   - `GRPOTrainingConfig` dataclass
   - `train_with_grpo()` function
   - `train_reward_model()` helper
   - Unsloth integration pattern
2. [ ] Create TRL `GRPOTrainer` wrapper:
   - Handle Unsloth model loading
   - Configure GRPOConfig from our config
   - Manage training loop
   - Save checkpoints
3. [ ] Add tests: `tests/test_grpo_trainer.py`

**Deliverable:** Can run GRPO training on preference pairs

### Phase 4: Configuration & Configs (Week 4)

**Goals:** Make GRPO training accessible via configs

**Tasks:**
1. [ ] Update `openadapt_ml/training/trl_trainer.py`:
   - Add `trainer_type: "sft" | "grpo"` parameter
   - Route to appropriate trainer
2. [ ] Create example configs:
   - `configs/qwen3vl_preference.yaml`
   - `configs/qwen3vl_reward_model.yaml`
3. [ ] Update CLI scripts to support GRPO training
4. [ ] Update README with GRPO quickstart

**Deliverable:** Can train GRPO via YAML config

### Phase 5: Evaluation & Integration (Weeks 5-6)

**Goals:** Evaluate GRPO models and integrate with cloud training

**Tasks:**
1. [ ] Create `openadapt_ml/evals/preference.py`:
   - Compute preference accuracy
   - Measure alignment with human feedback
   - Ranking correlation metrics
2. [ ] Add to benchmark suite: `openadapt_ml/benchmarks/`
3. [ ] Integrate with cloud:
   - Update `lambda_labs.py` for GRPO
   - Update `azure_inference.py` for GRPO
   - Add `--trainer-type grpo` to CLI
4. [ ] Add tests: `tests/test_grpo_integration.py`

**Deliverable:** GRPO models can be evaluated and trained in cloud

### Phase 6: Documentation (Week 6)

**Goals:** Document GRPO workflow

**Tasks:**
1. [ ] Create `docs/grpo_training.md` guide
2. [ ] Update `README.md` with GRPO section
3. [ ] Add example notebooks: `examples/grpo_training.ipynb`
4. [ ] Document preference data formats
5. [ ] Add troubleshooting guide

**Deliverable:** Users can follow GRPO tutorial

---

## 8. PRACTICAL IMPLEMENTATION DETAILS

### 8.1 Integration Points with Existing Code

**Preference Data Builder Integration:**
```python
# In openadapt_ml/training/trl_trainer.py
def train_with_trl(episodes, config, trainer_type="sft", ...):
  if trainer_type == "grpo":
    # Convert episodes to preference pairs
    from openadapt_ml.datasets.preference import build_preference_pairs_from_trajectories
    pairs = build_preference_pairs_from_trajectories(episodes)
    
    # Train with GRPO
    from openadapt_ml.training.grpo_trainer import train_with_grpo
    return train_with_grpo(pairs, config)
  else:
    # SFT path (existing)
    ...
```

**Reward Model Checkpoint Save:**
```python
# In openadapt_ml/models/reward_vl.py
def save_checkpoint(self, path: str):
  save_path = Path(path)
  save_path.mkdir(parents=True, exist_ok=True)
  
  # Save base model adapter
  self.model.save_pretrained(str(save_path / "base"))
  
  # Save reward head separately
  torch.save(self.reward_head.state_dict(), save_path / "reward_head.pt")
  
  # Save config
  with open(save_path / "config.json", "w") as f:
    json.dump({"model_type": "reward_vl"}, f)

@classmethod
def from_checkpoint(cls, path: str):
  path = Path(path)
  model = PeftModel.from_pretrained(str(path / "base"))
  reward_head = nn.Sequential(...)
  reward_head.load_state_dict(torch.load(path / "reward_head.pt"))
  return cls(model, processor, reward_head=reward_head)
```

**TRL GRPOTrainer Integration:**
```python
# Required TRL version: >=0.7.10
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
  output_dir=config.output_dir,
  per_device_train_batch_size=config.batch_size,
  learning_rate=config.learning_rate,
  num_train_epochs=config.num_epochs,
  
  # GRPO-specific
  max_prompt_length=config.max_prompt_length,
  max_completion_length=config.max_completion_length,
  temperature=config.temperature,
  beta=config.beta,
)

trainer = GRPOTrainer(
  model=model,
  train_dataset=dataset,
  args=training_args,
  data_collator=data_collator,
)
```

### 8.2 Preference Data Format for Different Sources

**From Model Comparison:**
```python
# Train on episodes from two runs
episodes_v1 = load_episodes("checkpoints/v1")
episodes_v2 = load_episodes("checkpoints/v2")

# Assumption: Same observations, different actions
pairs = build_preference_pairs_from_trajectories(
  episodes_v1,  # Older model
  episodes_v2,  # Newer model
  # Assumes v2 is better
)
```

**From Human Feedback:**
```python
# Load episodes
episodes = load_episodes("captures/my_task")

# Load human ratings
feedback = {
  "episode_001": {
    "0": {"action": "CLICK([1])", "score": 0.9},  # Good
    "1": {"action": "CLICK([2])", "score": 0.1},  # Bad
  }
}

pairs = build_preference_pairs_from_human_feedback(episodes, feedback)
```

**From Trajectory Ranking:**
```python
# All episodes complete same task, rank by quality
episodes = load_episodes("synthetic_login")

def compare_actions(a1, a2):
  # Return True if a1 > a2
  return success_rate(a1) > success_rate(a2)

pairs = build_preference_pairs_from_ranking(episodes, compare_actions)
```

### 8.3 GRPO Training Example Usage

```python
from openadapt_ml.training.grpo_trainer import train_with_grpo, GRPOTrainingConfig
from openadapt_ml.datasets.preference import build_preference_pairs_from_trajectories

# Load episodes from two models
base_episodes = load_episodes("checkpoints/base")
improved_episodes = load_episodes("checkpoints/v1")

# Build preference pairs
pairs = build_preference_pairs_from_trajectories(base_episodes, improved_episodes)

# Configure GRPO
config = GRPOTrainingConfig(
  model_name="unsloth/Qwen3-VL-2B-Instruct",
  num_epochs=5,
  beta=0.5,  # KL weight
  group_size=4,
  output_dir="checkpoints/grpo_v2",
)

# Train
checkpoint = train_with_grpo(pairs, config)
print(f"Trained GRPO model: {checkpoint}")
```

---

## 9. CLOUD TRAINING CONSIDERATIONS

### 9.1 Lambda Labs Integration

**Current Status:** `openadapt_ml/cloud/lambda_labs.py` (102KB)
- SFT training implemented
- Need to add GRPO support

**What to Add:**
```python
def run_grpo_training(
  instance_id: str,
  preference_data_path: str,  # Path to preference pairs
  config_path: str,
  output_dir: str = "checkpoints_grpo",
):
  """Run GRPO training on Lambda Labs instance."""
  # Upload preference data
  # Run training script
  # Poll status
  # Download checkpoint
```

### 9.2 Azure ML Integration

**Current Status:** `openadapt_ml/cloud/azure_inference.py`
- WAA benchmark orchestration
- Need GRPO support

**Pattern to Follow:**
- Create compute target with GRPO script
- Submit job with preference data
- Monitor via dashboard
- Sync output checkpoints

### 9.3 CLI Updates

**Current:** `openadapt_ml/cloud/lambda_labs.py` has CLI
```bash
python -m openadapt_ml.cloud.lambda_labs train --capture ...
```

**Add:**
```bash
python -m openadapt_ml.cloud.lambda_labs train-grpo \
  --preference-data preferences.json \
  --config configs/grpo.yaml \
  --instance-type gpu_1x_a10
```

---

## 10. TESTING STRATEGY

### 10.1 Unit Tests to Add

**File:** `tests/test_grpo_trainer.py`
```python
def test_grpo_config_creation():
  config = GRPOTrainingConfig()
  assert config.beta == 0.5
  assert config.group_size == 4

def test_build_preference_pairs():
  pairs = build_preference_pairs_from_trajectories(episodes_v1, episodes_v2)
  assert len(pairs) > 0
  assert "chosen" in pairs[0]
  assert "rejected" in pairs[0]

def test_reward_model_initialization():
  adapter = RewardVLMAdapter.from_pretrained("qwen3-2b")
  assert hasattr(adapter, "reward_head")

def test_reward_computation():
  sample = {"images": [...], "messages": [...]}
  reward = adapter.compute_reward(sample)
  assert 0 <= reward <= 1  # Sigmoid output
```

**File:** `tests/test_preference_data.py`
```python
def test_preference_format_trl_compat():
  pairs = build_preference_pairs_from_trajectories(episodes)
  for pair in pairs:
    assert "prompt" in pair
    assert "chosen" in pair
    assert "rejected" in pair
    assert "images" in pair
```

### 10.2 Integration Tests

```python
def test_grpo_training_dry_run(temp_episodes):
  """Test GRPO training flow with mocked TRL."""
  pairs = build_preference_pairs_from_trajectories(temp_episodes)
  
  config = GRPOTrainingConfig(num_epochs=1)
  
  # Mock GRPOTrainer
  with patch("openadapt_ml.training.grpo_trainer.GRPOTrainer"):
    checkpoint = train_with_grpo(pairs, config)
    assert checkpoint.exists()
```

---

## 11. KEY GAPS & RISKS

### 11.1 Critical Gaps

| Gap | Risk | Mitigation |
|-----|------|-----------|
| Preference data source | Model needs labeled data | Start with model comparison (automatic) |
| TRL GRPOTrainer stability | Potential API changes | Pin TRL version, test compatibility |
| Reward head architecture | Wrong design = poor performance | Research optimal architectures |
| Group size selection | Affects convergence | Ablation study needed |
| Unsloth GRPO support | May not be optimized | Fall back to standard transformers |

### 11.2 Performance Considerations

**Memory Requirements:**
- SFT: ~8GB for Qwen2B with batch_size=1
- GRPO: ~12GB (dual forward passes: chosen + rejected)

**Training Time:**
- SFT: 1-2 hours on A10 GPU
- GRPO: 2-3 hours (more forward passes)

**Recommendation:** Start with smaller models (2B) for GRPO validation

### 11.3 Data Requirements

**Preference Data Size:**
- Minimum: 100 pairs (proof of concept)
- Recommended: 500+ pairs (production)
- Each pair: 2 observations + 2 actions

**Collection Methods:**
1. **Automatic (easier):** Model comparison - run same task on v1 and v2, label v2 as "chosen"
2. **Semi-automatic:** Trajectory filtering - keep high-success, remove low-success
3. **Manual (hardest):** Human annotation - expert rates action quality

---

## 12. SUCCESS CRITERIA

### Phase 1 Complete When:
- [ ] Preference pairs can be created from episode data
- [ ] TRL-compatible format validated
- [ ] Tests pass for preference builders

### Phase 2 Complete When:
- [ ] RewardVLMAdapter loads and computes rewards
- [ ] Preference loss is computed correctly
- [ ] Checkpoint save/load works

### Phase 3 Complete When:
- [ ] GRPO training runs with synthetic data
- [ ] Training loss decreases
- [ ] Checkpoints are saved

### Phase 4 Complete When:
- [ ] YAML configs work for GRPO
- [ ] CLI accepts `--trainer-type grpo`
- [ ] Documentation is complete

### Full GRPO Support When:
- [ ] GRPO models outperform SFT baseline
- [ ] Cloud training works (Lambda Labs + Azure)
- [ ] Production example provided

---

## CONCLUSION

**Summary:**
- OpenAdapt-ML has solid SFT training via TRL + Unsloth
- Zero GRPO support currently exists
- Implementation requires ~6 weeks of focused development
- Key missing pieces: reward models, preference data, GRPO trainer wrapper

**Next Steps:**
1. **Immediate:** Implement Phase 1 (schema + preference data builders)
2. **Short-term:** Complete Phases 2-3 (models + trainer)
3. **Medium-term:** Integrate with cloud + evaluate
4. **Long-term:** Validate GRPO > SFT on benchmarks

**Success Metric:** GRPO-trained models achieve higher WAA task success rates than SFT baseline within 2x training time.

# Synthetic Login Jitter and Ablation

This note documents how layout jitter is implemented in the synthetic login
UI generator, why it is designed the way it is, and how to run with/without
jitter for ablations.

## 1. Design goals

- **Episode-consistent layout**
  - Within a single episode (login workflow), all UI elements keep fixed
    positions across frames.
  - The username box, password box, login button, and decoy `Help` button
    do not move between steps.
- **Episode-varying jitter**
  - Across episodes/sessions, the layout is randomly jittered.
  - This prevents models from memorizing absolute coordinates and forces
    them to actually read the UI.

This matches real GUI behavior: during one login attempt, the UI does not
randomly shift between actions, but different apps/contexts may have
slightly different layouts.

## 2. Implementation: episode-level jitter

**File:** `openadapt_ml/ingest/synthetic.py`

Key pieces:

- `LoginUIElements`
  - Dataclass holding absolute pixel bounds for key regions:
    - `username_box`, `password_box`, `login_button`.
- `_compute_login_layout(max_offset: int = 10, jitter: bool = True) -> LoginUIElements`
  - Samples a **single** login layout for an episode.
  - If `jitter=True`:
    - Applies small random offsets (bounded by `max_offset`) to the base
      positions of the username field, password field, and login button.
  - If `jitter=False`:
    - Uses the base, non-jittered positions.
  - Returns pixel bounds `(x, y, w, h)` for each element.
- `_draw_login_screen(username: str = "", password: str = "", layout: Optional[LoginUIElements] = None, jitter: bool = True)`
  - Renders a login screen given a **fixed** `layout`.
  - If `layout is None`, it calls `_compute_login_layout(jitter=jitter)`
    once and reuses that for all elements.
  - This function is otherwise deterministic given `layout` and the
    `username`/`password` strings.

The episode script:

- `_script_login_episode(root: Path, episode_id: str, username: str, password: str, jitter: bool = True) -> Episode`
  - Samples the layout **once per episode**:
    - `layout = _compute_login_layout(jitter=jitter)`.
  - Uses this `layout` for all frames/steps:
    - Step 0: blank login screen.
    - Step 1: click username field.
    - Step 2: type username.
    - Step 3: click password field.
    - Step 4: type password.
    - Step 5: click login.
    - Step 6: logged-in screen + `done`.
  - All click coordinates (`CLICK(x=..., y=...)`) are computed from the
    **same** `layout` via `_center(...)`, ensuring consistency between the
    visuals and the ground-truth actions.

Result:

- Within an episode, the username/password fields and login button are
  stable across steps.
- Across episodes, those elements are randomly shifted (if `jitter=True`).

## 3. `generate_synthetic_sessions` and the `jitter` flag

Public entry point:

```python
from openadapt_ml.ingest.synthetic import generate_synthetic_sessions

sessions = generate_synthetic_sessions(
    num_sessions=32,
    seed=123,
    output_dir="synthetic_train_dev",
    jitter=True,  # or False for ablations
)
```

Signature (simplified):

```python
def generate_synthetic_sessions(
    num_sessions: int = 10,
    seed: int | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    jitter: bool = True,
) -> List[Session]:
    ...
```

Behavior:

- `jitter=True` (default)
  - For each session/episode, `_script_login_episode(..., jitter=True)`
    samples a jittered layout.
  - This is the **hardened** setting used in the main Qwen synthetic login
    benchmark.
- `jitter=False`
  - Layouts use base positions with no random offsets.
  - Useful as a **control** condition for ablations.

## 4. Ablation: with vs without jitter

To understand the impact of layout jitter on training/fine-tuning, we
recommend a simple ablation:

1. **No-jitter environment (control)**
   - Call `generate_synthetic_sessions(..., jitter=False)` in both the
     training and evaluation configs.
   - Keep everything else identical (model, LoRA config, number of
     sessions, epochs, learning rate, etc.).
2. **Jittered environment (hardened)**
   - Call `generate_synthetic_sessions(..., jitter=True)`.
   - Again, keep all non-jitter settings identical.

For each environment, run:

- Base model eval (no LoRA / `--ignore-lora`).
- Fine-tuned model eval (with LoRA weights loaded).

Compare the standard metrics (from `eval_policy.py` and
`trajectory_matching.py`):

- `action_type_accuracy`
- `mean_coord_error`
- `click_hit_rate`
- `episode_success_rate`

Expected qualitative behavior:

- **No jitter**
  - Task is easier and partially solvable by coordinate heuristics.
  - LoRA may show smaller or ambiguous gains over base.
- **With jitter (per-episode)**
  - Base performance is lower (task is harder and more realistic).
  - Fine-tuned model should show a clearer improvement, especially on
    coordinate-quality metrics (`mean_coord_error`, `click_hit_rate`).

## 5. Invariants to preserve

When modifying or extending the synthetic login generator, keep these
invariants:

- **No intra-episode jitter**
  - Once a layout is sampled for an episode, all frames in that episode
    must reuse it.
- **Jitter across episodes**
  - Different episodes should see different layouts when `jitter=True`.
- **Action/layout consistency**
  - All ground-truth click coordinates must be derived from the same
    layout that is used for rendering, to avoid mismatches between what
    the model sees and the labeled target.

These constraints ensure the synthetic environment remains realistic while
still providing meaningful variation for robust VLM training and
evaluation.

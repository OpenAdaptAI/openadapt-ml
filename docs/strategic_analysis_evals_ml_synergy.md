# Strategic Analysis: openadapt-evals + openadapt-ml Synergy

*Date: 2026-03-02*
*Author: OpenAdapt Strategy*

---

## 1. The Value Chain: Environment + Brain = Complete RL Stack

### What We Actually Have

The two packages form a vertically integrated RL training stack for GUI agents:

```
openadapt-evals (ENVIRONMENT)              openadapt-ml (BRAIN)
+-----------------------------------+      +-----------------------------------+
| VM lifecycle (Azure + AWS)        |      | VLM adapters (Qwen, Claude, GPT)  |
| Pool orchestration (N workers)    |      | SFT training (TRL + Unsloth)      |
| WAA integration (QEMU + Windows)  |      | GRPO training loop                |
| RLEnvironment wrapper             |      | Rollout collector                 |
|   - reset() / step() / observe()  |      | Reward functions                  |
|   - pixel_action() (frac + abs)   |      | CoT warm-up                       |
|   - evaluate() (WAA verifier)     |      | Action DSL (CLICK/TYPE/WAIT/DONE) |
|   - collect_rollout()             |      | Trajectory schemas (Episode/Step) |
|   - stuck detection               |      | Demo retrieval                    |
| SSH tunnel management             |      | Cloud GPU orchestration           |
| Cost tracking                     |      | LoRA checkpointing                |
+-----------------------------------+      +-----------------------------------+
         |                                           |
         +------------- PyPI dependency -------------+
           openadapt-evals depends on openadapt-ml
           openadapt-ml[benchmarks] depends on openadapt-evals
```

The integration point is clean: `openadapt-ml`'s `GRPORolloutCollector` imports
`RLEnvironment` from `openadapt-evals`, calls `collect_rollout()` with a VLM-based
`agent_fn`, and receives back `RolloutStep` objects with binary rewards from the
WAA evaluator. The GRPO trainer then computes group-relative advantages and updates
the policy. Neither package works alone for RL training--you need both.

### Market Comparison

The GUI agent RL training landscape as of March 2026:

| Solution | Environment | Training | Open Source | Desktop Support |
|----------|-------------|----------|-------------|-----------------|
| **openadapt-evals + openadapt-ml** | Azure/AWS VMs, WAA, pool mgmt | SFT + GRPO, LoRA, cloud GPU | Yes (MIT) | Windows (QEMU) |
| WebAgent-R1 (ByteDance) | WebArena (browser only) | GRPO via veRL | Paper only, no infra | No |
| Agent-R1 | Custom envs | PPO/GRPO/REINFORCE++ via veRL | Partial | No |
| DigiRL (Xu et al.) | AndroidWorld | PPO | Research code | Android only |
| AgentTrek | Web tasks | SFT from synthesized data | Research code | No |
| OSWorld | Desktop VMs | Eval only (no training loop) | Eval harness only | Linux/macOS |
| TRL GRPOTrainer | None (you BYO env) | GRPO (text + VLM) | Yes | N/A |
| SWE-bench/SWE-agent | Docker containers | Eval only | Eval harness only | No |

**Key observation**: No one else ships a turnkey environment-to-training pipeline for
desktop GUI agents. Research groups publish papers with custom setups that cannot be
reused. TRL provides the training algorithm but not the environment. OSWorld provides
the environment but not the training loop. We provide both, connected via a clean API.

### Is There a Competitive Moat?

**Yes, but it is an operational moat, not a theoretical one.** The moat is:

1. **The environment is genuinely hard to build.** Running Windows 11 inside QEMU
   inside Docker on cloud VMs, with SSH tunnels, port forwarding workarounds (the
   socat proxy for port 5050), 128GB OS disks, pool orchestration across Azure and
   AWS, 35-minute cold boot handling, stuck detection, and task reset--this is months
   of accumulated operational knowledge encoded as working software. The
   `infrastructure/` directory in openadapt-evals (azure_vm.py, aws_vm.py, pool.py,
   ssh_tunnel.py, vm_monitor.py) represents significant engineering that is tedious
   and error-prone to replicate.

2. **The WAA integration surface is non-trivial.** The `WAALiveAdapter` handles
   accessibility tree fetching, element-based grounding, screenshot capture, the
   `/evaluate` endpoint patching, synthetic vs. real task ID routing, and coordinate
   normalization. This is glue code, but it is battle-tested glue code.

3. **The training side is NOT a moat.** GRPO is well-documented. TRL now provides it
   natively (v0.29.0 with `rollout_func` support). Our own architecture analysis
   (`grpo_architecture_analysis.md`) concluded that our 809-line custom trainer should
   be replaced with a ~200-line TRL adapter. The GRPO math is ~30 lines of PyTorch.
   Any competent ML team can implement this in a week.

4. **The schema and demo retrieval are a narrow moat.** The trajectory schema
   (Episode/Step/Action/Observation) and demo-conditioned prompting approach are
   published and open, but the specific integration with WAA and the retrieval
   pipeline represent domain expertise that takes time to replicate.

**Bottom line: The environment is the moat. The training code is not.**

---

## 2. Monetization Analysis

### Cost Structure

Running the RL training stack has real infrastructure costs:

| Resource | Cost | Notes |
|----------|------|-------|
| Azure D8ds_v5 (8 vCPU, 32GB) | $0.38/hr | General purpose VM for WAA |
| AWS m5.metal (96 vCPU) | $4.61/hr | Bare metal for KVM/QEMU |
| GPU for training (A10, Lambda Labs) | ~$0.60/hr | LoRA fine-tuning |
| Single rollout (15 steps, ~2 min) | $0.006-$0.02 | At Azure rate |
| GRPO step (8 rollouts) | $0.05-$0.16 | Sequential, single VM |
| 1000-step training run | $50-$160 | Plus GPU time |

With pool parallelism (N VMs), rollout collection scales linearly but cost scales
linearly too. The GPU training cost is relatively small compared to the environment
cost for online RL.

### Monetization Models

#### Model A: Open Core (Recommended)

| Layer | License | Revenue |
|-------|---------|---------|
| openadapt-ml (schemas, training, VLM adapters) | MIT (open) | Community + contributors |
| openadapt-evals (eval harness, RL env, adapters) | MIT (open) | Community + contributors |
| Managed RL Environment Service | Commercial | Subscription/usage |

The managed service would provide:
- Pre-configured VM pools with WAA pre-installed (skip the 35-min cold boot)
- Snapshot-based fast resets (~15s vs. ~5s task-setup-only, with determinism)
- Pool auto-scaling based on training job demand
- Trajectory storage and replay (collect once, train many times)
- Multi-tenant isolation (each customer gets their own VM pool)
- Dashboard for training progress, cost tracking, rollout visualization
- API endpoint: `POST /rollout` returns trajectory + reward (the customer never
  manages VMs)

This is the "Weights & Biases for GUI agent RL" play: the open-source library is
free; the hosted infrastructure is paid.

#### Model B: Infrastructure-as-a-Service (IaaS for RL)

Charge per-rollout or per-GPU-hour:
- $0.05/rollout for standard (sequential, shared pool)
- $0.15/rollout for priority (dedicated VM, fast reset)
- $X/hr for dedicated training clusters (pool of N VMs + GPU)

This is more transactional and easier to start with. The partner relationship
(discussed in section 6) could be the first customer.

#### Model C: Training-as-a-Service

Accept a model checkpoint and a set of tasks, run the full GRPO training loop,
return improved checkpoint + training metrics. The customer never touches VMs or
GPUs. This is the highest-margin option but requires more trust and a proven
track record.

#### Model D: Data Marketplace

Collect and sell high-quality GUI agent trajectories. Every rollout collected
through the platform produces structured trajectory data. Over time, this
becomes a dataset moat:
- Successful trajectories are training signal for SFT
- Failed trajectories are negative examples for reward modeling
- Cross-task trajectories enable transfer learning
- Trajectory diversity (different models, different strategies) enables
  preference learning

This is a long-term play that requires significant volume to be valuable, but
it compounds.

### Where Is the Real Value?

**The environment is where the value concentrates.** Here is why:

1. **Scarcity**: Cloud VMs with nested virtualization, QEMU, Windows licensing,
   and WAA integration are expensive and complex. A well-managed pool is scarce.

2. **Switching costs**: Once a partner's training pipeline is built around our
   `RLEnvironment` API (`reset`/`step`/`observe`/`evaluate`/`collect_rollout`),
   switching requires reimplementing all the VM management, WAA integration,
   and task handling. This is not trivial.

3. **Replicability of alternatives**: The GRPO training code is easy to replicate
   (TRL does it natively). The environment is hard to replicate (operational
   complexity, accumulated bug fixes, cloud provider quirks).

4. **Marginal cost vs. value**: A rollout costs us ~$0.01 in VM time. If it
   helps a customer train a model that automates a $50/hr process, the value
   capture opportunity is enormous.

---

## 3. Open Source Strategy

### What Should Stay Open

Everything that currently exists should remain MIT-licensed. Pulling code behind
a paywall after it is already open destroys trust and community. Specifically:

| Component | Repo | Why Keep Open |
|-----------|------|---------------|
| RLEnvironment API | openadapt-evals | Standard interface, community adoption |
| WAALiveAdapter | openadapt-evals | Enables self-hosting |
| Pool management (PoolManager) | openadapt-evals | Users can run their own pools |
| Azure/AWS VM management | openadapt-evals | Users can BYO cloud |
| GRPO training code | openadapt-ml | Easy to replicate anyway (TRL) |
| Trajectory schemas | openadapt-ml | Interoperability standard |
| VLM adapters | openadapt-ml | Community contributions |
| Demo retrieval | openadapt-ml | Core research contribution |

### What Could Be Premium (New Code, Not Existing)

| Feature | Why Premium |
|---------|-------------|
| Snapshot-based fast resets | Requires VM image management, not just code |
| Pool auto-scaling | Operational complexity, cost optimization algorithms |
| Multi-tenant isolation | Security, billing, resource allocation |
| Trajectory data marketplace | Curation, quality control, licensing |
| Pre-warmed VM pools | Capital expenditure (VMs running idle for fast starts) |
| Training job orchestration | End-to-end pipeline management, monitoring, alerts |
| Custom verifier registry | Domain-specific task verification (beyond WAA) |
| SLA guarantees | Uptime, latency, throughput commitments |

The key principle: **the open-source library enables self-hosting; the premium
service eliminates operational burden.** This is the same model as Elasticsearch
(open) vs. Elastic Cloud (paid), or Prometheus (open) vs. Grafana Cloud (paid).

### What Should Never Be Premium

- The `RLEnvironment` API itself (it is the interface standard)
- Basic training code (it is trivially replicable)
- Schemas and data formats (they need to be universal for adoption)
- Research contributions (demo retrieval, CoT warm-up)

---

## 4. Network Effects

### Direct Network Effects (Weak)

More users of the RL environment does not directly make the environment better
for each individual user. This is not a social network.

### Indirect Network Effects (Moderate to Strong)

Several indirect network effects could emerge:

#### A. Benchmark Standardization Effect

If multiple teams evaluate their GUI agents on WAA via openadapt-evals, results
become comparable. This creates a leaderboard dynamic:
- Teams WANT to use the same evaluation harness for credibility
- Researchers cite results from the standard harness
- New teams adopt the standard to be part of the conversation
- This makes openadapt-evals the "SWE-bench of desktop automation"

**Status**: Early. We need 3-5 published results from different teams to create
this dynamic. The partner's adoption could be the second data point (after our
own results).

#### B. Trajectory Data Network Effect

Every rollout collected through the platform produces structured trajectory data.
If we aggregate (with consent) across users:
- More trajectories = better demo retrieval (OpenAdapt's core differentiator)
- More diverse failures = better reward models
- More successful trajectories = better SFT training data
- Cross-model trajectories enable meta-learning

**This is the strongest potential network effect**, but it requires:
1. Many users collecting many rollouts
2. Consent and data governance framework
3. Quality filtering and curation pipeline

#### C. Verifier Registry Effect

The partner asked for a verifier registry. If we build it:
- Team A creates a verifier for "email client tasks"
- Team B creates a verifier for "spreadsheet tasks"
- Both contribute to a shared registry
- Each new verifier makes the platform more valuable for everyone
- This is the "app store" dynamic

**Status**: Not started. Depends on the partner relationship and future users.

#### D. Integration Effect

As more training frameworks (TRL, veRL, OpenRLHF) integrate with our
`RLEnvironment` API:
- Users of those frameworks get plug-and-play desktop environment support
- We become the default environment for desktop RL
- Switching costs increase for everyone in the ecosystem

**This is the strongest strategic play**: get `openadapt-evals` adopted as the
standard environment interface for desktop GUI agents, the way Gymnasium is the
standard for control tasks.

---

## 5. Build vs. Buy: The GRPO Trainer Question

### The Case for Dropping Our Custom GRPO Trainer

Our own architecture analysis (`grpo_architecture_analysis.md`) already makes this
case compellingly:

1. **Our custom trainer is 809 lines with 26 identified issues** (7 critical).
2. **TRL v0.29.0 provides native GRPO** with `rollout_func` support for multi-turn
   interactive environments, multimodal VLM support, and all the infrastructure
   we reimplemented (gradient accumulation, KL penalty, clipping, logging,
   multi-GPU, mixed precision).
3. **The migration path is clear**: ~200 lines of adapter code replacing ~800 lines
   of custom code.
4. **WebAgent-R1 and Agent-R1 use veRL**, not custom code. The research community
   is consolidating on established RL frameworks.

### What openadapt-ml Should Focus On Instead

The truly unique value in openadapt-ml is NOT the GRPO math. It is:

| Unique Asset | Why It Matters | Replicability |
|--------------|---------------|---------------|
| **Demo retrieval** | 46.7% -> 100% first-action accuracy. Core thesis. | Medium (research novelty) |
| **Trajectory schemas** | Standard Episode/Step/Action/Observation format | Low (we defined it) |
| **Multi-turn rollout collection** | Bridges TRL's single-turn API with interactive environments | Medium |
| **Action DSL** | CLICK/TYPE/WAIT/DONE with coordinate normalization | Easy to replicate |
| **CoT warm-up** | SFT with chain-of-thought before GRPO | Medium |
| **VLM adapter layer** | Unified interface across Qwen/Claude/GPT | Easy to replicate |

### Recommendation

**Phase out the custom GRPO trainer. Invest in a thin TRL adapter.**

Specifically:

```
KEEP (unique value):
  openadapt_ml/training/grpo/rollout_collector.py  -- bridges env <-> training
  openadapt_ml/training/grpo/reward.py             -- binary_task_success, group advantages
  openadapt_ml/training/grpo/config.py             -- our config interface
  openadapt_ml/training/grpo/cot_warmup.py         -- CoT SFT data generation

REPLACE (with TRL):
  openadapt_ml/training/grpo/trainer.py            -- 809 lines -> ~200 line TRL adapter

INVEST MORE IN (differentiators):
  Demo retrieval pipeline
  Trajectory schema evolution (multi-app, multi-turn)
  Environment-side features (fast reset, snapshot, verifier registry)
  Training data curation (trajectory quality scoring)
```

The rollout collector is the bridge code that makes us valuable: it knows how to
translate between TRL's `rollout_func` API and openadapt-evals' `RLEnvironment`.
Without this bridge, a user has to write their own. WITH this bridge, you get
plug-and-play GRPO training on desktop environments.

---

## 6. The Partner Relationship: Strategic Analysis

### What the Partner Asked For

8 specific features to make the eval infrastructure usable as an RL environment:

1. `pixel_action` -- convenience method for pixel-coordinate actions
2. `reset` -- environment reset with task loading
3. `observe` -- current observation without side effects
4. Example rollout script
5. EC2 guide (AWS support)
6. Verifier registry
7. `health_check`
8. Trajectory collection

### What We Built (and Are Building)

Items 1-4, 7-8 are implemented in PR #73 (`RLEnvironment`). Item 5 (AWS) is
implemented in the `aws_vm.py` module. Item 6 (verifier registry) is future work.

### Strategic Implications

#### A. Dependency Creation (Positive)

By building the features the partner requested, we make them dependent on our
infrastructure. Their training pipeline will import `RLEnvironment` from
`openadapt-evals` and use our `WAALiveAdapter`. Switching away means:
- Reimplementing VM management for their preferred cloud
- Reimplementing WAA integration (QEMU, Flask API, evaluate endpoint)
- Reimplementing the RL environment wrapper
- Rewriting their rollout collection code

This is healthy vendor lock-in IF we maintain quality and responsiveness.

#### B. Validation (Very Positive)

An external team using our infrastructure for their own training validates:
- The API design (if it works for them, it works for others)
- The operational reliability (if they can train on it, it is production-grade)
- The market need (someone is willing to build on it)

This is the "second customer" signal that investors and the community look for.

#### C. Feature Prioritization Risk (Manageable)

The partner's 8-feature request could pull us toward their specific needs at the
expense of our own roadmap. Mitigations:
- Their features align well with our architecture (they want the RL env, which
  we need for our own GRPO training)
- The verifier registry is genuinely useful for us too (custom verifiers for
  demo-conditioned evaluation)
- Set clear boundaries: we build the general-purpose RL environment API; they
  build their model-specific training on top

#### D. Revenue Opportunity

If the partner's training runs consume significant VM time, this is a natural
first customer for the managed RL environment service (Model A/B from section 2).
Even at cost-plus-margin pricing, this generates revenue and proves the business
model.

### How to Think About This Strategically

**The partner is not a distraction. They are a forcing function for product-market
fit.**

Their request forced us to:
1. Formalize the RL environment API (RLEnvironment class)
2. Add AWS support (not just Azure)
3. Think about pixel_action vs. element-based actions (important for generalization)
4. Build example scripts that prove the API works end-to-end
5. Consider verifier extensibility (beyond WAA's built-in evaluator)

All of these make the platform better for EVERYONE, including ourselves. The
partner's needs happen to align with building a general-purpose RL training
platform for GUI agents.

**Recommended posture**: Enthusiastic collaboration on the shared infrastructure
layer. Clear boundaries on the training layer (they use their own training code
on top of our environment, or they use openadapt-ml's GRPO module). Revenue
conversation when their usage is significant enough to warrant managed service.

---

## 7. Actionable Recommendations

### Immediate (This Month)

1. **Merge PR #73** (RLEnvironment) in openadapt-evals. This is the foundation
   for everything else. CI is passing. Ship it.

2. **Replace the custom GRPO trainer** with a TRL adapter. The architecture
   analysis already laid out the plan. Target ~200 lines of adapter code
   using TRL's `rollout_func` API.

3. **Publish the RL Quick Start guide** as a standalone page in the README.
   The `docs/rl_quick_start.md` is already written. Make it prominent.

4. **Get the partner running on our infrastructure.** Their success is our
   strongest validation signal. Prioritize unblocking them.

### Short-Term (Next 2 Months)

5. **Implement snapshot-based fast resets.** This is the single highest-impact
   feature for RL training throughput. Task-setup-only resets (~5s) work, but
   snapshot resets (~15s with full determinism) would enable reproducible
   experiments and faster iteration.

6. **Build the verifier registry.** Start simple: a directory of Python callables
   that take a screenshot and return a score. Let users contribute custom
   verifiers for their domains. This is the foundation for the network effect
   described in section 4C.

7. **Formalize the managed service offering.** Even if it starts as "we spin up
   VMs for you and give you an API endpoint," having a pricing page and sign-up
   flow creates revenue and validates the business model.

### Medium-Term (Next 6 Months)

8. **Push for adoption as the standard desktop RL environment.** Write a blog
   post showing GRPO training results on WAA. Submit to the Gymnasium/PettingZoo
   ecosystem. Integrate with veRL and OpenRLHF (not just TRL).

9. **Build the trajectory data pipeline.** Every rollout collected through the
   platform should be optionally stored, quality-scored, and made available for
   demo retrieval and SFT. This is the long-term data moat.

10. **Expand beyond WAA.** The `RLEnvironment` is adapter-agnostic. Build
    adapters for OSWorld (Linux/macOS), AndroidWorld, and web benchmarks.
    Each new adapter increases the platform's value without changing the core API.

### What NOT to Do

- Do not invest more engineering in the custom GRPO trainer. TRL does this better.
- Do not close-source existing code. The trust cost is too high.
- Do not build a custom training framework (veRL, OpenRLHF competitor). Focus on
  the environment and let the ecosystem handle training.
- Do not over-optimize for the partner's specific needs at the expense of
  generality. Build the general API; let them specialize on top.

---

## 8. Summary: The Two-Sided Platform Thesis

The strategic vision is a two-sided platform:

```
SUPPLY SIDE                    PLATFORM                     DEMAND SIDE
(Environment providers)        (openadapt-evals)            (Training teams)

WAA (Windows) ----+                                    +---- Partner (GRPO)
OSWorld (Linux) --+-->  RLEnvironment API  <--+--------+---- OpenAdapt (DC eval)
AndroidWorld -----+    (reset/step/observe    +--------+---- Research labs
WebArena ---------+     /evaluate/pixel_action)        +---- Enterprise customers
Custom verifiers -+                                    +---- Open-source community
```

**Supply side**: Every new environment adapter (WAA, OSWorld, AndroidWorld) makes
the platform more valuable. Each adapter is open source.

**Demand side**: Every team that trains GUI agents using the platform contributes
(optionally) trajectory data, verifiers, and bug reports. The managed service
captures revenue from those who want convenience over self-hosting.

**The flywheel**: More environments attract more trainers. More trainers produce
more trajectories. More trajectories improve demo retrieval. Better demo retrieval
attracts more users. More users justify more environment investment.

openadapt-ml is the brain. openadapt-evals is the body. Together they form a
complete organism. But in the business model, **the body (environment) is where
the value accrues**, because it is hard to build, expensive to run, and creates
switching costs. The brain (training) rides on top of commodity RL frameworks
(TRL, veRL) and differentiates through domain-specific innovations (demo
retrieval, trajectory schemas, CoT warm-up).

**The partner asking us to build RL environment features is the market telling
us where the value is. We should listen.**

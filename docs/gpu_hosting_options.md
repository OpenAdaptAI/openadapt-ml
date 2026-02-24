# GPU Compute Hosting Options for OpenAdapt.ai

**Context**: Fine-tuning Qwen3-VL (2B-8B), MIT-licensed open source project, no budget.

**Hardware requirements**:
- Qwen3-VL 2B with QLoRA: ~8-10 GB VRAM (fits on T4 16GB)
- Qwen3-VL 8B with QLoRA: ~16-24 GB VRAM (needs A10/A100/L4)
- Unsloth reduces VRAM by ~60% and speeds training 1.7x

---

## TIER 1: Apply Immediately (Highest Value)

### 1. AWS Cloud Credits for Open Source
- **Status**: ACTIVE (running since 2019, reaffirmed April 2025)
- **Credits**: Varies per project; AWS has given millions across 200+ OSS projects
- **GPUs**: Full AWS fleet (P4d/P5 with A100/H100, G5 with A10G, G6 with L4)
- **Eligibility**: OSI-approved license (MIT qualifies), active AWS account, valid payment method
- **Strings attached**: Credits for project infrastructure, not personal use
- **How to apply**: https://aws.amazon.com/blogs/opensource/aws-promotional-credits-open-source-projects/
- **Why #1**: Most directly relevant. MIT-licensed, established on GitHub — this program was designed for projects like OpenAdapt.

### 2. Google TPU Research Cloud (TRC)
- **Status**: ACTIVE (rolling admissions)
- **Credits**: Free access to 1,000+ Cloud TPU devices
- **Hardware**: Cloud TPUs (v2, v3, v4) — not GPUs, but excellent for VLM training via JAX/PyTorch XLA
- **Eligibility**: Anyone can apply; no academic affiliation required
- **Strings attached**: Must share research publicly (blog posts, open-source code, or papers)
- **How to apply**: https://sites.research.google/trc/
- **Why #2**: Completely free, rolling admissions. Sharing publicly is already met by MIT license.

### 3. NVIDIA Inception Program (Unlocks Multi-Platform Credits)
- **Status**: ACTIVE, free to join
- **Direct benefits**: Free DLI training credits, SDK access, preferred hardware pricing
- **Indirect benefits**:
  - AWS Activate credits: $25,000-$100,000
  - Nebius AI Lift: up to $150,000 in cloud credits
  - DGX Cloud Innovation Lab: 2 months of DGX Cloud access (select members)
- **Eligibility**: At least one developer, working website, officially incorporated, <10 years old
- **Strings attached**: No equity required
- **How to apply**: https://www.nvidia.com/en-us/startups/
- **Why #3**: Force multiplier. Free to join, unlocks $25K-$150K across partners.

### 4. Microsoft for Startups (Azure Credits)
- **Status**: ACTIVE (restructured July 2025)
- **Credits**: $1,000 immediately (no application), up to $5,000 with business verification, up to $150,000 for investor-backed startups
- **GPUs**: Azure NC/ND series (A100, H100, V100, T4)
- **Eligibility**: No funding required for $5K tier
- **Strings attached**: Credits expire (90-180 days)
- **How to apply**: https://www.microsoft.com/en-us/startups (self-service sign-up)
- **Why #4**: $1K-$5K with no funding needed. Quick to get started.

---

## TIER 2: Strong Options (Apply Soon)

### 5. Google for Startups Cloud Program
- **Credits**: $2,000 (pre-funding), up to $100,000 (VC-backed), up to $350,000 (AI-first track)
- **GPUs**: Full GCP fleet (A100, H100, L4, T4, TPU)
- **Eligibility**: Start tier requires no funding
- **How to apply**: https://startup.google.com/cloud/

### 6. Lambda Labs Research Grant
- **Credits**: Up to $5,000 in Cloud Credits plus mentoring
- **GPUs**: Lambda Cloud fleet (A100, H100, A6000)
- **Eligibility**: Academic researchers or published research projects; 50% academic discount also available
- **How to apply**: https://lambda.ai/research

### 7. fal.ai Research Grants
- **Credits**: Free compute (amount per project)
- **Eligibility**: Open to anyone — no formal degree required
- **How to apply**: Email grants@fal.ai with project description. See https://fal.ai/grants

### 8. AMD AI Developer Program & Developer Cloud
- **Credits**: $100 free (~50 hours) to start; additional credits for public projects
- **GPUs**: AMD Instinct MI300X (192GB VRAM)
- **Note**: Check ROCm compatibility with your training stack
- **How to apply**: https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html

### 9. a16z Open Source AI Grants
- **Type**: Cash grants (not investment, not equity)
- **Eligibility**: Open-source AI developers, hackers, researchers, small teams
- **How to apply**: Watch https://a16z.com/supporting-the-open-source-ai-community/ for batch announcements

---

## TIER 3: Free Platforms (Use Today, No Application)

| Platform | GPU | Free Limits | Best For |
|----------|-----|-------------|----------|
| [Kaggle](https://kaggle.com) | T4/P100 | 30 hrs/week, background exec | **Training Qwen3-VL 2B now** |
| [Google Colab](https://colab.research.google.com) | T4 | 15-30 hrs/week | Backup training |
| [Lightning.ai](https://lightning.ai) | T4 | 22 hrs/month | Development |
| [SageMaker Studio Lab](https://studiolab.sagemaker.aws) | T4 | 4hr sessions, 8hr/day, no credit card | Experimentation |
| [Modal](https://modal.com) | A100/H100 | $30/month credit (~2hr A100) | Burst training |
| [Paperspace Gradient](https://paperspace.com/gradient/free-gpu) | M4000 (8GB) | 6hr sessions, restartable | Too small for VLM training |

---

## TIER 4: Research/Academic Programs

| Program | Credits | Eligibility |
|---------|---------|-------------|
| [NSF ACCESS](https://access-ci.org) | Large GPU allocations (thousands of hours) | US institution PI |
| [Amazon Research Awards](https://amazon.science/research-awards) | $70K funds + $50K AWS credits | Academic institutions |
| [NVIDIA Academic Hardware Grant](https://academicgrants.nvidia.com/academicgrantprogram/s/Application) | Physical GPU hardware | Faculty at PhD-granting institutions |
| [HOSTKEY GPU Grant](https://hostkey.com/about-us/grants-for-scientific-projects-and-startups/) | Free GPU server time | Research/startup proposals (monthly windows) |

---

## TIER 5: Inference Hosting (Post-Training)

| Platform | Free Tier | Notes |
|----------|-----------|-------|
| [HuggingFace ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu) | H200 inference, daily quota | Best free option for deploying fine-tuned model |
| [Replicate](https://replicate.com) | Limited free runs | Deploying custom models as APIs |
| [Fireworks AI](https://fireworks.ai) | 10 RPM free, $1 credit | Supports Qwen models |
| [Together AI](https://together.ai) | No free tier ($5 min) | Startup accelerator offers up to $50K |

---

## Recommended Action Plan

| Phase | Action | Expected Value | Timeline |
|-------|--------|---------------|----------|
| **Today** | Sign up Microsoft for Startups | $1K-$5K Azure | Same day |
| **Today** | Start training on Kaggle | 30 hrs/week free T4 | Immediate |
| **This week** | Apply AWS OSS Credits | $5K-$50K+ | 2-4 weeks |
| **This week** | Apply Google TRC | Free TPU access | 1-2 weeks |
| **This week** | Email fal.ai (grants@fal.ai) | Free compute | 1-2 weeks |
| **This month** | Join NVIDIA Inception | Unlocks $25K-$150K partner credits | 2-4 weeks |
| **This month** | Apply Google for Startups | $2K-$350K | 1-2 weeks |
| **This month** | Apply Lambda Research Grant | $5K | 2-4 weeks |
| **Post-training** | Deploy on HuggingFace ZeroGPU | Free H200 inference | When model ready |

---

*Research conducted February 2026. Verify program availability before applying.*

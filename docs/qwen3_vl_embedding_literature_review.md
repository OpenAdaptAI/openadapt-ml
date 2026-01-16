# Literature Review: Qwen3-VL-Embedding and Multimodal Retrieval for GUI Automation

**Date:** January 2026
**Purpose:** Comprehensive survey of multimodal embedding models, visual document understanding, and GUI automation for retrieval-augmented systems.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Multimodal Embedding Models](#2-multimodal-embedding-models)
   - 2.1 [CLIP and Variants](#21-clip-and-variants)
   - 2.2 [BLIP-2](#22-blip-2)
   - 2.3 [LLaVA Embeddings](#23-llava-embeddings)
   - 2.4 [Qwen-VL Series](#24-qwen-vl-series)
   - 2.5 [VLM2Vec](#25-vlm2vec)
   - 2.6 [E5-V](#26-e5-v)
   - 2.7 [GME (General Multimodal Embeddings)](#27-gme-general-multimodal-embeddings)
   - 2.8 [Comparison of Architectures](#28-comparison-of-architectures)
3. [Visual Document Understanding](#3-visual-document-understanding)
   - 3.1 [LayoutLM Series](#31-layoutlm-series)
   - 3.2 [DocFormer](#32-docformer)
   - 3.3 [Donut](#33-donut)
   - 3.4 [Pix2Struct](#34-pix2struct)
4. [Retrieval-Augmented Generation for Vision](#4-retrieval-augmented-generation-for-vision)
   - 4.1 [Multimodal RAG Approaches](#41-multimodal-rag-approaches)
   - 4.2 [VisRAG](#42-visrag)
   - 4.3 [GUI Agent Retrieval](#43-gui-agent-retrieval)
5. [GUI Understanding Models](#5-gui-understanding-models)
   - 5.1 [CogAgent](#51-cogagent)
   - 5.2 [UI-TARS](#52-ui-tars)
   - 5.3 [Ferret-UI](#53-ferret-ui)
   - 5.4 [ScreenAI](#54-screenai)
6. [Embedding Techniques](#6-embedding-techniques)
   - 6.1 [Matryoshka Representation Learning](#61-matryoshka-representation-learning)
   - 6.2 [Contrastive Learning and InfoNCE](#62-contrastive-learning-and-infonce)
   - 6.3 [Hard Negative Mining](#63-hard-negative-mining)
   - 6.4 [Cross-Encoder Reranking](#64-cross-encoder-reranking)
7. [Benchmarks](#7-benchmarks)
   - 7.1 [MMEB / MMEB-V2](#71-mmeb--mmeb-v2)
   - 7.2 [MTEB / MMTEB](#72-mteb--mmteb)
   - 7.3 [ScreenSpot](#73-screenspot)
   - 7.4 [MoTIF](#74-motif)
   - 7.5 [AITW](#75-aitw)
   - 7.6 [OSWorld and AndroidWorld](#76-osworld-and-androidworld)
   - 7.7 [Mind2Web](#77-mind2web)
8. [Qwen3-VL-Embedding Deep Dive](#8-qwen3-vl-embedding-deep-dive)
9. [Implications for GUI Automation](#9-implications-for-gui-automation)
10. [References](#10-references)

---

## 1. Introduction

The intersection of multimodal embeddings and GUI automation represents a rapidly evolving research frontier. As autonomous agents increasingly interact with graphical user interfaces, the ability to semantically understand and retrieve relevant visual-textual information becomes critical. This literature review surveys the state-of-the-art in multimodal embedding models, with particular focus on their application to GUI understanding and retrieval-augmented systems.

The core challenge lies in creating unified representations that capture both visual layout (icons, buttons, text fields) and semantic content (labels, instructions, context) in a way that enables effective retrieval and reasoning. Recent advances in vision-language models (VLMs) have opened new possibilities for addressing this challenge.

---

## 2. Multimodal Embedding Models

### 2.1 CLIP and Variants

#### CLIP (Contrastive Language-Image Pre-training)

CLIP, introduced by OpenAI in 2021, established the foundation for modern multimodal embeddings. The model learns visual concepts from natural language supervision through contrastive pre-training on 400 million image-text pairs.

**Architecture:**
- Dual-encoder design with a Vision Transformer (ViT) or widened ResNet for images
- 12-layer Transformer for text
- Both encoders project into a shared 512-dimensional latent space
- Symmetric cross-entropy loss optimizes similarity scores between correct and incorrect pairings

**Key Innovation:** CLIP demonstrated that predicting which caption goes with which image is a scalable and effective pre-training objective. The model enables zero-shot transfer to downstream tasks by using natural language to reference learned visual concepts.

**Paper:** Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

#### OpenCLIP

OpenCLIP is an open-source implementation of CLIP maintained by ML Foundations, trained on datasets including LAION-400M, LAION-2B, and DataComp-1B.

**Available Models:**
- ViT-L/14 trained on LAION-2B
- ViT-H/14 with higher performance
- ViT-bigG/14 for maximum capacity

**Key Finding:** Large batch sizes (up to 159k) significantly improve performance in contrastive learning due to the N-squared logit matrix relationship.

**Resources:** [GitHub - mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

#### SigLIP

SigLIP (Sigmoid Loss for Language Image Pre-training), developed by Google, proposes a pairwise sigmoid loss instead of the softmax-based contrastive loss used in CLIP.

**Advantages:**
- Better performance with smaller batch sizes
- More memory efficient, enabling larger training batches without additional resources
- Optimal performance achieved with batch size of 32k (vs. much larger batches needed for CLIP)

**SigLIP 2 (2025)** extends the original with:
- Captioning-based pretraining
- Self-supervised losses (self-distillation, masked prediction)
- Online data curation
- Models from ViT-B (86M) to ViT-g (1B) parameters
- Improved multilingual understanding and fairness

**Papers:**
- Original: [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- SigLIP 2: [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)

### 2.2 BLIP-2

BLIP-2 (Bootstrapping Language-Image Pre-training) from Salesforce introduces an efficient pre-training strategy using frozen pre-trained image encoders and frozen large language models.

**Architecture:**
- Lightweight Querying Transformer (Q-Former) bridges the modality gap
- Two-stage training:
  1. Bootstrap vision-language representation learning from frozen image encoder
  2. Bootstrap vision-to-language generative learning from frozen LLM

**Efficiency:** BLIP-2 achieves state-of-the-art with only ~188M trainable parameters (less than 2% of an 11B LLM), outperforming Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters.

**Paper:** Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)

### 2.3 LLaVA Embeddings

LLaVA (Large Language and Vision Assistant) is an end-to-end trained multimodal model connecting a vision encoder and LLM.

**Architecture:**
- CLIP ViT-L/14 visual encoder
- Vicuna LLM backbone
- Trainable projection matrix (linear in v1, MLP in v1.5) connecting visual to language embeddings

**Training:**
1. **Stage 1:** Pre-training for feature alignment (only projection matrix updated)
2. **Stage 2:** Visual instruction tuning with multimodal data

**LLaVA-1.5 Improvements:**
- MLP vision-language connector instead of linear projection
- CLIP-ViT-L-336px with higher resolution
- Academic-task-oriented VQA data
- State-of-the-art across 11 benchmarks with only 1.2M training examples

**Papers:**
- Original: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) (NeurIPS'23 Oral)
- LLaVA-1.5: [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)

### 2.4 Qwen-VL Series

The Qwen-VL series from Alibaba represents a comprehensive evolution in vision-language models.

#### Qwen-VL (2023)
Initial model with visual receptor, input-output interface, 3-stage training pipeline, and multilingual multimodal corpus. Implements grounding and text-reading through image-caption-box tuple alignment.

**Paper:** [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)

#### Qwen2-VL (2024)
**Key Innovations:**
- **Naive Dynamic Resolution:** Dynamically processes images of varying resolutions into different numbers of visual tokens
- **Multimodal Rotary Position Embedding (M-RoPE):** Fuses positional information across text, images, and videos
- **Unified paradigm** for processing images and videos

**Model Sizes:** 2B, 8B, 72B parameters. Qwen2-VL-72B achieves results comparable to GPT-4o and Claude3.5-Sonnet.

**Paper:** [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)

#### Qwen2.5-VL (2025)
Enhanced understanding and interaction with foundational capabilities in visual recognition, object localization, document parsing, and long-video comprehension.

**Paper:** [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)

#### Qwen3-VL (2025)
The flagship model supporting 256K-token context windows with interleaved text, images, and video.

**Architectural Upgrades:**
- Enhanced interleaved-MRoPE for spatial-temporal modeling
- DeepStack integration for multi-level ViT features
- Text-based time alignment for video temporal grounding

**Performance:**
- 100% accuracy in needle-in-haystack tests for 30-minute videos
- 99.5% accuracy for 2-hour videos (~1M tokens)
- 96.5% on DocVQA, 875 points on OCRBench
- 39-language OCR support

**Model Sizes:** 2B, 4B, 8B, 32B (dense) and 30B-A3B, 235B-A22B (MoE)

**Paper:** [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)

### 2.5 VLM2Vec

VLM2Vec (Vision-Language Model to Vector) transforms any vision-language model into an embedding model through contrastive training.

**Key Contribution:** Unlike CLIP and BLIP which encode text or images independently, VLM2Vec processes any combination of images and text to generate fixed-dimensional vectors based on task instructions.

**MMEB Benchmark:** Introduced alongside VLM2Vec, covering 4 meta-tasks (classification, VQA, multimodal retrieval, visual grounding) and 36 datasets.

**Results:** 10-20% absolute improvement over existing multimodal embedding models on both in-distribution and out-of-distribution datasets.

**VLM2Vec-V2 (2025)** extends to videos and visual documents with MMEB-V2 benchmark featuring 78 tasks across 9 meta-tasks.

**Papers:**
- Original: [arXiv:2410.05160](https://arxiv.org/abs/2410.05160) (ICLR 2025)
- V2: [arXiv:2507.04590](https://arxiv.org/abs/2507.04590)

### 2.6 E5-V

E5-V adapts MLLMs for universal multimodal embeddings using prompts to bridge the modality gap.

**Key Innovation:** Single-modality training approach where the model is trained exclusively on text pairs, reducing training costs by approximately 95% while achieving state-of-the-art performance across text-image retrieval, composed image retrieval, sentence embeddings, and image-image retrieval.

**Paper:** [arXiv:2407.12580](https://arxiv.org/abs/2407.12580)

### 2.7 GME (General Multimodal Embeddings)

GME from Alibaba's Tongyi Lab is an MLLM-based dense retriever for Universal Multimodal Retrieval (UMR).

**Key Features:**
- Supports three input types: text, image, and image-text pairs
- Enables Any2Any search (text retrieval, image retrieval from text, image-to-image)
- Based on Qwen2-VL with contrastive learning fine-tuning

**Training Data:** Large-scale synthetic fused-modal training dataset to address modality imbalance in existing data.

**Paper:** [arXiv:2412.16855](https://arxiv.org/abs/2412.16855)

### 2.8 Comparison of Architectures

| Model | Architecture | Training Data | Embedding Dim | Key Strength |
|-------|-------------|---------------|---------------|--------------|
| CLIP | Dual encoder (ViT + Transformer) | 400M pairs | 512 | Zero-shot transfer |
| SigLIP | Dual encoder with sigmoid loss | WebLI | 768-1024 | Batch efficiency |
| BLIP-2 | Q-Former + frozen LLM | Multiple | Varies | Parameter efficiency |
| LLaVA | ViT + LLM + projection | 1.2M | LLM hidden size | Instruction following |
| Qwen3-VL-Embedding | Qwen3-VL backbone | Multi-stage | 3584 | MRL, 32k context |
| VLM2Vec | Any VLM backbone | MMEB | Varies | Task instruction |
| E5-V | MLLM backbone | Text pairs | Varies | Training efficiency |
| GME | Qwen2-VL backbone | Synthetic | Varies | Any2Any retrieval |

---

## 3. Visual Document Understanding

### 3.1 LayoutLM Series

Microsoft's LayoutLM series pioneered joint modeling of text and layout information for document understanding.

#### LayoutLM (2019)
First model to jointly learn text and layout in a single framework for document-level pre-training.

**Results:** Form understanding (70.72 to 79.27), receipt understanding (94.02 to 95.24), document classification (93.07 to 94.42).

**Paper:** [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)

#### LayoutLMv2 (2020)
Introduced two-stream multi-modal Transformer encoder with:
- Masked visual-language modeling
- Text-image alignment and matching tasks

**Results:** FUNSD (0.7895 to 0.8420), CORD (0.9493 to 0.9601), SROIE (0.9524 to 0.9781), DocVQA (0.7295 to 0.8672)

**Paper:** [arXiv:2012.14740](https://arxiv.org/abs/2012.14740)

#### LayoutLMv3 (2022)
Unified architecture with text and image masking objectives for both text-centric and image tasks.

**Paper:** [arXiv:2204.08387](https://arxiv.org/pdf/2204.08387)

#### LayoutXLM
Multilingual extension supporting 7 languages with XFUND benchmark.

**Paper:** [arXiv:2104.08836](https://arxiv.org/abs/2104.08836)

### 3.2 DocFormer

DocFormer is a multi-modal transformer architecture for Visual Document Understanding (VDU) that combines text, vision, and spatial features.

**Key Features:**
- Novel multi-modal self-attention layer
- Shared learned spatial embeddings across modalities
- CNN backbone for visual feature extraction
- End-to-end training

**Results:** State-of-the-art on 4 datasets, sometimes beating models 4x its size.

**Paper:** [arXiv:2106.11539](https://arxiv.org/abs/2106.11539) (ICCV 2021)

### 3.3 Donut

Donut (Document Understanding Transformer) is the first OCR-free VDU model, directly mapping raw images to desired outputs.

**Motivation:** Traditional OCR-based approaches suffer from:
1. High computational costs
2. Inflexibility across languages/document types
3. OCR error propagation

**Architecture:** Transformer encoder-decoder learning to predict next words conditioned jointly on image and previous text context.

**Includes:** SynthDoG (Synthetic Document Generator) for flexible pre-training across languages and domains.

**Paper:** [arXiv:2111.15664](https://arxiv.org/abs/2111.15664) (ECCV 2022)

### 3.4 Pix2Struct

Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML.

**Key Innovation:** Screenshot parsing objective that predicts HTML-based parse from masked screenshots, leveraging the web's rich visual elements and clean HTML structure.

**Model Sizes:** 282M (base) and 1.3B (large) parameters.

**Evaluation Domains:** Illustrations, user interfaces, natural images, and documents.

**Paper:** [arXiv:2210.03347](https://arxiv.org/abs/2210.03347)

---

## 4. Retrieval-Augmented Generation for Vision

### 4.1 Multimodal RAG Approaches

Multimodal RAG extends traditional text-based RAG by incorporating multiple modalities (text, images, audio, video) to enhance generated outputs.

**Key Challenges:**
- Cross-modal alignment and reasoning
- Information loss during parsing
- Limited retrieval performance for complex semantic structures

**Recent Frameworks:**
- **MRAMG, M2RAG, M2IO-R1:** Enforce explicit interleaving of images within text
- **Inserter modules (e.g., Inserter-R1-3B):** RL-based sequential image selection and placement
- **mR2AG:** "Reflection" steps for adaptive retrieval necessity judgment

**Survey Papers:**
- "Ask in Any Modality: A Comprehensive Survey on Multimodal RAG" (Feb 2025)
- "A Survey of Multimodal Retrieval-Augmented Generation" (Mar 2025) - [arXiv:2504.08748](https://arxiv.org/abs/2504.08748)
- "Retrieval Augmented Generation and Understanding in Vision: A Survey" (Mar 2025) - [arXiv:2503.18016](https://arxiv.org/html/2503.18016v1)

### 4.2 VisRAG

VisRAG is a vision-based RAG approach for multi-modality documents that maximizes retention of original document information.

**Key Advantage:** Eliminates information loss introduced during traditional text-based parsing.

**Results:** 20-40% end-to-end performance gain over traditional text-based RAG pipelines.

**Paper:** [arXiv:2410.10594](https://arxiv.org/abs/2410.10594)

### 4.3 GUI Agent Retrieval

Several approaches apply retrieval-augmentation specifically to GUI agents:

#### RAG-GUI
Lightweight VLM leveraging web tutorials at inference time, with supervised finetuning warm-start and self-guided rejection sampling finetuning refinement.

**Results:** Outperforms baseline agents by 2.6% to 13.3% across model sizes.

**Paper:** [arXiv:2509.24183](https://arxiv.org/abs/2509.24183)

#### KG-RAG
Enhances GUI agent decision-making via knowledge graphs, providing UTG-derived knowledge graph for efficient navigational knowledge retrieval.

#### AppAgent v2
LLM-based multimodal agent using RAG to efficiently retrieve and update information from a knowledge base during exploration and deployment phases.

#### PAL-UI
Planning with Active Look-back empowers GUI agents to actively retrieve pertinent details from history on demand during planning.

---

## 5. GUI Understanding Models

### 5.1 CogAgent

CogAgent is an 18-billion-parameter VLM specializing in GUI understanding and navigation from Tsinghua University.

**Architecture:**
- 11B visual + 7B language parameters
- Dual image encoders (low and high resolution)
- 1120x1120 input resolution for recognizing tiny page elements

**Performance:**
- State-of-the-art on Mind2Web (PC) and AITW (Android) using only screenshots
- State-of-the-art on 9 cross-modal benchmarks (VQAv2, MM-Vet, POPE, etc.)

**CogAgent-9B (Dec 2024)** features improvements in GUI perception, reasoning accuracy, action space completeness, and generalization.

**Paper:** [arXiv:2312.08914](https://arxiv.org/abs/2312.08914) (CVPR 2024 Highlight, top 3%)

### 5.2 UI-TARS

UI-TARS from ByteDance is a native GUI agent model perceiving screenshots and performing human-like interactions (keyboard/mouse operations).

**Key Innovations:**
1. **Enhanced Perception:** Large-scale GUI screenshot dataset for context-aware understanding
2. **Unified Action Modeling:** Standardized action space across platforms
3. **System-2 Reasoning:** Deliberate reasoning with task decomposition, reflection, milestone recognition
4. **Iterative Training:** Automatic collection, filtering, and reflective refinement of interaction traces

**Performance:**
- OSWorld: 24.6 (50 steps), outperforming Claude (22.0)
- AndroidWorld: 46.6, surpassing GPT-4o (34.5)
- State-of-the-art on 10+ GUI agent benchmarks

**Model Sizes:** 2B, 7B, 72B

**UI-TARS-1.5** integrates advanced reasoning via reinforcement learning.

**Paper:** [arXiv:2501.12326](https://arxiv.org/abs/2501.12326)

### 5.3 Ferret-UI

Ferret-UI from Apple is tailored for mobile UI understanding with referring, grounding, and reasoning capabilities.

**Key Features:**
- "Any resolution" approach dividing screens into sub-images based on aspect ratio
- Handles elongated aspect ratios and small objects (icons, texts)

**Results:** Surpasses GPT-4V on elementary UI tasks.

**Ferret-UI 2 (Oct 2024):**
- Multi-platform support (iPhone, Android, iPad, Webpage, AppleTV)
- Adaptive gridding for high-resolution perception
- GPT-4o powered task training data generation with set-of-mark visual prompting

**Ferret-UI Lite (3B):** Competitive GUI grounding and navigation for on-device deployment.

**Papers:**
- Original: [arXiv:2404.05719](https://arxiv.org/abs/2404.05719) (ECCV 2024)
- Ferret-UI 2: [arXiv:2410.18967](https://arxiv.org/abs/2410.18967)

### 5.4 ScreenAI

ScreenAI from Google DeepMind specializes in UI and infographics understanding.

**Architecture:** PaLI architecture with Pix2Struct's flexible patching strategy.

**Key Contribution:** Novel screen annotation task identifying type and location of UI elements, used to generate QA, navigation, and summarization training data at scale.

**Results (5B parameters):**
- State-of-the-art: Multi-page DocVQA, WebSRC, MoTIF, Widget Captioning
- Best-in-class: ChartQA, DocVQA, InfographicVQA

**Released Datasets:** Screen annotation dataset and two QA-focused datasets.

**Paper:** [arXiv:2402.04615](https://arxiv.org/abs/2402.04615)

---

## 6. Embedding Techniques

### 6.1 Matryoshka Representation Learning

MRL is a flexible representation learning approach encoding information at different granularities within a single embedding.

**Core Concept:** Inspired by Matryoshka dolls, MRL trains models to produce multi-scale representations where smaller truncated embeddings remain useful.

**Benefits:**
- Up to 14x smaller embedding size at same accuracy (ImageNet-1K)
- Up to 14x real-world retrieval speedups
- Up to 2% accuracy improvements for long-tail few-shot classification

**Adoption:** OpenAI text-embedding-3-large, Nomic nomic-embed-text-v1, Alibaba gte-multilingual-base, Qwen3-VL-Embedding.

**Implementation:** Minor modification to architecture via MRL linear layer with nesting list (e.g., [8, 16, 32, 64, 128, 256, 512, 1024, 2048]).

**Paper:** [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) (NeurIPS 2022)

### 6.2 Contrastive Learning and InfoNCE

InfoNCE (Information Noise-Contrastive Estimation) is the cornerstone loss function in contrastive learning.

**Origin:** Introduced in "Representation Learning with Contrastive Predictive Coding" by van den Oord et al. (2018).

**How It Works:**
- Frames unsupervised learning as instance discrimination
- Estimates lower bound on mutual information
- Uses categorical cross-entropy to identify positive samples among noise samples
- Positive samples drawn from conditional distribution p(x|c), negatives from proposal distribution p(x)

**Key Hyperparameter:** Temperature (τ) - smaller values sharpen similarity differences but risk exploding gradients; larger values smooth learning but reduce contrast.

**Variants:** NT-Xent (Normalized Temperature-scaled Cross Entropy) is essentially equivalent to InfoNCE.

**Reference:** [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

### 6.3 Hard Negative Mining

Hard negative mining is crucial for training effective embedding models in contrastive learning settings.

#### NV-Retriever (NVIDIA, 2024)
Introduces positive-aware mining methods using positive relevance scores as anchors for false negative removal.

**Methods:**
- **TopK-MarginPos:** Subtracts margin (0.05) from positive score as maximum threshold
- **TopK-PercPos:** Percentage-based thresholding

**Results:** NV-Retriever-v1 achieves 60.9 on MTEB Retrieval (BEIR), ranked #1 when published.

**Paper:** [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)

#### Conan-Embedding (2024)
Proposes dynamic hard negative mining during training, allowing models to adapt to changing data as weights update.

**Insight:** Hard negatives mined during preprocessing may become less challenging after training iterations.

**Paper:** [arXiv:2408.15710](https://arxiv.org/html/2408.15710v2)

#### Key Finding
Research on MS-MARCO found ~70% of naively mined hard negatives should actually be labeled as positive, highlighting the need for positive-aware mining.

### 6.4 Cross-Encoder Reranking

Cross-encoders process query-document pairs jointly through a transformer, outputting relevance scores (0-1).

**Retrieve-and-Rerank Paradigm:**
1. Fast bi-encoder retrieves many candidates (ensuring recall)
2. Cross-encoder reranks candidates for high precision

**Trade-offs:**
- Higher accuracy than bi-encoders due to cross-attention
- High computational cost (N transformer passes for N candidates)
- Tractable for shortlists (50-200 candidates), impractical for full-scale retrieval

**Recent Advances:**
- mGTE extends context to 8192 tokens with RoPE
- CMC (Comparing Multiple Candidates) framework for jointly comparing multiple candidates

**Qwen3-VL-Reranker:** Cross-encoder architecture with cross-attention mechanisms for fine-grained relevance estimation.

---

## 7. Benchmarks

### 7.1 MMEB / MMEB-V2

**MMEB (Massive Multimodal Embedding Benchmark):**
- 4 meta-tasks: classification, VQA, multimodal retrieval, visual grounding
- 36 datasets (20 training, 16 evaluation)
- Introduced alongside VLM2Vec

**MMEB-V2:**
- 9 meta-tasks, 78 tasks total
- New task types: visual document retrieval, video retrieval, temporal grounding, video classification, video QA
- Extends to videos and visual documents

**Leaderboard:** [TIGER-Lab/MMEB-Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)

**Dataset:** [TIGER-Lab/MMEB-V2](https://huggingface.co/datasets/TIGER-Lab/MMEB-V2)

### 7.2 MTEB / MMTEB

**MTEB (Massive Text Embedding Benchmark):**
- 8 embedding tasks
- 58 datasets, 112 languages
- Published at EACL 2023

**Paper:** [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)

**MMTEB (Massive Multilingual Text Embedding Benchmark):**
- 500+ quality-controlled tasks
- 250+ languages
- Novel tasks: instruction following, long-document retrieval, code retrieval
- 85 co-authors

**Paper:** [arXiv:2502.13595](https://arxiv.org/abs/2502.13595)

**Resources:** [GitHub - embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)

### 7.3 ScreenSpot

ScreenSpot is the first realistic GUI grounding benchmark encompassing mobile, desktop, and web environments.

**Introduced In:** "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents"

**Paper:** [arXiv:2401.10935](https://arxiv.org/abs/2401.10935)

**ScreenSpot-Pro:** Extended benchmark for high-resolution professional settings.
- 23 applications across 5 industries and 3 operating systems
- Best model achieves only 18.9%, highlighting remaining challenges
- ScreenSeekeR cascaded search method achieves 48.1% without additional training

**Paper:** [arXiv:2504.07981](https://arxiv.org/abs/2504.07981)

### 7.4 MoTIF

MoTIF (Mobile app Tasks with Iterative Feedback) addresses task feasibility in interactive visual environments.

**Key Features:**
- Natural language commands for interactive environments
- First to include unsatisfiable requests
- Follow-up questions for task uncertainty resolution

**Initial Results:** Only 37.3 F1 on feasibility classification, verifying need for richer representations.

**Paper:** [arXiv:2104.08560](https://arxiv.org/abs/2104.08560)

### 7.5 AITW

AITW (Android In The Wild) is a large-scale dataset for Android device control with human demonstrations.

**Scale:**
- 715k episodes
- 30k unique instructions
- 4 Android versions (v10-13)
- 8 device types (Pixel 2 XL to Pixel 6)

**Scenarios:** General, Install, GoogleApps, Single, WebShopping

**Challenge:** Actions must be inferred from visual appearance; action space consists of precise gestures.

**Paper:** [arXiv:2307.10088](https://arxiv.org/abs/2307.10088) (NeurIPS 2023)

### 7.6 OSWorld and AndroidWorld

#### OSWorld
First scalable real computer environment for multimodal agents across Ubuntu, Windows, and macOS.

**Benchmark:**
- 369 computer tasks
- Real web and desktop apps
- OS file I/O and multi-application workflows
- Execution-based evaluation

**Results:** Humans achieve 72.36%, best model only 12.24%, primarily struggling with GUI grounding.

**Paper:** [arXiv:2404.07972](https://arxiv.org/abs/2404.07972) (NeurIPS 2024)

#### AndroidWorld
Live Android emulator environment for autonomous agents.

**Benchmark:**
- 116 hand-crafted tasks
- 20 real-world Android apps
- Dynamic task instantiation with random parameters
- Millions of unique task variations

**Results:** Best agent completes 30.6%; MobileRL-9B achieves 80.2%.

**Resources:** [GitHub - google-research/android_world](https://github.com/google-research/android_world)

### 7.7 Mind2Web

Mind2Web is the first dataset for generalist web agents following language instructions.

**Scale:**
- 2,000+ open-ended tasks
- 137 websites, 31 domains
- Crowdsourced action sequences

**MindAct Model:** Two-stage approach (small LM filtering + LLM selection).

**Paper:** NeurIPS'23 Spotlight

**Extensions:**
- **Online-Mind2Web:** 300 tasks, 136 websites for online evaluation
- **Mind2Web 2:** Long-horizon, time-varying agentic search tasks
- **Mind2Web-Live:** Dynamic evaluation in evolving web environments

**Resources:** [osu-nlp-group.github.io/Mind2Web](https://osu-nlp-group.github.io/Mind2Web/)

---

## 8. Qwen3-VL-Embedding Deep Dive

Qwen3-VL-Embedding and Qwen3-VL-Reranker provide an end-to-end pipeline for high-precision multimodal search.

### Architecture

**Base Model:** Qwen3-VL foundation model

**Embedding Model:**
- Multi-stage training paradigm: large-scale contrastive pre-training to reranking model distillation
- Generates semantically rich high-dimensional vectors
- Supports Matryoshka Representation Learning for flexible embedding dimensions
- Handles inputs up to 32k tokens

**Reranker Model:**
- Cross-encoder architecture
- Cross-attention mechanisms for fine-grained relevance estimation

### Model Variants

| Model | Parameters | Context | Use Case |
|-------|------------|---------|----------|
| Qwen3-VL-Embedding-2B | 2B | 32k | Resource-constrained |
| Qwen3-VL-Embedding-8B | 8B | 32k | High-performance |
| Qwen3-VL-Reranker-2B | 2B | 32k | Efficient reranking |
| Qwen3-VL-Reranker-8B | 8B | 32k | Precise reranking |

### Multilingual Support

Inherits Qwen3-VL's multilingual capabilities supporting 30+ languages.

### Benchmark Performance

**MMEB-V2 Results (as of January 8, 2025):**
- Qwen3-VL-Embedding-8B: 77.8 overall score (ranked #1)
- 6.7% improvement over previous best open-source model

### Supported Tasks

- Image-text retrieval
- Visual question answering
- Video-text matching
- Document retrieval
- Cross-modal search

### Key Training Innovations

1. **Multi-stage Training:**
   - Stage 1: Large-scale contrastive pre-training
   - Stage 2: Reranking model distillation

2. **Matryoshka Representation Learning:**
   - Flexible embedding dimensions
   - Efficient retrieval at various precision-speed trade-offs

3. **Cross-Attention Reranking:**
   - Fine-grained query-document pair scoring
   - Improved relevance estimation

**Paper:** [arXiv:2601.04720](https://arxiv.org/abs/2601.04720)

**GitHub:** [QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)

**Models:** [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)

---

## 9. Implications for GUI Automation

### Retrieval-Augmented GUI Agents

The combination of multimodal embeddings and GUI understanding enables several important capabilities:

1. **Screenshot-to-Action Retrieval:** Given a current screenshot, retrieve similar historical screenshots and their associated successful action sequences.

2. **Instruction-to-Screenshot Matching:** Given a natural language task, retrieve relevant UI states that represent progress toward task completion.

3. **Element-Level Retrieval:** Embed individual UI elements (buttons, text fields, icons) for fine-grained grounding and action prediction.

4. **Documentation Retrieval:** Retrieve relevant help documentation, tutorials, or API references based on current screen state.

### Architecture Recommendations

Based on this survey, an effective GUI automation retrieval system should consider:

1. **Embedding Model Selection:**
   - Qwen3-VL-Embedding for state-of-the-art multimodal performance
   - GME for Any2Any search flexibility
   - VLM2Vec for task-instruction-aware embeddings

2. **Two-Stage Retrieval:**
   - Fast bi-encoder for initial candidate retrieval
   - Cross-encoder reranker (e.g., Qwen3-VL-Reranker) for precision

3. **MRL for Efficiency:**
   - Use Matryoshka embeddings for adaptive precision-speed trade-offs
   - Lower dimensions for fast approximate search
   - Full dimensions for final reranking

4. **Hybrid Approaches:**
   - Combine screenshot embeddings with structured UI element data
   - Leverage knowledge graphs (KG-RAG) for navigational knowledge
   - Active look-back (PAL-UI) for dynamic history retrieval

### Open Challenges

1. **High-Resolution Processing:** GUI elements are often small; models need high-resolution support (Qwen3-VL: 1120x1120+, Ferret-UI: any resolution)

2. **Dynamic Content:** GUIs change frequently; embeddings must generalize across UI variations

3. **Long-Horizon Tasks:** Multi-step tasks require maintaining context across many screenshots

4. **Cross-Platform Generalization:** Models should transfer across mobile, desktop, and web

5. **Computational Efficiency:** Real-time agent interaction requires fast embedding and retrieval

---

## 10. References

### Multimodal Embedding Models

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

2. Zhai, X., et al. "Sigmoid Loss for Language Image Pre-Training." 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)

3. Zhai, X., et al. "SigLIP 2: Multilingual Vision-Language Encoders." 2025. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)

4. Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training." ICML 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)

5. Liu, H., et al. "Visual Instruction Tuning." NeurIPS 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)

6. Liu, H., et al. "Improved Baselines with Visual Instruction Tuning." [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)

7. Bai, J., et al. "Qwen-VL: A Versatile Vision-Language Model." 2023. [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)

8. Wang, P., et al. "Qwen2-VL: Enhancing Vision-Language Model's Perception." 2024. [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)

9. Qwen Team. "Qwen2.5-VL Technical Report." 2025. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)

10. Qwen Team. "Qwen3-VL Technical Report." 2025. [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)

11. Jiang, Z., et al. "VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks." ICLR 2025. [arXiv:2410.05160](https://arxiv.org/abs/2410.05160)

12. "VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents." 2025. [arXiv:2507.04590](https://arxiv.org/abs/2507.04590)

13. Jiang, T., et al. "E5-V: Universal Embeddings with Multimodal Large Language Models." 2024. [arXiv:2407.12580](https://arxiv.org/abs/2407.12580)

14. Zhang, X., et al. "GME: Improving Universal Multimodal Retrieval by Multimodal LLMs." 2024. [arXiv:2412.16855](https://arxiv.org/abs/2412.16855)

### Visual Document Understanding

15. Xu, Y., et al. "LayoutLM: Pre-training of Text and Layout for Document Image Understanding." KDD 2020. [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)

16. Xu, Y., et al. "LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding." 2020. [arXiv:2012.14740](https://arxiv.org/abs/2012.14740)

17. Huang, Y., et al. "LayoutLMv3: Pre-training for Document AI." ACM MM 2022. [arXiv:2204.08387](https://arxiv.org/pdf/2204.08387)

18. Xu, Y., et al. "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding." 2021. [arXiv:2104.08836](https://arxiv.org/abs/2104.08836)

19. Appalaraju, S., et al. "DocFormer: End-to-End Transformer for Document Understanding." ICCV 2021. [arXiv:2106.11539](https://arxiv.org/abs/2106.11539)

20. Kim, G., et al. "OCR-free Document Understanding Transformer (Donut)." ECCV 2022. [arXiv:2111.15664](https://arxiv.org/abs/2111.15664)

21. Lee, K., et al. "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding." ICML 2023. [arXiv:2210.03347](https://arxiv.org/abs/2210.03347)

### GUI Understanding Models

22. Hong, W., et al. "CogAgent: A Visual Language Model for GUI Agents." CVPR 2024. [arXiv:2312.08914](https://arxiv.org/abs/2312.08914)

23. Qin, Y., et al. "UI-TARS: Pioneering Automated GUI Interaction with Native Agents." 2025. [arXiv:2501.12326](https://arxiv.org/abs/2501.12326)

24. You, K., et al. "Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs." ECCV 2024. [arXiv:2404.05719](https://arxiv.org/abs/2404.05719)

25. Li, K., et al. "Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms." 2024. [arXiv:2410.18967](https://arxiv.org/abs/2410.18967)

26. Baechler, G., et al. "ScreenAI: A Vision-Language Model for UI and Infographics Understanding." 2024. [arXiv:2402.04615](https://arxiv.org/abs/2402.04615)

### Embedding Techniques

27. Kusupati, A., et al. "Matryoshka Representation Learning." NeurIPS 2022. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)

28. van den Oord, A., et al. "Representation Learning with Contrastive Predictive Coding." 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

29. Moreira, G., et al. "NV-Retriever: Improving text embedding models with effective hard-negative mining." 2024. [arXiv:2407.15831](https://arxiv.org/abs/2407.15831)

30. Li, S., et al. "Conan-embedding: General Text Embedding with More and Better Negative Samples." 2024. [arXiv:2408.15710](https://arxiv.org/html/2408.15710v2)

### Qwen3-VL-Embedding

31. Qwen Team. "Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking." 2025. [arXiv:2601.04720](https://arxiv.org/abs/2601.04720)

### Benchmarks

32. Muennighoff, N., et al. "MTEB: Massive Text Embedding Benchmark." EACL 2023. [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)

33. Enevoldsen, K., et al. "MMTEB: Massive Multilingual Text Embedding Benchmark." 2025. [arXiv:2502.13595](https://arxiv.org/abs/2502.13595)

34. Cheng, J., et al. "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents." 2024. [arXiv:2401.10935](https://arxiv.org/abs/2401.10935)

35. Li, K., et al. "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use." 2025. [arXiv:2504.07981](https://arxiv.org/abs/2504.07981)

36. Burns, A., et al. "Mobile App Tasks with Iterative Feedback (MoTIF)." 2021. [arXiv:2104.08560](https://arxiv.org/abs/2104.08560)

37. Rawles, C., et al. "Android in the Wild: A Large-Scale Dataset for Android Device Control." NeurIPS 2023. [arXiv:2307.10088](https://arxiv.org/abs/2307.10088)

38. Xie, T., et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." NeurIPS 2024. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972)

39. Deng, X., et al. "Mind2Web: Towards a Generalist Agent for the Web." NeurIPS 2023. [osu-nlp-group.github.io/Mind2Web](https://osu-nlp-group.github.io/Mind2Web/)

### RAG for Vision

40. Yu, W., et al. "VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents." 2024. [arXiv:2410.10594](https://arxiv.org/abs/2410.10594)

41. "Retrieval-augmented GUI Agents with Generative Guidelines." 2025. [arXiv:2509.24183](https://arxiv.org/abs/2509.24183)

42. Zaib, M., et al. "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG." 2025. [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)

### Additional Resources

- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
- [LAION CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- [GUI Agents Paper List](https://github.com/OSU-NLP-Group/GUI-Agents-Paper-List)
- [Awesome RAG Vision](https://github.com/zhengxuJosh/Awesome-RAG-Vision)
- [Multimodal RAG Survey](https://github.com/llm-lab-org/Multimodal-RAG-Survey)

---

*This literature review was compiled in January 2026 to support the development of multimodal retrieval systems for GUI automation.*

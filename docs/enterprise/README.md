# Enterprise Architecture Documentation

This directory contains architecture contracts and reference documents for enterprise automation integrations with OpenAdapt.

## Document Classification

### Canonical OpenAdapt Documents
These define the OpenAdapt platform architecture:

| Document | Version | Description |
|----------|---------|-------------|
| [SAC_v0.1.md](SAC_v0.1.md) | v0.1.0 | Shared Architecture Contract - technical interfaces and responsibilities |
| [DESIGN_ROADMAP_v0.1.md](DESIGN_ROADMAP_v0.1.md) | v0.1 | OpenAdapt Design & Roadmap - system architecture and implementation plan |

### Research & Reference Documents
Research-based documents informing OpenAdapt design:

| Document | Description |
|----------|-------------|
| [COORDS_VS_MARKS_ABLATION.md](COORDS_VS_MARKS_ABLATION.md) | Research: coordinate vs element-based approaches |

## Purpose

These documents define:
- System decomposition and data flow
- Canonical data and action contracts
- Learning stages and evaluation criteria
- Execution and safety boundaries
- Delivery timelines and milestones

## Usage

- **Integrators**: Reference SAC for interface contracts
- **Developers**: Reference Design & Roadmap for implementation details
- **Operations**: Reference Enterprise Plan for deployment phases

## Change Policy

Documents follow Semantic Versioning:
- MAJOR: Breaking changes to interfaces
- MINOR: New optional fields/features
- PATCH: Documentation fixes

All changes require explicit rationale and version bump.

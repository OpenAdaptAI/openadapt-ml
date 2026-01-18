# Repository History

Documentation of deprecated and archived OpenAdapt ecosystem projects.

## Deprecated/Archived Projects

### OpenAdapter (Archived January 2026)

**Repository**: https://github.com/OpenAdaptAI/OpenAdapter (ARCHIVED)
**Status**: Incomplete proof-of-concept from before OpenAdapt refactor

**Why Archived**:
- Incomplete proof-of-concept code (only 165 lines, missing imports)
- Created October 2024, minimal activity (14 commits, only 1 contributor)
- Cloud infrastructure now handled by `openadapt_ml/cloud/` module
- No active development, zero ecosystem usage
- Last substantial commit was February 2025 (marked as WIP)

**Original Purpose**:
Attempted to provide cloud deployment infrastructure for screenshot parsing and action models, specifically targeting AWS ECS/ECR deployment for OmniParser using CDKTF (Terraform via Python).

**Key Takeaways & Lessons Learned**:
- Cloud training support is critical for productivity
- Multiple backends (Lambda Labs, Azure) enable flexibility and cost optimization
- Infrastructure as Code (Terraform/CDK) is appropriate for cloud setup
- State management (tracking deployment IPs, configs) is important for multi-region deployments
- Single-provider solutions are fragile - always support multiple cloud backends

**What Replaced It**:
- `openadapt_ml/cloud/lambda_labs.py` - Lambda Labs GPU rental and management
- `openadapt_ml/cloud/azure_inference.py` - Azure ML integration for inference
- `openadapt_ml/benchmarks/azure.py` - Azure ML for automated WAA evaluation
- `scripts/setup_azure.py` - Full Azure setup automation with resource management
- Documentation: `docs/cloud_gpu_training.md`, `docs/azure_waa_setup.md`

**Modern Approach**:
The current openadapt-ml cloud infrastructure is production-ready and supports:
- Multiple cloud providers (Lambda Labs, Azure ML, local)
- Multiple model types (not just OmniParser)
- Automatic cleanup and quota management
- Tested deployment patterns with comprehensive documentation
- Cost estimation and monitoring tools

**References**:
- Original incomplete code: https://github.com/OpenAdaptAI/OpenAdapter/tree/feat/omniparser
- Cloud architecture docs: `docs/cloud_gpu_training.md`
- Azure setup guide: `docs/azure_waa_setup.md`

---

## Notes on Repository Management

**When to Archive**:
- No active development for 3+ months
- Incomplete/experimental code that won't be finished
- Functionality superseded by other ecosystem components
- Zero usage in production or by other repos
- Single contributor with no current interest

**Before Archiving**:
1. Review code for valuable patterns or ideas
2. Document key takeaways in this file
3. Update references in other repositories
4. Remove from GitHub organization profile README
5. Add archive notice to repository description

**Alternative to Archiving**:
- Move code to `legacy/` branch in main repository
- Keep as example/reference in documentation
- Convert to gist or snippet if very small

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SERA (Soft-Verified Efficient Repository Agents) is an LLM data generation pipeline by Allen AI. It extracts functions from codebases, creates synthetic coding tasks, runs LLM agents (SWE-agent or mini-swe-agent) to solve them in Docker containers, evaluates solutions via patch comparison, and formats trajectories into training data.

## Setup

```bash
git clone --recurse-submodules https://github.com/allenai/SERA.git
cd SERA
conda create -n sera python=3.12
conda activate sera
pip install -e . -e modules/code2flow -e modules/SERA-SWE-Agent -e modules/SERA-mini-swe-agent
```

Submodules in `modules/` (SERA-SWE-Agent, SERA-mini-swe-agent, code2flow) must be initialized for the pipeline to work.

## Running the Pipeline

Entry point is `sera/main.py` using Hydra for configuration:

```bash
# Full pipeline on SWE-Bench repos
python sera/main.py --config-name=specialization_django \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=http://HOST:PORT/v1

# Personal repositories
python sera/main.py --config-name=specialization_personal \
    distill.model.name=openai/GLM-4.5-Air \
    distill.model.url=URL

# With Anthropic models (no inference server needed)
python sera/main.py --config-name=specialization_anthropic

# Resume from a specific stage
python sera/main.py --config-name=specialization_django \
    stage=distill_stage_two name=EXISTING_RUN_NAME
```

All config overrides use OmegaConf dot notation. Config files live in `sera/configs/`.

## Architecture

The pipeline runs 5 sequential stages (defined in `sera/main.py:Experiment`):

1. **generate** — Clones repos, parses codebase graphs via code2flow, extracts functions, creates Docker containers
2. **distill_stage_one** — Runs LLM agent rollouts on extracted functions (bug introduction tasks)
3. **distill_stage_two** — Scrapes synthetic PRs from stage one results, runs second rollouts (bug fixing tasks)
4. **eval** — Compares agent patches against gold patches using `compare_patch_threshold`
5. **postprocess** — Formats successful trajectories into training data (hermes/xml/raw tool call formats)

Stages can be individually targeted via `stage=` (e.g., `stage=distill_stage_one`). When resuming, earlier stages are skipped but their metadata is still loaded.

### Key source files

- `sera/config_schema.py` — All configuration dataclasses (`SeraConfig`, `GenerateConfig`, `DistillConfig`, etc.)
- `sera/main.py` — `Experiment` class orchestrates the pipeline stages
- `sera/utils.py` — `ExperimentFolder` manages experiment directory structure, LLM query helpers
- `sera/constants.py` — Large constants file (prompt templates, etc.)
- `sera/datagen/data/generate/` — Codebase parsing (`codebase_parsing.py`), Docker container creation (`docker.py`), function extraction (`generate.py`)
- `sera/datagen/data/distill/distill.py` — Agent rollout orchestration via SWE-agent wrapper
- `sera/datagen/data/eval/eval.py` — Patch evaluation loop
- `sera/datagen/data/postprocess/` — Trajectory formatting for training

### Agent harnesses

Two supported harnesses set via `agent_harness=`:
- `"sweagent"` (default) — Uses SERA-SWE-Agent submodule, configs in `sera/configs/sweagent/e2e.yaml`
- `"mini-swe-agent"` — Uses SERA-mini-swe-agent submodule, configs in `sera/configs/sweagent/mini_e2e.yaml`

### Experiment output structure

```
experiments/<name>/
  configs/    # Copied SWE-agent configs
  data/       # Instances YAML, postprocessed training data
  trajs/      # Raw agent trajectories
```

## Training

Training configs and scripts are in `sera/datagen/train/`. Uses axolotl (primary), unsloth, or llamafactory. After axolotl training, run `convert_axolotl_checkpoint.py` to fix weight name prefixes for vLLM/sgLang compatibility.

## Environment Variables

- `ANTHROPIC_API_KEY` — Required for Anthropic model configs
- Standard OpenAI env vars for local vLLM/sglang servers

## Sharding for Scale

For large runs, shard across multiple inference servers:
```bash
python sera/main.py --config-name=swesmith_scaling \
    distill.shard=0 distill.total_shards=4 \
    distill.model.url=URL_1
```

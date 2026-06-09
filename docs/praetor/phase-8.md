# Phase 8 — Multi-Agent Pipeline

**Goal:** Two agents execute in sequence: research → code, triggered by a single Vikunja task with both labels.

## Pre-conditions

- Phases 5 and 6 complete and stable

## What Gets Built — `praetor/pipelines/`

- `research_then_code.py` — Pydantic Graph with two typed nodes:
  - `ResearchNode`: runs research agent, writes output to `task-{id}` Mem0 namespace
  - `CoderNode`: reads `task-{id}` Mem0 namespace, runs coder agent with that context
- New Hatchet worker executing the pipeline as a DAG (Hatchet has native DAG support)
- The Phase 5 Vikunja webhook adapter already handles this: when a `task.updated` event arrives with BOTH label 14 and label 11 present, the adapter pushes `pipeline:research_code` instead of the individual `agent:research` and `agent:code` events. No new webhook code needed — update the adapter's label routing logic.

**Deploy:** add `pipeline-worker` component to `praetor.toml` → `provision_app("praetor")` → `open_deploy_pr("praetor", "phase-8: pipeline worker")`

## Ready Conditions for Phase 8

1. Create Vikunja task "Implement X feature", label both `ai-research` and `ai-go`
2. Hatchet UI shows pipeline run with two sequential steps: `research` then `code`, both `SUCCEEDED`
3. Opened PR description references findings from the research step
4. If research step fails, code step does not start → Vikunja task updated with failure reason
5. Pipeline re-run skips research if `task-{id}` memory already exists (idempotent research step)

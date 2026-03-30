---
title: AI Gateway Orchestrator
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# AI Gateway Orchestrator (OpenEnv RL Environment)

Welcome to the AI Gateway Orchestrator. This project provides a stateful Reinforcement Learning environment built on OpenEnv. It simulates production-like gateway orchestration where the agent balances latency (SLA), compute cost, and model capability.

The simulation runs for 3,600 steps (1 hour at 1 step per second). The orchestrator must handle bursty traffic, route simple tasks to SLMs and complex tasks to LLMs, scale VM capacity for queue pressure, and leverage spot-price dynamics.

### Problem Statement

Modern AI gateways cannot route blindly. They must orchestrate decisions across:

- SLA and latency risk
- Compute and infrastructure cost
- Model quality and fit (LLM vs SLM)

### Environment Formulation

1. Observation space (state)

- `queue_depth`: requests waiting in queue
- `active_vms`: currently active VM count
- `current_request_type`: `simple` or `complex`
- `current_request_sla`: request deadline step
- `current_spot_price`: current spot compute price
- `price_moving_avg_5m`: trailing 5-minute average (300 steps)
- `queue_velocity`: queue growth/shrink signal

2. Action space

- `ROUTE` (`target = LLM | SLM`): process the request at queue front
- `HOLD`: move current request to back of queue
- `PROVISION` (`lora_id`): provision a VM (cold start applies)
- `TERMINATE` (`instance_id`): terminate an active VM

### Three Tasks and Graders

Task 1: Heuristic routing

- Goal: route `simple -> SLM`, `complex -> LLM`
- Grader intent: reward correct routing and penalize mismatches
- SLA pressure: overdue requests incur per-step penalties and eventual drop penalties

Task 2: Spot-price arbitrage

- Goal: defer non-urgent jobs during expensive windows and execute when cheaper
- Canonical framing for price movement:

$$
P(t) = 0.05 + 0.03\sin(t/50) + \epsilon
$$

- Grader intent: optimize cost-aware scheduling without violating real-time SLAs

Task 3: Infrastructure and cold starts

- Goal: anticipate queue load and provision capacity proactively
- Cold start: provisioned VMs require 120 steps to become active
- Grader intent: penalize under-provisioning and prolonged idle over-provisioning

### Universal Adapter Pattern

The environment uses a global hydration vault (`ACTIVE_SESSIONS`) to support stateless HTTP evaluation while preserving per-episode state between calls.

## What This Environment Simulates

- 3,600-step episodes (1 step = 1 second).
- Bursty request arrivals with two request classes: `simple` and `complex`.
- Real-time and long-SLA requests.
- VM cold starts for infrastructure actions.
- Spot-price variability that affects scheduling strategy.

## Current Action Space

Defined in [my_env/models.py](my_env/models.py):

- `route_request` (`ActionType.ROUTE`)
- `hold_in_queue` (`ActionType.HOLD`)
- `provision_vm` (`ActionType.PROVISION`)
- `terminate_vm` (`ActionType.TERMINATE`)

`MyAction` supports these fields:

- `action_type`
- `req_id`
- `target` (`LLM` or `SLM`)
- `instance_type`
- `lora_id`
- `instance_id`
- `episode_id` (used for session hydration)

## Current Observation Space

Defined in [my_env/models.py](my_env/models.py):

- `episode_id`
- `queue_depth`
- `active_vms`
- `current_spot_price`
- `current_request_id`
- `current_request_type`
- `current_request_sla`
- `queue_velocity`
- `avg_wait_time_ms`
- `price_moving_avg_5m`
- Plus OpenEnv fields such as `reward` and `done`

## Reward and Dynamics (As Implemented)

Behavior below matches [my_env/server/my_env_environment.py](my_env/server/my_env_environment.py):

1. Routing reward
- Correct route (`complex -> LLM`, `simple -> SLM`): `+0.1`
- Wrong route: `-0.5`

2. SLA penalties
- If current step exceeds request SLA deadline: `-0.1` per step
- If request is overdue by more than 15 steps: additional `-2.0` and request is dropped

3. Infrastructure penalties
- If queue is empty and there are active VMs: `-0.05 * active_vms` idle penalty

4. VM cold start
- Provisioned VMs become active after 120 steps

5. Spot price process
- Generated at reset as a bounded random walk starting from `0.05`
- Per-step delta sampled from `[-0.005, 0.005]`
- Lower bounded at `0.01`

## Deterministic Traffic Generation

At reset (with seed), traffic is pre-generated for all 3,600 steps:

- 2% chance of burst (`3..8` requests)
- 20% chance of single request
- Otherwise no request

Each request:

- `type`: `complex` with probability 0.3, else `simple`
- `sla_deadline`: `step + 5` (80% of requests) or `step + 86400` (20%)

## Session Hydration Model

The environment supports stateless serving by storing per-episode data in an in-memory session vault (`ACTIVE_SESSIONS`) and restoring state via `action.episode_id` during `step()` calls.

## Setup

From repository root:

```bash
git clone https://github.com/your-username/ai-gateway-orchestrator.git
cd ai-gateway-orchestrator
cd my_env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

From [my_env](my_env):

```bash
# Validate environment behavior and graders
python demo.py

# Run inference loop (requires .env)
python inference.py

# Run API server
python -m server.app
```

## Environment Variables

Create [my_env/.env](my_env/.env) for inference usage:

```env
API_BASE_URL=http://localhost:8000
HF_TOKEN=your_huggingface_token_here
MODEL_NAME=your_model_name_here
```

## Docker

From [my_env](my_env):

```bash
docker build -t ai-gateway-env .
docker run -p 8000:8000 --env-file .env ai-gateway-env
```

## Running the Demos

This project includes two main scripts:

- [my_env/demo.py](my_env/demo.py): validates core environment physics and grading checks locally
- [my_env/inference.py](my_env/inference.py): runs LLM-driven orchestration loop through configured inference endpoint

Run from [my_env](my_env):

```bash
python demo.py
python inference.py
```

### Sample Evaluation Output (Seed 42)

Representative output from [my_env/demo.py](my_env/demo.py) at random seed = 42:

- Task 1 optimal agent reward around `101.80`
- Task 1 naive baseline reward around `-110.00`
- Task 2 arbitrage reward around `96.80`
- Task 3 infrastructure reward around `96.30`

This demonstrates that the grader is dynamic and strategy-sensitive.

## API Endpoints

When the server is running via [my_env/server/app.py](my_env/server/app.py), the OpenEnv FastAPI app exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

Also available:

- `GET /docs` (OpenAPI/Swagger UI)
- `GET /health` (health check)
- `GET /web` (web interface)

## Deploying to Hugging Face Spaces

The environment manifest is declared in [my_env/openenv.yaml](my_env/openenv.yaml), so deployment can be done directly with OpenEnv tooling.

From [my_env](my_env):

```bash
openenv push
```

Common options:

- `openenv push --repo-id <namespace>/<space-name>`
- `openenv push --private`
- `openenv push --base-image <image>`

After deployment, your Space URL is:

- `https://huggingface.co/spaces/<repo-id>`

## Concurrency Configuration

Current server configuration uses `max_concurrent_envs=1` in [my_env/server/app.py](my_env/server/app.py). Increase this value if you need multiple concurrent WebSocket sessions.

## Project Structure (Current Workspace)

```text
ai_gateway/
├── README.md
└── my_env/
    ├── __init__.py
    ├── client.py
    ├── demo.py
    ├── Dockerfile
    ├── inference.py
    ├── models.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── requirements.txt
    └── server/
        ├── __init__.py
        ├── app.py
        └── my_env_environment.py
```

## Notes

- The canonical environment behavior is implemented in [my_env/server/my_env_environment.py](my_env/server/my_env_environment.py).
- Action and observation schemas are defined in [my_env/models.py](my_env/models.py).

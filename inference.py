"""
OpenEnv Hackathon Inference Script
===================================
Target Environment: AI Gateway Orchestrator
"""

import os
import json
from typing import Dict, Any, Optional, List

from openai import OpenAI
from dotenv import load_dotenv

from server.my_env_environment import MyEnvironment
from models import MyAction, ActionType

load_dotenv(".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "ai_gateway_env")

# Evaluate all 3 required tasks consecutively
TASKS = [
    "heuristic_routing", 
    "spot_price_arbitrage", 
    "infrastructure_scaling"
]

TEMPERATURE = 0.0 
FALLBACK_ACTION = MyAction(action_type=ActionType.HOLD)

MAX_STEPS_PER_TASK = 240
MAX_TOTAL_REWARD = 240.0  
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = """
You are an AI Gateway Orchestrator RL Agent. 
Your goal is to route requests, manage cloud infrastructure, and perform spot-price arbitrage.

Rules:
1. If a request is 'complex', route to 'LLM'. If 'simple', route to 'SLM'.
2. If spot price > moving average, hold batch jobs. Real-time jobs must be routed immediately.
3. If queue_velocity > 3 and active_vms = 0 and booting_vms = 0, provision a VM (g4dn.xlarge, lora: medical-v1).
4. NEVER terminate your last active VM. Only terminate a VM if queue_depth = 0 AND active_vms > 1.

You MUST reply with EXACTLY valid JSON matching this schema. Do not include markdown formatting or explanations.
{
  "action_type": "route_request" | "hold_in_queue" | "provision_vm" | "terminate_vm",
  "req_id": "string or null",
  "target": "LLM or SLM or null",
  "instance_type": "string or null",
  "lora_id": "string or null",
  "instance_id": "string or null"
}
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def parse_llm_response(response_text: str) -> MyAction:
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        action_data = json.loads(clean_text)
        return MyAction(**action_data)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM action: {e}")

def run_task(client: OpenAI, env: MyEnvironment, task_name: str, seed: int) -> None:
    """Executes a single 240-step simulation for a specific task."""
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env.reset(seed=seed)
        
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            action_str = ""
            error = None
            
            # --- SMART BOOTING BYPASS (Saves API calls and prevents TLE) ---
            is_waiting_for_boot = obs.active_vms == 0 and len(env.provisioning_vms) > 0
            
            if is_waiting_for_boot:
                action = FALLBACK_ACTION
                
            elif obs.queue_depth > 0 or obs.queue_velocity > 3 or (obs.queue_depth == 0 and obs.active_vms > 0):
                obs_dict = {
                    "queue_depth": obs.queue_depth,
                    "active_vms": obs.active_vms,
                    "booting_vms": len(env.provisioning_vms),
                    "current_spot_price": round(obs.current_spot_price, 4),
                    "price_moving_avg_5m": round(obs.price_moving_avg_5m, 4),
                    "queue_velocity": obs.queue_velocity,
                    "current_request_id": obs.current_request_id,
                    "current_request_type": obs.current_request_type,
                    "current_request_sla": obs.current_request_sla,
                    "current_step": getattr(env.state, 'step_count', step)
                }

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Current State: {json.dumps(obs_dict)}"}
                        ],
                        response_format={ "type": "json_object" },
                        temperature=TEMPERATURE,
                        seed=seed
                    )
                    response_text = completion.choices[0].message.content or "{}"
                    action = parse_llm_response(response_text)
                    
                    # 🛡️ STRICT GUARDRAILS
                    if action.action_type == ActionType.TERMINATE and obs.active_vms <= 1:
                        action = FALLBACK_ACTION 
                    elif action.action_type == ActionType.ROUTE and obs.active_vms == 0:
                        if len(env.provisioning_vms) == 0:
                            action = MyAction(action_type=ActionType.PROVISION, lora_id="medical-v1")
                        else:
                            action = FALLBACK_ACTION
                            
                except Exception as exc:
                    action = FALLBACK_ACTION
                    error = str(exc).replace("\n", " ") 
            else:
                action = FALLBACK_ACTION

            action_str = f"{action.action_type.value}"
            if getattr(action, "target", None):
                action_str += f"({action.target})"
            elif getattr(action, "lora_id", None):
                action_str += f"({action.lora_id})"

            try:
                obs = env.step(action)
                reward = obs.reward
                done = obs.done
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc).replace("\n", " ")

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        total_reward = sum(rewards)
        raw_score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        
        # ⚠️ CRITICAL FIX: Score must be STRICTLY between (0, 1). 
        # Cannot be exactly 0.0 or exactly 1.0!
        score = min(max(raw_score, 0.01), 0.99)  
        success = score >= SUCCESS_SCORE_THRESHOLD

    except KeyboardInterrupt:
        print(f"\n[DEBUG] Task {task_name} interrupted.", flush=True)
        score = 0.01
        success = False
    except Exception as e:
        print(f"\n[ERROR] Task {task_name} failed: {e}", flush=True)
        score = 0.01
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not API_KEY:
        print("[ERROR]: HF_TOKEN is missing. Cannot initialize client.", flush=True)
        return

    # Implemented Timeout to prevent TLE
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=15.0)
    env = MyEnvironment()
    
    # Run the 3 tasks consecutively
    seeds = [42, 15, 99]
    for i, task in enumerate(TASKS):
        run_task(client, env, task, seeds[i])

if __name__ == "__main__":
    main()
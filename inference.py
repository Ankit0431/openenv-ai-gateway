"""
OpenEnv Hackathon Inference Script
===================================
Target Environment: AI Gateway Orchestrator
"""

import os
import time
import json
from typing import Dict, Any, Optional, List

from openai import OpenAI
from dotenv import load_dotenv

from server.my_env_environment import MyEnvironment
from models import MyAction, ActionType

load_dotenv(".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct") # Put your actual model here
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Evaluation Identifiers
TASK_NAME = os.getenv("MY_ENV_TASK", "gateway_orchestration")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "ai_gateway_env")

TEMPERATURE = 0.0  # Keep at 0.0 for deterministic, reproducible results
FALLBACK_ACTION = MyAction(action_type=ActionType.HOLD)

# Normalization constants for [0, 1] scoring
MAX_STEPS = 360
MAX_TOTAL_REWARD = 360.0  # Approx 1.0 optimal reward * 360 steps
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = """
You are an AI Gateway Orchestrator RL Agent. 
Your goal is to route requests, manage cloud infrastructure, and perform spot-price arbitrage.

Rules:
1. If a request is 'complex', route to 'LLM'. If 'simple', route to 'SLM'.
2. If spot price > moving average, hold batch jobs. Real-time jobs must be routed immediately.
3. If queue_velocity > 3 and active_vms = 0 and booting_vms = 0, provision a VM (g4dn.xlarge, lora: medical-v1).
4. 4. NEVER terminate your last active VM. Only terminate a VM if queue_depth = 0 AND active_vms > 1.

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
    """Safely parse the LLM's JSON text into our strict Pydantic Action model."""
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        action_data = json.loads(clean_text)
        return MyAction(**action_data)
    except Exception as e:
        print(f"Failed to parse LLM action: {e}. Falling back to HOLD.")
        return FALLBACK_ACTION

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY, timeout=10.0)

    env = MyEnvironment()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
            obs = env.reset(seed=42)
            
            for step in range(1, MAX_STEPS + 1):
                action_str = ""
                error = None
                
                # If we have 0 active VMs but one is provisioning, we are in a cold start.
                # Do NOT waste time/API calls asking the LLM. Just HOLD.
                is_waiting_for_boot = obs.active_vms == 0 and len(env.provisioning_vms) > 0
                if is_waiting_for_boot:
                    action = FALLBACK_ACTION

                # To save API costs on 3600 steps, query LLM only on actionable states
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
                            seed=42
                        )
                        response_text = completion.choices[0].message.content or "{}"
                        action = parse_llm_response(response_text)
                        
                        # ==============================================================
                        # PROGRAMMATIC GUARDRAILS
                        # ==============================================================
                        
                        # Guardrail 1: Prevent infrastructure suicide
                        if action.action_type == ActionType.TERMINATE and obs.active_vms <= 1:
                            action = FALLBACK_ACTION # Force it to HOLD instead
                            
                        # Guardrail 2: Prevent routing to dead air (no servers online)
                        elif action.action_type == ActionType.ROUTE and obs.active_vms == 0:
                            # If no servers are booting, force it to PROVISION!
                            if len(env.provisioning_vms) == 0:
                                action = MyAction(action_type=ActionType.PROVISION, lora_id="medical-v1")
                            else:
                                action = FALLBACK_ACTION # Just wait for the boot
                                
                    except Exception as exc:
                        action = FALLBACK_ACTION
                        error = str(exc).replace("\n", " ")  # Scrub newlines
                else:
                    action = FALLBACK_ACTION

                # Format action cleanly for the log (e.g., "route_request(LLM)")
                action_str = f"{action.action_type.value}"
                if getattr(action, "target", None):
                    action_str += f"({action.target})"
                elif getattr(action, "lora_id", None):
                    action_str += f"({action.lora_id})"

                # Execute the action
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

            # Calculate normalized score
            total_reward = sum(rewards)
            score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = min(max(score, 0.0), 1.0)  # Clamp exactly between [0, 1]
            success = score >= SUCCESS_SCORE_THRESHOLD

    except KeyboardInterrupt:
        print("\n[DEBUG] Inference interrupted by user.", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Fatal inference error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
if __name__ == "__main__":
    main()
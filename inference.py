"""
OpenEnv Hackathon Inference Script
===================================
Target Environment: AI Gateway Orchestrator
"""

import os
import time
import json
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

from server.my_env_environment import MyEnvironment
from models import MyAction, ActionType

load_dotenv(".env")

API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    print("WARNING: API_BASE_URL is not set. Defaulting to Hugging Face Router API.")
    API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") 
MODEL_NAME = os.getenv("MODEL_NAME")

TEMPERATURE = 0.0  # Keep at 0.0 for deterministic, reproducible results
FALLBACK_ACTION = MyAction(action_type=ActionType.HOLD)

SYSTEM_PROMPT = """
You are an AI Gateway Orchestrator RL Agent. 
Your goal is to route requests, manage cloud infrastructure, and perform spot-price arbitrage.

Rules:
1. If a request is 'complex', route to 'LLM'. If 'simple', route to 'SLM'.
2. If spot price > moving average, hold batch jobs. Real-time jobs must be routed immediately.
3. If queue_velocity > 3 and active_vms = 0 and booting_vms = 0, provision a VM (g4dn.xlarge, lora: medical-v1).
4. If queue_depth = 0 and active VMs > 0, terminate a VM.

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
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = MyEnvironment()
    print("Initializing Gateway Environment...")
    
    # 42 ensures the traffic/market data is identical for the grader
    obs = env.reset(seed=42) 
    total_reward = 0.0

    try:
        for step in range(env.max_steps):
            
            # To save API costs and massive inference time on 3600 steps, 
            # we only query the LLM if there is an actionable state.
            if obs.queue_depth > 0 or obs.queue_velocity > 3 or (obs.queue_depth == 0 and obs.active_vms > 0):
                
                # Format observation for the LLM
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
                    "current_step": env.state.step_count
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
                        seed=42 # Enforces reproducibility
                    )
                    response_text = completion.choices[0].message.content or "{}"
                    action = parse_llm_response(response_text)
                    print(f"-> Agent chose: {action.action_type.value}")
                    
                except Exception as exc:
                    print(f"API Error at step {step}: {exc}. Using fallback.")
                    action = FALLBACK_ACTION
            else:
                # If queue is completely dead and no VMs need managing, just wait.
                action = FALLBACK_ACTION

            # Execute the action in the environment
            obs = env.step(action)
            total_reward += obs.reward
            
            # Progress Logging
            if step % 500 == 0:
                print(f"Step {step}/{env.max_steps} | Action: {action.action_type.value} | Cumulative Reward: {total_reward:.2f}")

        print("\nEpisode Complete.")
        print(f"Final Inference Reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
    
if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: HF_TOKEN is missing. Cannot initialize client.")
    elif not MODEL_NAME:
        print("ERROR: MODEL_NAME is missing. Cannot initialize client.")
    else:
        main()
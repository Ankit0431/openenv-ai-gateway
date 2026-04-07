import requests
import time
import sys

from models import ActionType

BASE_URL = "https://ankit112358-my-env.hf.space"

def run_judge_simulation(seed=42, max_steps=150):
    print(f"🚀 Starting OpenEnv Mock Judge (Seed: {seed})...")
    
    print("➡️ Sending /reset request to initialize environment...")
    try:
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"seed": seed})
    except requests.exceptions.ConnectionError:
        print("❌ CRITICAL ERROR: Could not connect to the server. Is your Docker container running?")
        sys.exit(1)
        
    if reset_resp.status_code != 200:
        print(f"❌ Reset failed! Server returned {reset_resp.status_code}: {reset_resp.text}")
        sys.exit(1)
        
    data = reset_resp.json()
    
    # Grab the episode ID so we can hydrate the server's vault!
    episode_id = data.get("observation", {}).get("episode_id")
    obs = data.get("observation", data) 
    
    if not episode_id:
        print("❌ Could not find 'episode_id' in the reset response!")
        sys.exit(1)
        
    print(f"✅ Episode started successfully! Session ID: {episode_id}\n")
    
    cumulative_reward = 0.0
    
    print("🤖 Agent taking over. Executing steps...")
    
    for step in range(max_steps):
        queue_depth = obs.get("queue_depth", 0)
        req_type = obs.get("current_request_type")
        active_vms = obs.get("active_vms", 0)
        
        # --- UPDATED HEURISTIC AGENT LOGIC ---
        
        # 1. If we have no servers, we MUST provision one, or we bleed points!
        if active_vms == 0 and step == 0:
            # Notice the .value to make it JSON serializable!
            action = {"action_type": ActionType.PROVISION.value, "lora_id": "default", "episode_id": episode_id}
            
        # 2. Wait for the server to boot up (HOLD everything until active_vms > 0)
        elif active_vms == 0:
             action = {"action_type": ActionType.HOLD.value, "episode_id": episode_id}
             
        # 3. Once servers are online, start routing!
        elif queue_depth > 0:
            if req_type == "complex":
                action = {"action_type": ActionType.ROUTE.value, "target": "LLM", "episode_id": episode_id}
            else:
                action = {"action_type": ActionType.ROUTE.value, "target": "SLM", "episode_id": episode_id}
        else:
            action = {"action_type": ActionType.HOLD.value, "episode_id": episode_id}
            
        # Build the payload with the episode_id injected as a kwarg

        
        step_resp = requests.post(f"{BASE_URL}/step", json={"action": action})
        
        if step_resp.status_code != 200:
            print(f"\n❌ Step {step} failed! HTTP {step_resp.status_code}: {step_resp.text}")
            break
            
        step_data = step_resp.json()
        obs = step_data.get("observation", step_data)
        
        step_reward = step_data.get("reward", obs.get("reward", 0.0))
        done = step_data.get("done", obs.get("done", False))
        
        cumulative_reward += step_reward
        
        # Formatted to handle strings cleanly
        print(f"Step {step:03d} | Action: {action['action_type']:<9} | Target: {action.get('target', 'N/A'):<4} | Queue: {queue_depth:<2} | Reward: {step_reward:+.2f} | Cumulative: {cumulative_reward:+.2f}")
        
        if done:
            print("\n🏁 Environment signaled 'done=True'. Episode finished early.")
            break
            
        time.sleep(0.02)

    print("\n" + "="*50)
    print(f"🎉 JUDGE SIMULATION COMPLETE")
    print(f"🎯 Final Cumulative Reward: {cumulative_reward:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_judge_simulation(seed=42, max_steps=150)
import random
from collections import deque
from typing import List, Dict, Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation, ActionType
except ImportError:
    from models import MyAction, MyObservation, ActionType

ACTIVE_SESSIONS = {}
class MyEnvironment(Environment):
    """
    AI Gateway Orchestrator Simulation.
    Manages LLM/SLM routing, spot price arbitrage, and VM autoscaling.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 3600
        self.vm_cold_start_steps = 120
        self.hourly_budget = 50.0  # Example budget
        self.queue = deque()
        self.active_vms = 0
        self.provisioning_vms = []
        self.step_count = 0
        self.spot_prices = []

    def reset(self, seed: int = 42, episode_id: str = None) -> MyObservation: # type: ignore
            """Initialize the episode deterministically based on a seed."""
            actual_episode_id = episode_id if episode_id else str(uuid4())
            self._state = State(episode_id=actual_episode_id, step_count=0)
            random.seed(seed)
            
            # 1. Pre-generate Market Volatility (Spot Prices)
            # Simulating a volatile market using a simple random walk
            self.spot_prices = []
            current_price = 0.05
            for _ in range(self.max_steps):
                current_price += random.uniform(-0.005, 0.005)
                self.spot_prices.append(max(0.01, current_price)) # Price cannot be negative
                
            # 2. Pre-generate Traffic Bursts
            # We will store a list of incoming requests for every single second (timestep)
            # 2. Pre-generate Traffic Bursts
            self.traffic_schedule = [[] for _ in range(self.max_steps)]
            for step in range(self.max_steps):
                # Survivable traffic: 2% chance of burst, 20% chance of 1 req, 78% empty
                r = random.random()
                if r < 0.02:
                    num_requests = random.randint(3, 8)
                elif r < 0.22:
                    num_requests = 1
                else:
                    num_requests = 0
                
                for _ in range(num_requests):
                    self.traffic_schedule[step].append({
                        "id": str(uuid4())[:8],
                        "type": "complex" if random.random() < 0.3 else "simple",
                        "sla_deadline": step + (5 if random.random() < 0.8 else 86400), # Real-time SLA = 5 steps
                        "wait_time": 0
                    })

            # 3. Initialize Live State Variables
            self.queue = deque()
            self.queue_history = deque(maxlen=10) # For calculating queue_velocity
            self.active_vms = [] 
            self.provisioning_vms = [] # Tracks VMs in the 120-step cold start phase
            self.price_history_5m = deque(maxlen=300) # 5 minutes = 300 steps
            self.step_count = 0
            ACTIVE_SESSIONS[actual_episode_id] = {
                "queue": self.queue,
                "active_vms": self.active_vms,
                "provisioning_vms": self.provisioning_vms,
                "traffic_schedule": self.traffic_schedule,
                "spot_prices": self.spot_prices,
                "queue_history": self.queue_history,
                "price_history_5m": self.price_history_5m,
                "step_count": self.step_count
            }
            #DEBUG
            # print("Saved initial state to vault for episode_id:", actual_episode_id)
            return self._get_observation(reward=0.0, done=False)

    def step(self, action: MyAction) -> MyObservation: #type: ignore
        """Execute one timestep (1 simulated second) of the environment."""
        # 1. HYDRATE: Wake up the dead object using the vault
        #DEBUG
        # print(action)
        # print("Active Sessions episodes")
        # print(list(ACTIVE_SESSIONS.keys()))
        episode_id = action.episode_id

        if episode_id and episode_id in ACTIVE_SESSIONS:
            # print("Hydrating state from vault for episode_id:", episode_id)
            session = ACTIVE_SESSIONS[episode_id]
            self.queue = session["queue"]
            self.active_vms = session["active_vms"]
            self.provisioning_vms = session["provisioning_vms"]
            self.traffic_schedule = session["traffic_schedule"]
            self.spot_prices = session["spot_prices"]
            self.queue_history = session["queue_history"]
            self.price_history_5m = session["price_history_5m"]
            self.step_count = session["step_count"]
            self._state.step_count = self.step_count
        else:
            self.step_count = getattr(self, 'step_count', 0)
            
        reward = 0.0
        
        # 1. Process the Agent's Action
        if action.action_type == ActionType.ROUTE:
            # Task 1 & 3 Logic: Check if routed correctly and apply dense rewards
            # (Simplified logic: assuming agent gets +0.1 for acting on a request)
            if self.queue:
                req = self.queue.popleft() # Route the front request
                if action.target == "LLM" and req["type"] == "complex":
                    reward += 0.1
                elif action.target == "SLM" and req["type"] == "simple":
                    reward += 0.1
                else:
                    reward -= 0.5 # Penalty for hallucination/failure
                    
        elif action.action_type == ActionType.HOLD:
            if self.queue:
                # Move held request to the back so others can be processed
                req = self.queue.popleft()
                self.queue.append(req)
                    
        elif action.action_type == ActionType.PROVISION:
            # Task 3 Logic: Start the 120-step timer for a new VM
            self.provisioning_vms.append({
                "id": str(uuid4())[:8],
                "lora_id": action.lora_id,
                "steps_remaining": self.vm_cold_start_steps
            })
            
        elif action.action_type == ActionType.TERMINATE:
            # Task 3 Logic: Terminate an active VM to save money
            if action.instance_id:
                self.active_vms = [vm for vm in self.active_vms if vm["id"] != action.instance_id]
            elif self.active_vms:
                self.active_vms.pop(0) # Terminate the oldest active VM

        # 2. Advance the Simulation State (Physics Engine)
        self._state.step_count += 1
        price_index = min(self._state.step_count, self.max_steps - 1)
        current_spot_price = self.spot_prices[price_index]
        self.price_history_5m.append(current_spot_price)
        
        # Bring new traffic into the queue from the deterministic schedule
        if self._state.step_count < self.max_steps:
            new_traffic = self.traffic_schedule[self._state.step_count]
            self.queue.extend(new_traffic)
            
        # Update VM cold starts
        for vm in list(self.provisioning_vms):
            vm["steps_remaining"] -= 1
            if vm["steps_remaining"] <= 0:
                self.active_vms.append(vm)
                self.provisioning_vms.remove(vm)

        # 3. Calculate Passive Penalties
        # Task 2: SLA Violations (-0.1 per step over SLA)
        active_queue = deque()
        for req in self.queue:
            req["wait_time"] += 1
            
            # Penalty for missing SLA
            if self._state.step_count > req["sla_deadline"]:
                reward -= 0.1
                
        # Task 3: Idle VM Penalty (-0.05 per idle VM)
            if self._state.step_count > req["sla_deadline"] + 15:
                reward -= 2.0 # Heavy drop penalty
            else:
                active_queue.append(req)

        self.queue = active_queue
        if len(self.queue) == 0 and len(self.active_vms) > 0:
            reward -= (0.05 * len(self.active_vms))

        # Make sure step count increments
        self.step_count = getattr(self, 'step_count', 0) + 1 
        self._state.step_count = self.step_count

        # 3. DEHYDRATE: Save the modified variables back to the vault before the server kills this object!
        if episode_id:
            ACTIVE_SESSIONS[episode_id] = {
                "queue": self.queue,
                "active_vms": self.active_vms,
                "provisioning_vms": self.provisioning_vms,
                "traffic_schedule": self.traffic_schedule,
                "spot_prices": self.spot_prices,
                "queue_history": self.queue_history,
                "price_history_5m": self.price_history_5m,
                "step_count": self.step_count
            }
        # 4. Check Termination Condition
        done = self._state.step_count >= (self.max_steps - 1)
        
        return self._get_observation(reward, done)
    
    def _get_observation(self, reward: float, done: bool) -> MyObservation:
        """Helper to construct the observation state."""
        current_price = self.spot_prices[self._state.step_count] if self._state.step_count < self.max_steps else 0.0
        avg_price_5m = sum(self.price_history_5m) / len(self.price_history_5m) if self.price_history_5m else current_price
        
        self.queue_history.append(len(self.queue))
        queue_velocity = self.queue_history[-1] - self.queue_history[0] if len(self.queue_history) > 1 else 0.0
        
        avg_wait = sum(r["wait_time"] for r in self.queue) / len(self.queue) if self.queue else 0.0
        if len(self.queue) > 0:
            current_request_id = self.queue[0]["id"]
            current_request_type = self.queue[0]["type"]
            current_request_sla = self.queue[0]["sla_deadline"]
        else:
            current_request_id = None
            current_request_type = None
            current_request_sla = None

        return MyObservation(
            episode_id=self._state.episode_id,
            queue_depth=len(self.queue),
            active_vms=len(self.active_vms),
            current_request_id=current_request_id,
            current_request_type=current_request_type,
            current_request_sla=current_request_sla,
            current_spot_price=current_price,
            queue_velocity=queue_velocity,
            avg_wait_time_ms=avg_wait,
            price_moving_avg_5m=avg_price_5m,
            reward=reward,
            done=done
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

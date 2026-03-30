from openenv.core.env_server.types import Action, Observation
from pydantic import Field

from enum import Enum
from typing import Optional

class ActionType(str, Enum):
    ROUTE = "route_request"
    HOLD = "hold_in_queue"
    PROVISION = "provision_vm"
    TERMINATE = "terminate_vm"

class MyAction(Action):
    """Action space defining the infrastructure and routing commands."""
    
    action_type: ActionType = Field(
        ..., 
        description="The specific action to take: route_request, hold_in_queue, provision_vm, or terminate_vm."
    )
    
    # Parameters for routing/holding
    req_id: Optional[str] = Field(default=None, description="Target request ID for routing or holding.")
    target: Optional[str] = Field(default=None, description="Target model tier (e.g., 'LLM' or 'SLM').")
    
    # Parameters for infrastructure
    instance_type: Optional[str] = Field(default=None, description="VM type to provision.")
    lora_id: Optional[str] = Field(default=None, description="Specific LoRA adapter ID to load on the VM.")
    instance_id: Optional[str] = Field(default=None, description="ID of the VM to terminate.")

    # Episode id for state mamagement
    episode_id: Optional[str] = Field(default=None, description="Unique identifier for the current episode.")

class MyObservation(Observation):
    """Observation space for the AI Gateway Orchestrator."""
    
    #Episode ID for tracking
    episode_id: str = Field(default="", description="Unique identifier for the current episode.")
    # Core State
    queue_depth: int = Field(default=0, description="Current number of requests waiting.")
    active_vms: int = Field(default=0, description="Number of currently active backend SLM VMs.")
    current_spot_price: float = Field(default=0.0, description="Current spot price of the external LLM API.")
    # These fields are required for Task 1 to evaluate routing decisions, but can be None if the queue is empty
    current_request_id: Optional[str] = Field(default=None, description="ID of the request at the front of the queue.")
    current_request_type: Optional[str] = Field(default=None, description="Type of the request: 'complex' or 'simple'.")
    current_request_sla: Optional[int] = Field(default=None, description="Absolute step deadline for the request.")
    # Predictive Signals
    queue_velocity: float = Field(default=0.0, description="Change in queue depth over the last 10 steps.")
    avg_wait_time_ms: float = Field(default=0.0, description="Average wait time of requests in the queue in ms.")
    price_moving_avg_5m: float = Field(default=0.0, description="5-minute moving average of the spot price.")

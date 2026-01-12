import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from model import *
from utils import to_birds_eye

class PPOAgent:
    def __init__(self, num_agents, map_name, steps, params, transfer=[None, None]):
        # --- Hyperparameters ---
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 3e-4  # Standard PPO learning rate for initial learning
        self.lr_critic = 1e-3  # Critic learns faster to provide better value estimates
        self.gamma = 0.99  # Discount factor for future rewards
        self.gae_lambda = 0.95  # Higher lambda for better credit assignment
        self.clip_epsilon = 0.2  # Standard PPO clipping
        self.state_dim = 3 # x_vel, y_vel, z_ang_vel
        self.num_scan_beams = 1080
        self.lidar_fov = 4.7  # Radians
        self.image_size = 256
        self.minibatch_size = 512
        self.epochs = 10  # Standard PPO epochs for learning
        self.epochs_with_demos = 10
        self.bc_epochs = 3  # More BC epochs to learn from demonstrations
        self.params = params
        
        # --- Demonstration Retention ---
        self.demo_buffer = None  # Store demonstrations for continual learning
        
        # Actor BC weights (action imitation)
        self.demo_bc_actor_weight_initial = 0.50
        self.demo_bc_actor_weight_final = 0.1
        
        # Critic BC weights (value estimation) - separate schedule
        self.demo_bc_critic_weight_initial = 0.5
        self.demo_bc_critic_weight_final = 0.0
        
        self.demo_bc_decay_gens = 400  # Generations to decay BC weight (slower decay)
        self.demo_pretrain_generation = 0  # Track when pretraining occurred
        
        # --- Waypoints for Raceline Reward ---
        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents) 
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        self.start_s = np.zeros(self.num_agents)
        self.current_lap_count = np.zeros(self.num_agents, dtype=int)
        self.last_checkpoint = np.zeros(self.num_agents, dtype=int)  # Track last checkpoint (0-9)
        
        # --- Reward Scalars ---
        self.PROGRESS_REWARD_SCALAR = 32.0
        self.LAP_REWARD = 80.0
        self.CHECKPOINT_REWARD = self.LAP_REWARD * 0.1  # 10% of lap reward per checkpoint
        self.COLLISION_PENALTY = -40.0
       
       
        # --- Networks & Wrappers ---
        
        # Separate encoders for actor and critic to prevent gradient conflicts
        actor_encoder = self._transfer_vision(transfer[0])
        critic_encoder = self._transfer_vision(transfer[1])  # Independent encoder

        # Create the recurrent networks with larger capacity
        self.actor_network = RecurrentActorNetwork(
            self.state_dim, 2, 
            encoder=actor_encoder,
            lstm_hidden_size=256,
            memory_length=20,
            memory_stride=5
        ).to(self.device)
        
        self.critic_network = RecurrentCriticNetwork(
            self.state_dim, 
            encoder=critic_encoder,
            lstm_hidden_size=512,
            memory_length=20,
            memory_stride=5
        ).to(self.device)
        
        self.actor_network = self._transfer_weights(transfer[0], self.actor_network)
        self.critic_network = self._transfer_weights(transfer[1], self.critic_network)
        
        self.actor_lstm_buffer = [self.actor_network.create_observation_buffer(num_agents, self.device)]
        self.actor_hidden = [self.actor_network.get_init_hidden(num_agents, self.device, transpose=True)]
        self.critic_lstm_buffer = [self.critic_network.create_observation_buffer(num_agents, self.device)]
        self.critic_hidden = [self.critic_network.get_init_hidden(num_agents, self.device, transpose=True)]

        # Direct TensorDict modules (no wrappers - networks handle LSTM internally)
        self.actor_module = ProbabilisticActor(
            module=TensorDictModule(
                self.actor_network,
                in_keys=["observation_scan", "observation_state", "actor_lstm_buffer", "actor_lstm_hidden_h", "actor_lstm_hidden_c"],
                out_keys=["loc", "scale", "actor_lstm_buffer_new", "actor_lstm_hidden_h", "actor_lstm_hidden_c"]
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            out_keys=["action"],
            return_log_prob=True
        )
        
        self.critic_module = TensorDictModule(
            self.critic_network,
            in_keys=["observation_scan", "observation_state", "critic_lstm_buffer", "critic_lstm_hidden_h", "critic_lstm_hidden_c"],
            out_keys=["state_value", "critic_lstm_buffer_new", "critic_lstm_hidden_h", "critic_lstm_hidden_c"]
        )
        
        # --- Optimizers ---
        self.actor_optimizer = optim.AdamW(self.actor_module.parameters(), lr=self.lr_actor, weight_decay=0.01)
        self.critic_optimizer = optim.AdamW(self.critic_module.parameters(), lr=self.lr_critic, weight_decay=0.01)
        
        # --- Loss Modules ---
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor_module,
            critic_network=None,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=0.005,
            normalize_advantage=False,
            critic_coeff=0.0,
            clip_value=False,
            separate_losses=True,
            reduction="mean"
        )
        
        self.loss_module.set_keys(
            sample_log_prob="action_log_prob",
            advantage="advantage",  # Explicitly set advantage key
        )

        # --- Storage ---
        self.buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=steps) # 2048 steps per generation
        )
        
        # --- Diagnostics ---
        self.plot_save_path = "plots/training_diagnostics_history.png"
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Initialize storage for historical averages
        self.diagnostic_keys = ["loss_objective", "loss_entropy", "loss_critic",
                                "entropy", "kl_approx", "clip_fraction", "collisions", "reward"]
        self.diagnostics_history = {key: [] for key in self.diagnostic_keys}
        self.generation_counter = 0 # Track generation for x-axis
        
    def _transfer_weights(self, path, network):

        if path is None:
            return network.to(self.device)

        checkpoint = torch.load(path)
        
        prefix = "0.module."

        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                state_dict[new_key] = v
            else: state_dict[k] = v

        if state_dict:
            network.load_state_dict(state_dict)
            print("Successfully loaded pre-trained weights!")
            
        return network.to(self.device)

    def _transfer_vision(self, path):
        new_encoder = VisionEncoder(self.num_scan_beams)
        if path is None:
            return new_encoder.to(self.device)
        
        checkpoint = torch.load(path)
        prefix = "conv_layers."

        encoder_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                encoder_state_dict[new_key] = v
            elif k.startswith("0.module." + prefix):
                new_key = k[len("0.module." + prefix):]
                encoder_state_dict[new_key] = v

        if encoder_state_dict:
            new_encoder.load_state_dict(encoder_state_dict)
            print("Successfully loaded pre-trained encoder weights!")
        else:
            print(checkpoint.keys())
            print(f"Warning: No weights found with prefix '{prefix}'. Starting with a random encoder.")

        return new_encoder.to(self.device)

    def _map_range(self, value, in_min, in_max, out_min=-1, out_max=1):
        if in_max == in_min:
            return out_min if value <= in_min else out_max

        return out_min + (float(value - in_min) / float(in_max - in_min)) * (out_max - out_min)

    def _load_waypoints(self, map_name):
        """
        Loads waypoints from a CSV file for the given map.
        """
        waypoint_file = f"maps/{map_name}/{map_name}_raceline.csv"
        waypoints = np.loadtxt(waypoint_file, delimiter=';')
        waypoints_xy = waypoints[:, 1:3]
        
        # 2. Calculate Cumulative Distance (s)
        positions = waypoints[:, 1:3]
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        waypoints_s = np.insert(np.cumsum(distances), 0, 0)
        raceline_length = waypoints_s[-1]

        return waypoints_xy, waypoints_s, raceline_length

    def _obs_to_tensors(self, obs):
        scans = obs['scans'][:self.num_agents]
        scan_tensors = torch.from_numpy(np.array(scans, dtype=np.float64)).float().to(self.device)
        
        # scan_tensors = to_birds_eye(
        #     scan_tensors.flatten(1),
        #     num_beams=self.num_scan_beams,
        #     fov=self.lidar_fov,
        #     image_size=self.image_size
        # )
        scan_tensors = scan_tensors.unsqueeze(1).to(self.device)
        
        state_data = np.stack(
            (obs['linear_vels_x'], obs['linear_vels_y'], obs['ang_vels_z']), 
            axis=1
        )
        state_tensor = torch.from_numpy(state_data).float().to(self.device)[:self.num_agents]
        
        # Fixed normalization using known physical ranges
        # vx: [-5, 20] -> normalize to ~[-1, 1]
        # vy: assume similar range to vx for lateral velocity
        # omega: assume ~[-3.2, 3.2] from sv_max (steering velocity)
        state_ranges = torch.tensor([
            [self.params['v_min'], self.params['v_max']],    # vx range
            [self.params['v_min'], self.params['v_max']],    # vy range
            [self.params['sv_min'], self.params['sv_max']]      # omega range
        ], device=self.device)
        
        state_min = state_ranges[:, 0]
        state_max = state_ranges[:, 1]
        state_center = (state_max + state_min) / 2
        state_scale = (state_max - state_min) / 2
        
        # Normalize to [-1, 1] range
        state_tensor = (state_tensor - state_center) / state_scale
        # state_tensor = torch.clamp(state_tensor, -1.0, 1.0)  # Clip to ensure within bounds
        
        return scan_tensors, state_tensor

    def get_action_and_value(self, scan_tensor, state_tensor, deterministic=False, store=True):
        """
        Gets an action from the Actor and a value from the Critic.
        Simple LSTM networks - no persistent state management needed.
        """
        self.actor_network.eval()
        self.critic_network.eval()
            
        with torch.no_grad():
            # Get action from actor
            loc, scale, actor_new_buffer, actor_new_hidden_h, actor_new_hidden_c = self.actor_network(
                scan_tensor[:self.num_agents],
                state_tensor[:self.num_agents],
                self.actor_lstm_buffer[-1],
                self.actor_hidden[-1][0],
                self.actor_hidden[-1][1]
            )

            # Create distribution and sample action
            dist = TanhNormal(loc, scale)
            if deterministic:
                action = loc  # Use mean for deterministic policy
            else:
                action = dist.rsample()  # Sample for stochastic policy
            
            log_prob = dist.log_prob(action)
            
            # Get value from critic
            value, critic_new_buffer, critic_new_hidden_h, critic_new_hidden_c = self.critic_network(
                scan_tensor,
                state_tensor,
                self.critic_lstm_buffer[-1],
                self.critic_hidden[-1][0],
                self.critic_hidden[-1][1]
            )

            if store:
                self.actor_lstm_buffer.append(actor_new_buffer)
                self.actor_hidden.append((actor_new_hidden_h, actor_new_hidden_c))
                self.critic_lstm_buffer.append(critic_new_buffer)
                self.critic_hidden.append((critic_new_hidden_h, critic_new_hidden_c))
            
            # Scale to environment's action space
            steer_scale = (self.params['s_max'] - self.params['s_min']) / 2
            steer_shift = (self.params['s_max'] + self.params['s_min']) / 2
            speed_scale = (self.params['v_max'] - self.params['v_min']) / 2
            speed_shift = (self.params['v_max'] + self.params['v_min']) / 2
            
            steering = steer_scale * action[..., 0].unsqueeze(-1) + steer_shift
            speed = speed_scale * action[..., 1].unsqueeze(-1) + speed_shift
            
            # Clamp to valid ranges
            # steering = torch.clamp(steering, min=self.params['s_min'], max=self.params['s_max'])
            # speed = torch.clamp(speed, min=self.params['v_min'], max=self.params['v_max'])
            
            scaled_action = torch.cat((steering, speed), dim=-1)
            
        return action, log_prob, value, scaled_action
    
    def _compute_gae(self, data: TensorDict) -> TensorDict:
        """
        Manually compute GAE advantages and value targets using stored critic outputs.
        Avoids re-running critic through vmap (which fails with LSTM).
        
        Expects data shape: [time_steps, num_agents, ...]
        Computes GAE separately for each agent to maintain temporal dependencies.
        """
        rewards = data.get(("next", "reward"))
        dones = data.get(("next", "done")).float()
        values = data.get("state_value")
        next_values = data.get(("next", "state_value"))
            
        # Squeeze extra dimensions if present: [time_steps, num_agents, 1] -> [time_steps, num_agents]
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)
        
        # Handle different shapes
        if rewards.ndim == 1:
            # Single timestep or single agent: [time_steps] -> [time_steps, 1]
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            values = values.unsqueeze(-1)
            next_values = next_values.unsqueeze(-1)
            timesteps = rewards.shape[0]
            num_agents = 1
        elif rewards.ndim == 2:
            # Multi-agent case: [time_steps, num_agents]
            timesteps = rewards.shape[0]
            num_agents = rewards.shape[1]
        else:
            raise ValueError(f"Unexpected rewards shape after squeezing: {rewards.shape}")
        
        print(f"Computing GAE for {timesteps} timesteps across {num_agents} agents")
        
        advantages = torch.zeros_like(values)
        
        # Compute GAE separately for each agent
        for agent_idx in range(num_agents):
            gae = 0.0
            # Compute backwards through time for this specific agent
            for t in reversed(range(timesteps)):
                delta = rewards[t, agent_idx] + self.gamma * next_values[t, agent_idx] * (1 - dones[t, agent_idx]) - values[t, agent_idx]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t, agent_idx]) * gae
                advantages[t, agent_idx] = gae
        
        returns = advantages + values
        
        # Don't clip returns - let critic learn true value scale
        # Gradient clipping in optimizer will prevent explosion
        
        # Manually normalize advantages (mean=0, std=1) for stable training
        # This is critical since ClipPPOLoss can't normalize without a critic network
        advantages_flat = advantages.flatten()
        advantages_mean = advantages_flat.mean()
        advantages_std = advantages_flat.std() + 1e-8  # Add epsilon to prevent division by zero
        advantages = (advantages - advantages_mean) / advantages_std
        
        # Squeeze if single agent
        if num_agents == 1 and advantages.shape[-1] == 1:
            advantages = advantages.squeeze(-1)
            returns = returns.squeeze(-1)
        
        data.set("advantage", advantages)
        data.set("value_target", returns)
        return data
    
    def reset_lstm_states(self, agent_indices=None):
        if agent_indices is None:
            # Reset all
            self.actor_lstm_buffer = [self.actor_network.create_observation_buffer(self.num_agents, self.device)]
            self.actor_hidden = [self.actor_network.get_init_hidden(self.num_agents, self.device, transpose=True)]
            self.critic_lstm_buffer = [self.critic_network.create_observation_buffer(self.num_agents, self.device)]
            self.critic_hidden = [self.critic_network.get_init_hidden(self.num_agents, self.device, transpose=True)]
        else:
            # Reset specific agents
            if self.actor_lstm_buffer[-1] is not None:
                for idx in agent_indices:
                    self.actor_lstm_buffer[-1][idx] = 0.0
                    self.critic_lstm_buffer[-1][idx] = 0.0
                    if self.actor_hidden[-1] is not None:
                        self.actor_hidden[-1][0][idx, :, :] = 0.0  # h
                        self.actor_hidden[-1][1][idx, :, :] = 0.0  # c
                    if self.critic_hidden[-1] is not None:
                        self.critic_hidden[-1][0][idx, :, :] = 0.0  # h
                        self.critic_hidden[-1][1][idx, :, :] = 0.0  # c
    def store_transition(self, obs, next, action, log_prob, reward, done, value):
        """
        Stores a single step of experience for ALL agents.
        This is a bit complex as we must convert from "list of obs" to "batch."
        """

        # Prepare data for TensorDict (now uses normalized observations)
        scans, states = self._obs_to_tensors(obs)
        next_scans, next_states = self._obs_to_tensors(next)
        
        _, _, next_value, _ = self.get_action_and_value(
            next_scans, next_states, self.params, store=False
        )
        
        # `reward` and `done` need to be converted
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device).unsqueeze(-1)
        done_tensor = torch.tensor(done, dtype=torch.bool).to(self.device).unsqueeze(-1)

        # This dict contains a *batch* of experiences (one for each agent)
        step_data = TensorDict({
            "observation_scan": scans,
            "observation_state": states,
            "action": action,
            "action_log_prob": log_prob,
            "state_value": value,
            "actor_lstm_buffer": self.actor_lstm_buffer[-2].to(self.device),
            "actor_lstm_hidden_h": self.actor_hidden[-2][0].to(self.device),
            "actor_lstm_hidden_c": self.actor_hidden[-2][1].to(self.device),
            "critic_lstm_buffer": self.critic_lstm_buffer[-2].to(self.device),
            "critic_lstm_hidden_h": self.critic_hidden[-2][0].to(self.device),
            "critic_lstm_hidden_c": self.critic_hidden[-2][1].to(self.device),
            "next": TensorDict({
                "observation_scan": next_scans,
                "observation_state": next_states,
                "state_value": next_value,
                "reward": reward_tensor,
                "done": done_tensor,
                "actor_lstm_buffer": self.actor_lstm_buffer[-1].to(self.device),
                "actor_lstm_hidden_h": self.actor_hidden[-1][0].to(self.device),
                "actor_lstm_hidden_c": self.actor_hidden[-1][1].to(self.device),
                "critic_lstm_buffer": self.critic_lstm_buffer[-1].to(self.device),
                "critic_lstm_hidden_h": self.critic_hidden[-1][0].to(self.device),
                "critic_lstm_hidden_c": self.critic_hidden[-1][1].to(self.device)
            }, batch_size=[self.num_agents]).to(self.device)
        }, batch_size=[self.num_agents])
        
        # Add the whole batch to the buffer
        self.buffer.add(step_data.cpu())
    
    def _project_to_raceline(self, current_pos, start_idx, lookahead):
        """
        Projects the agent's current position onto the raceline segment defined
        by the search window to get the most accurate, continuous s-distance.
        
        Returns: projected_s (float), global_wp_index (int)
        """
        wp_count = len(self.waypoints_xy)
        
        # Create a wrapped search slice for the waypoints
        search_indices = np.arange(start_idx, start_idx + lookahead) % wp_count
        search_waypoints = self.waypoints_xy[search_indices]
        
        # Find the closest waypoint (W_curr) within the lookahead window
        distances_in_window = np.linalg.norm(search_waypoints - current_pos, axis=1)
        closest_wp_in_window = np.argmin(distances_in_window)
        
        # Map the local index back to the global index (Index C)
        closest_wp_index_global = search_indices[closest_wp_in_window]
        
        # Define the segment W_prev -> W_curr for projection
        W_curr = self.waypoints_xy[closest_wp_index_global]
        W_prev_index = (closest_wp_index_global - 1 + wp_count) % wp_count
        W_prev = self.waypoints_xy[W_prev_index]
        
        # Vector V: Segment direction (W_prev -> W_curr)
        V = W_curr - W_prev
        V_len_sq = np.dot(V, V)
        
        # Vector W: Vector from W_prev to Agent's Pos
        W = current_pos - W_prev
        
        # Calculate projection length (L) of W onto V. L is a scalar.
        if V_len_sq > 1e-6:
            L = np.dot(W, V) / V_len_sq
        else:
            L = 0.0

        # Clamp L to ensure the projected point P' is within the segment [0, 1]
        # L_clamped = np.clip(L, 0.0, 1.0) 
        
        # Calculate the true continuous s-value
        s_prev = self.waypoints_s[W_prev_index]
        s_curr = self.waypoints_s[closest_wp_index_global]
        
        segment_distance = s_curr - s_prev
        
        # Handle the lap wrap-around condition where s_curr is near 0 and s_prev is near max_length
        if segment_distance < 0:
            segment_distance += self.raceline_length
        
        # Projected S value: s(P') = s(W_prev) + L_clamped * segment_distance
        projected_s = s_prev + L * segment_distance
        
        return projected_s, closest_wp_index_global
        
    def calculate_reward(self, next_obs, step, just_crashed):
        rewards = []
        for i in range(self.num_agents):
            collided = just_crashed[i] == 1
            reward = 0.0
            
            # -- Raceline Progress --
            # Logic: Find closest waypoint ahead of last achieved waypoint AND within lookahead distance
            #        Then calculate progress along raceline at waypoint, subtract from last distance_s
            current_pos = np.array([next_obs['poses_x'][i], next_obs['poses_y'][i]])
            
            # Define a search window: from the last achieved WP to the next 75.
            # This prevents the car from constantly locking onto the same, passed point.
            lookahead = 75
            
            current_s, global_wp_index = self._project_to_raceline(current_pos, self.last_wp_index[i], lookahead)
            
            # Calculate progress since last step, handling lap wrap-around
            progress = current_s - self.last_cumulative_distance[i]
            
            if progress < -self.raceline_length / 2:
                # Agent crossed finish line FORWARD
                progress += self.raceline_length
            elif progress > self.raceline_length / 2:
                # Agent crossed finish line BACKWARD (went wrong way)
                progress -= self.raceline_length
                        
            # Update tracker
            self.last_cumulative_distance[i] = current_s
            self.last_wp_index[i] = global_wp_index
            
            # Calculate distance from spawn point with proper wrap-around handling
            # distance_from_spawn should always be in range [0, raceline_length)
            # representing how far the agent has traveled in their CURRENT lap from spawn
            distance_from_spawn = current_s - self.start_s[i]
            
            # Wrap-Around cases:
            # 1. Velocity too low (agent has just started moving): clamp to 0
            # 2. Large negative (agent crossed finish line forward): add raceline_length
            # 3. Positive: keep as-is (normal forward progress)
            if distance_from_spawn < 0:
                if next_obs['linear_vels_x'][i] < 2.0:  # Threshold for "close to spawn" (50m tolerance)
                    distance_from_spawn = 0.0
                else:
                    # Agent has wrapped around (crossed finish line forward from high s-values)
                    distance_from_spawn += self.raceline_length
            
            # Add completed laps to get total distance
            total_distance = distance_from_spawn + (self.raceline_length * self.current_lap_count[i])

            # Lap count as a continuous float (e.g., 0.5 = halfway through first lap)
            new_lap_count = total_distance / self.raceline_length

            # Check for checkpoint completion (every 10% of lap)
            # Checkpoints are based on progress within CURRENT lap only
            lap_progress = new_lap_count - self.current_lap_count[i]  # Progress in current lap (0.0 to 1.0)
            current_checkpoint = int(lap_progress * 10)  # 0-9
            
            # Check for lap completion (completed a full lap from spawn)
            # Use 0.95 threshold to account for discrete waypoint projection (agent may not reach exactly 1.0)
            if new_lap_count >= self.current_lap_count[i] + 0.95:
                self.current_lap_count[i] += 1
                reward += self.LAP_REWARD * self.current_lap_count[i]
                print(f"Lap {self.current_lap_count[i]} completed by agent {i}! Step: {step} Bonus: {self.LAP_REWARD}", end="\r\n")
                
                # Reset checkpoint tracking for new lap
                self.last_checkpoint[i] = 0
                
                # This prevents runaway lap counts because next calculation will be:
                # distance_from_spawn = current_s - start_s ≈ 0 (fresh start for new lap)
                self.start_s[i] = current_s
                
            elif current_checkpoint > self.last_checkpoint[i]:
                checkpoint_bonus = self.CHECKPOINT_REWARD * current_checkpoint
                reward += checkpoint_bonus * (self.current_lap_count[i] + 1)
                self.last_checkpoint[i] = current_checkpoint
                print(f"Agent {i} reached checkpoint {current_checkpoint} ({int(lap_progress*100)}% of lap) - Bonus: {checkpoint_bonus:.2f}", end="\r\n")
                
            # Add step progress reward (incentivises speed)
            reward += progress * self.PROGRESS_REWARD_SCALAR * (self.current_lap_count[i] + 1)
            
            # -- Collision Penalty --
            if collided:
                rewards.append(self.COLLISION_PENALTY)
                continue # No rewards if collided

            # -- Reward Clipping --
            if reward <= -1000.0:
                print(f"\n=== LARGE REWARD DEBUG ===")
                print(f"Agent {i}, Step {step}")
                print(f"current_s: {current_s:.2f}")
                print(f"start_s: {self.start_s[i]:.2f}")
                print(f"distance_from_spawn: {distance_from_spawn:.2f}")
                print(f"total_distance: {total_distance:.2f}")
                print(f"current_lap_count: {self.current_lap_count[i]}")
                print(f"new_lap_count: {new_lap_count:.4f}")
                print(f"lap_progress: {lap_progress:.4f}")
                print(f"current_checkpoint: {current_checkpoint}")
                print(f"progress: {progress:.2f}")
                print(f"velocity: {next_obs['linear_vels_x'][i]:.2f}")
                print(f"========================\n")
                reward = -20.0
            elif reward >= 1000.0:
                print(f"\n=== LARGE REWARD DEBUG ===")
                print(f"Agent {i}, Step {step}")
                print(f"current_s: {current_s:.2f}")
                print(f"start_s: {self.start_s[i]:.2f}")
                print(f"distance_from_spawn: {distance_from_spawn:.2f}")
                print(f"total_distance: {total_distance:.2f}")
                print(f"current_lap_count: {self.current_lap_count[i]}")
                print(f"new_lap_count: {new_lap_count:.4f}")
                print(f"lap_progress: {lap_progress:.4f}")
                print(f"current_checkpoint: {current_checkpoint}")
                print(f"progress: {progress:.2f}")
                print(f"velocity: {next_obs['linear_vels_x'][i]:.2f}")
                print(f"========================\n")
                reward = 5.0
                
            rewards.append(reward)
                
        return np.array(rewards), np.array(rewards).mean() # Return list and avg
    
    def reset_progress_trackers(self, initial_poses_xy, agent_idxs=None):
        """Resets the cumulative distance tracker for all agents after an episode reset."""
        if agent_idxs is not None:
            for i in agent_idxs:
                current_pos = initial_poses_xy[i]
                
                # Find the globally closest waypoint (no lookahead needed here)
                distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
                closest_wp_index = np.argmin(distances)
                
                start_s_val = self.waypoints_s[closest_wp_index]
                self.last_cumulative_distance[i] = start_s_val
                self.last_wp_index[i] = closest_wp_index
                
                self.start_s[i] = start_s_val
                self.current_lap_count[i] = 0
                
                # Reset checkpoint tracking for crashed agents
                self.last_checkpoint[i] = 0
            return
        
        new_last_cumulative_distance = np.zeros(self.num_agents)
        new_last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        
        # Iterate over all starting positions
        for i in range(self.num_agents):
            current_pos = initial_poses_xy[i]
            
            # Find the globally closest waypoint (no lookahead needed here)
            distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
            closest_wp_index = np.argmin(distances)

            # Set the initial cumulative distance and index            
            start_s_val = self.waypoints_s[closest_wp_index]
            new_last_cumulative_distance[i] = start_s_val
            new_last_wp_index[i] = closest_wp_index
            
            self.start_s[i] = start_s_val
            self.current_lap_count[i] = 0
            
            # Reset checkpoint tracking for crashed agents
            self.last_checkpoint[i] = 0
            
        self.last_cumulative_distance = new_last_cumulative_distance
        self.last_wp_index = new_last_wp_index

    def pretrain_from_demonstrations(self, demo_buffer=None, epochs=2, gradient_accumulation_steps=4, bc_weights=None):
        """
        Supervised learning from human demonstrations using behavior cloning.
        Also stores demos for continual learning (prevents catastrophic forgetting).
        Pretrains BOTH actor (action prediction) and critic (value estimation).
        
        Optimizations:
        - Micro-batching: Process small sequential chunks together
        - Gradient accumulation: Update weights less frequently
        - Reduced epochs: 5 instead of 20 (sequential data = more info per sample)
        """
        # Store demos for continual BC regularization during RL training
        if demo_buffer is not None:
            self.demo_buffer = demo_buffer
            self.demo_pretrain_generation = self.generation_counter  # Mark when pretraining occurred
            print(f"\nPretraining from {len(self.demo_buffer)} human demonstrations...")
            print(f"Stored {len(self.demo_buffer)} demos for continual learning")
            print(f"Config: {epochs} epochs, grad_accum={gradient_accumulation_steps}")
            bc_weights = (1., 1.) # Full weight during pretraining
        elif demo_buffer is None and self.demo_buffer is not None:
            demo_buffer = self.demo_buffer
            print(f"\n  Running BC regularization from stored {len(self.demo_buffer)} human demonstrations...")
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        self.actor_network.train()
        self.critic_network.train()
        
        for epoch in range(epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0

            actor_buffer = None
            actor_hidden = (None, None)
            critic_buffer = None
            critic_hidden = (None, None)
            
            # Zero gradients at start of epoch
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            update_counter = 0

            for i, d in enumerate(demo_buffer):   
                # Add batch dimension to scan and state
                scan = torch.from_numpy(d['scan']).float().unsqueeze(0).to(self.device)
                scan = to_birds_eye(
                    scan.flatten(1),
                    num_beams=self.num_scan_beams,
                    fov=self.lidar_fov,
                    image_size=self.image_size
                ).unsqueeze(1).to(self.device)
                state = torch.from_numpy(d['state']).float().unsqueeze(0).to(self.device)
                action = torch.from_numpy(d['action']).float().unsqueeze(0).to(self.device)
                value = torch.tensor([d['value']], dtype=torch.float32).unsqueeze(0).to(self.device)

                # --- Actor forward pass ---
                predicted, _, actor_buffer_n, actor_hidden_h, actor_hidden_c = self.actor_network(
                    scan, state, actor_buffer, actor_hidden[0], actor_hidden[1]
                )
                actor_loss = torch.nn.functional.huber_loss(predicted, action, delta=1.0) * bc_weights[0] / gradient_accumulation_steps
                actor_loss.backward()
                epoch_actor_loss += actor_loss.item() * gradient_accumulation_steps / bc_weights[0]
                
                # Detach LSTM states to prevent backprop through entire sequence
                actor_buffer = actor_buffer_n.detach()
                actor_hidden = (actor_hidden_h.detach(), actor_hidden_c.detach())

                # --- Critic forward pass ---
                # predicted_values, critic_buffer_n, critic_hidden_h, critic_hidden_c = self.critic_network(
                #     scan, state, critic_buffer, critic_hidden[0], critic_hidden[1]
                # )
                # critic_loss = torch.nn.functional.huber_loss(predicted_values, value, delta=10.0) * bc_weights[1] / gradient_accumulation_steps
                # critic_loss.backward()
                # epoch_critic_loss += critic_loss.item() * gradient_accumulation_steps / bc_weights[1]
                
                # Detach LSTM states to prevent backprop through entire sequence
                # critic_buffer = critic_buffer_n.detach()
                # critic_hidden = (critic_hidden_h.detach(), critic_hidden_c.detach())
                
                # Update weights every gradient_accumulation_steps
                update_counter += 1
                if update_counter >= gradient_accumulation_steps:
                    self.actor_optimizer.step()
                    # self.critic_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    # self.critic_optimizer.zero_grad()
                    update_counter = 0
                
                    progress = (i + 1) / len(demo_buffer) * 100
                    avg_actor_loss = epoch_actor_loss / (i + 1)
                    # avg_critic_loss = epoch_critic_loss / (i + 1)
                    print(f"    Epoch {epoch+1}/{epochs}, Actor Loss: {avg_actor_loss:.4f} - {progress:.1f}% complete", end='\r')
            
            # Final update if there are remaining gradients
            if update_counter > 0:
                self.actor_optimizer.step()
                # self.critic_optimizer.step()
                self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                
            avg_actor_loss = epoch_actor_loss / len(demo_buffer)
            # avg_critic_loss = epoch_critic_loss / len(demo_buffer)
            
            if i % 5 == 0: print(f"    Epoch {epoch+1}/{epochs}, Actor Loss: {avg_actor_loss:.4f}")
            
            total_actor_loss += avg_actor_loss
            # total_critic_loss += avg_critic_loss
        
        self.buffer.empty()
        print(f"Pretraining complete. Avg Actor Loss: {total_actor_loss/epochs:.4f}, Avg Critic Loss: {total_critic_loss/epochs:.4f}\n")

    def learn(self, collisions, reward):
        """
        The "Coaching Session" where we train the networks.
        """
        print("Starting learning phase...")
        # Get all data from the "generation"
        data = self.buffer.sample(batch_size=len(self.buffer))
        minibatch_size = self.minibatch_size
        
        if len(data) <= minibatch_size:
            print(f"Computing GAE on GPU for {len(data)} samples...")
            minibatch_size = len(data)
            with torch.no_grad():
                data = self._compute_gae(data.to(self.device))
        else:
            print(f"Computing GAE on CPU for {len(data)} samples...")
            with torch.no_grad():
                data = self._compute_gae(data.to('cpu'))

        data = data.flatten(0, 1)
        total_samples = data.batch_size[0]
        
        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        current_gen_diagnostics["collisions"] = [collisions]
        current_gen_diagnostics["reward"] = [reward]
        
        # ============= PHASE 1: PPO Training (Full epochs on RL data) =============
        num_ppo_epochs = self.epochs_with_demos if (self.demo_buffer is not None and len(self.demo_buffer) > 0) else self.epochs
        print(f"PPO Training: {num_ppo_epochs} epochs on RL data")
        
        self.actor_network.train()
        self.critic_module.train()
        self.loss_module.train()
        for _ in range(num_ppo_epochs):                
            # Shuffle data indices for this epoch
            indices = torch.randperm(total_samples)
                        
            # Loop over all samples in minibatches
            for start in range(0, total_samples, minibatch_size):
                end = start + minibatch_size
                if end > total_samples:
                    continue # Skip the last, incomplete minibatch
                
                minibatch_indices = indices[start:end]
                
                # Sample the minibatch from the full dataset
                minibatch_data = data[minibatch_indices].to(self.device)
                
                # Get the loss from torchrl's PPO module (actor only)
                loss_td = self.loss_module(minibatch_data)

                # Actor loss from PPO
                actor_loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
                
                # # Add collision penalty to actor loss (auxiliary loss for safety)
                # # This provides a direct gradient signal to avoid collision-prone actions
                # if "next" in minibatch_data.keys() and "done" in minibatch_data["next"].keys():
                #     collision_mask = minibatch_data["next"]["done"].float()
                #     collision_rate = collision_mask.mean()
                #     # Scale by 2.0 to give collisions significant weight (same scale as entropy loss)
                #     collision_penalty = 10.0 * collision_rate
                #     actor_loss = actor_loss + collision_penalty
                    
                #     if start == 0 and epoch_idx == 0:  # Debug print once
                #         print(f"Collision rate in minibatch: {collision_rate.item():.4f}, penalty: {collision_penalty.item():.4f}")
                
                # Manually compute critic loss - re-run critic to get values with gradients
                self.critic_module(minibatch_data)
                predicted_values = minibatch_data.get("state_value")
                target_values = minibatch_data.get("value_target")
    
                
                # Ensure same shape for loss computation
                if predicted_values.shape != target_values.shape:
                    if predicted_values.ndim == 2 and predicted_values.shape[-1] == 1:
                        predicted_values = predicted_values.squeeze(-1)
                    elif target_values.ndim == 1:
                        target_values = target_values.unsqueeze(-1)
                
                critic_loss = nn.functional.huber_loss(predicted_values, target_values, delta=10.0)
                
                # Backpropagation for actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                # Backpropagation for critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

                # Collect diagnostics
                loss_td["loss_critic"] = critic_loss.detach()
                for key in self.diagnostic_keys:
                    if key in loss_td.keys():
                        value = loss_td[key].detach().cpu().item()
                        current_gen_diagnostics[key].append(value)
                        
        # ============= PHASE 2: BC Regularization (Reuse pretrain function for LSTM compatibility) =============
        if self.demo_buffer is not None and len(self.demo_buffer) > 0:
            # Calculate current BC weights with separate decay schedules (for loss weighting if needed)
            gens_since_pretrain = self.generation_counter - self.demo_pretrain_generation
            if gens_since_pretrain < self.demo_bc_decay_gens:
                # Linear decay from initial to final weight
                decay_progress = gens_since_pretrain / self.demo_bc_decay_gens
                
                # Actor BC weight
                current_bc_actor_weight = self.demo_bc_actor_weight_initial - decay_progress * (
                    self.demo_bc_actor_weight_initial - self.demo_bc_actor_weight_final)
                
                # Critic BC weight (separate schedule)
                current_bc_critic_weight = self.demo_bc_critic_weight_initial - decay_progress * (
                    self.demo_bc_critic_weight_initial - self.demo_bc_critic_weight_final)
            else:
                current_bc_actor_weight = self.demo_bc_actor_weight_final
                current_bc_critic_weight = self.demo_bc_critic_weight_final
            
            
            # Reuse pretrain function to maintain sequential LSTM processing
            if (self.generation_counter + 1) % int(5) == 0: 
                self.pretrain_from_demonstrations(demo_buffer=None, epochs=self.bc_epochs, gradient_accumulation_steps=4, bc_weights=(current_bc_actor_weight, current_bc_critic_weight))
                print(f"BC Regularization: {self.bc_epochs} epochs on {len(self.demo_buffer)} demos (actor_weight={current_bc_actor_weight:.3f}, critic_weight={current_bc_critic_weight:.3f})")

        self.generation_counter += 1
        for key in self.diagnostic_keys:
            values = current_gen_diagnostics.get(key)
            if values: # Check if list is not empty
                avg_value = np.mean(values)
                self.diagnostics_history[key].append(avg_value)
        
        # CRITICAL: Detect entropy collapse and stop training
        if current_gen_diagnostics.get("entropy"):
            avg_entropy = np.mean(current_gen_diagnostics["entropy"])
            if avg_entropy < -1.0 or np.isnan(avg_entropy):
                print(f"\n{'='*60}")
                print(f"⚠️  ENTROPY COLLAPSE DETECTED! ⚠️")
                print(f"Average entropy: {avg_entropy:.4f}")
                print(f"This indicates policy distribution has broken.")
                print(f"Training stopped at generation {self.generation_counter}")
                print(f"{'='*60}\n")
                # Save emergency checkpoint
                torch.save(self.actor_network.state_dict(), f"models/actor/emergency_gen_{self.generation_counter}.pt")
                torch.save(self.critic_network.state_dict(), f"models/critic/emergency_gen_{self.generation_counter}.pt")
                raise RuntimeError("Entropy collapse detected - training halted")
        
        if self.generation_counter > 0: self._plot_historical_diagnostics()
        
        # Clear the buffer for the next "generation"
        self.buffer.empty()
        self.reset_lstm_states()
        print("Learning complete.")
        
    def _plot_historical_diagnostics(self):
        """
        Generates and saves a plot showing the trend of average diagnostics
        across all completed generations. Overwrites the file each time.
        """
        # Define keys to plot (exclude generation if it's not in history dict)
        keys_to_plot = [k for k in self.diagnostic_keys if k != "generation" and k in self.diagnostics_history]
        num_metrics = len(keys_to_plot)

        if num_metrics == 0 or self.generation_counter == 0:
            print("No diagnostics data to plot yet.")
            return

        plt.style.use('dark_background')
        fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 4 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes] # Ensure axes is always iterable
        
        # Set global font size
        plt.rcParams['font.size'] = 16  # Adjust as needed

        # Set global line width
        plt.rcParams['lines.linewidth'] = 3 
        
        # X-axis: Generation number
        x_axis = np.arange(1, self.generation_counter + 1) # Generations 1, 2, 3...
        # Plot each metric's history
        for idx, key in enumerate(keys_to_plot):
            values = self.diagnostics_history.get(key, [])
            ax = axes[idx] # Get the correct subplot axis

            if not values: # Skip if no data for this key
                ax.set_ylabel(key)
                ax.grid(True)
                continue

            # Convert to numpy array, handling potential NaNs if some generations had errors
            values_np = np.array(values)

            # Plot only valid (non-NaN) points
            valid_indices = ~np.isnan(values_np)
            if np.any(valid_indices): # Check if there are any valid points to plot
                 ax.plot(x_axis[valid_indices], values_np[valid_indices], marker='.', linestyle='-', label=f'Avg {key}')
            ax.set_ylabel(key)
            ax.legend(loc='upper right')
            ax.grid(True)

        axes[-1].set_xlabel("Generation Number")
        fig.suptitle("Training Diagnostics History", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

        try:
            plt.savefig(self.plot_save_path)
            print(f"Diagnostics history plot saved to {self.plot_save_path}")
        except Exception as e:
            print(f"Error saving diagnostics history plot: {e}")
        plt.close(fig) # Close the figure to free memory
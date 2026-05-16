import gymnasium as gym
import gym_anytrading
from stable_baselines3 import PPO

# --- SECTION 1: THE ENVIRONMENT WRAPPER ---
# This class handles your custom reward logic and fixes the TypeError
class CustomTradingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_balance = 0
        self.prev_position = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- FIX FOR THE TYPEERROR ---
        # info['position'] returns a 'Positions' Enum object.
        # .value converts it to 0 (Short) or 1 (Long).
        position_object = info.get('position', 0)
        
        try:
            current_position = position_object.value   # Access the integer value
        except AttributeError:
            current_position = int(position_object)   # Fallback if it's already an int
            
        current_balance = info.get('total_profit', 0) 
        
        # --- REWARD CALCULATION ---
        # Formula: 0.2 * delta_balance + 0.8 * delta_position
        balance_diff = current_balance - self.prev_balance
        position_diff = current_position - self.prev_position
        
        custom_reward = (0.2 * balance_diff) + (0.8 * abs(position_diff))
        
        # Update trackers
        self.prev_balance = current_balance
        self.prev_position = current_position
        
        return obs, custom_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset custom variables when the environment restarts
        self.prev_balance = 0
        self.prev_position = 0
        return self.env.reset(**kwargs)

# --- SECTION 2: SETUP AND TRAINING ---
# Added render_mode='human' to ensure it opens a window correctly


base_env = gym.make('stocks-v0', frame_bound=(50, 1000), window_size=10, render_mode='human')

# Wrap it with your logic
env = CustomTradingWrapper(base_env)

# Hyperparameters: Change gamma or learning_rate here
model = PPO('MlpPolicy', env, 
            gamma=0.99, 
            learning_rate=0.0003, 
            verbose=1)

# Run training
print("Starting training...")
model.learn(total_timesteps=10000) 

# --- SECTION 3: EVALUATION ---
# Test the agent after training
print("Starting evaluation...")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.render()
print("Final Total Profit:", info['total_profit'])
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium import spaces

class LineFollowerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(LineFollowerEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float32)
        self.state = np.array([5, 5])
        self.target = np.array([10, 10])
        self.max_steps = 200
        self.current_step = 0
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([5, 5])
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.state = self.state + action
        self.current_step += 1
        done = np.linalg.norm(self.state - self.target) < 1 or self.current_step >= self.max_steps
        reward = -np.linalg.norm(self.state - self.target)
        truncated = self.current_step >= self.max_steps  # Eğer zaman sınırına ulaştıysa true olmalı
        return self.state, reward, done, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            plt.figure()
            plt.plot(self.state[0], self.state[1], 'bo')
            plt.plot(self.target[0], self.target[1], 'ro')
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.show()
            plt.close()

# Ortamı oluştururken render_mode parametresini belirtin
env = DummyVecEnv([lambda: LineFollowerEnv(render_mode='human')])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10)

obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    if i % 10 == 0:  # Her 10 adımda bir render çağır
        env.render()

model.save("line_follower_ddpg")

# Test süreci
obs = env.reset()
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    if i % 10 == 0:
         env.render()
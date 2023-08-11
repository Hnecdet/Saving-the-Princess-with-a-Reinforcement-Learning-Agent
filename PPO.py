# Import environment libraries
import cv2
import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils import startGameRand,
saveGameRand,startGameModel,saveGameModel
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack,
DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt
from utils import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO
from stable_baselines3 import DQN
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
# Wrapping the environment
# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Greyscale the environment
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Create the stacked frames
env = VecFrameStack(env, 4,channels_order='last')
env = VecMonitor(env, "./train/TestMonitor") # Monitor your
progress
callback = SaveOnBestTrainingRewardCallback(save_freq=10000,
check_freq=1000, chk_dir=CHECKPOINT_DIR)
model = PPO('CnnPolicy', env, verbose=1,
tensorboard_log=LOG_DIR, learning_rate=0.000001,n_steps=512)
model.learn(total_timesteps=4000000, callback=callback)
model = PPO.load('./train/best_model')
saveGameModel(env,model)

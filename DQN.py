# Import environment libraries
import gym_super_mario_bros
import cv2
from utils import startGameRand,
SaveOnBestTrainingRewardCallback
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack,
DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack,
DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Start the environment
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
env = gym_super_mario_bros.make('SuperMarioBros-v0') #
Generates the environment
env = JoypadSpace(env, SIMPLE_MOVEMENT) # Limits the joypads
moves with important moves
env = GrayScaleObservation(env, keep_dim=True) # Convert to
grayscale to reduce dimensionality
env = DummyVecEnv([lambda: env])
# Alternatively, you may use SubprocVecEnv for multiple CPU
processors
env = VecFrameStack(env, 4, channels_order='last') # Stack
frames
env = VecMonitor(env, "./train/TestMonitor") # Monitor your
progress
callback = SaveOnBestTrainingRewardCallback(save_freq=10000,
check_freq=1000,
chk_dir=CHECKPOINT_DIR)
model = DQN('CnnPolicy',
 env,
 batch_size=192,
 verbose=1,
 learning_starts=10000,
 learning_rate=5e-3,
 exploration_fraction=0.1,
 exploration_initial_eps=1.0,
 exploration_final_eps=0.1,
 train_freq=8,
 buffer_size=10000,
 tensorboard_log=LOG_DIR
)
model.learn(total_timesteps=1000000, log_interval=1,
callback=callback)
model = DQN.load('./train/best_model')
saveGameModel(env,model)

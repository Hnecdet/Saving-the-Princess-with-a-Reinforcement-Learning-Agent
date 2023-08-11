# Saving-the-Princess-with-a-Reinforcement-Learning-Agent

I trained a Reinforcement Learning agent in atari environment to win games.
More specifically, i created an RL model to play the first stage of the Super Mario Bros game. The
implementations will be in Python language and you will be using PyTorch, OpenAI Gym, NES
Emulator and Stable-Baselines3. I also used TensorBoard to track the progress of your agent. You
can download necessary libraries using pip install gym super mario bros==7.3.0 nes py stable-baselines3. 
Make sure PyTorch is already installed before installing Stable Baselines3, otherwise Stable Baselines3 may automatically
download CPU version of PyTorch, regardless whether you have GPU or not.

You may simply use the following code to start a game environment. Don’t forget to load startGameRand
from utils.py.
# Import environment libraries
import gym super mario bros
from nes py.wrappers import JoypadSpace
from gym super mario bros.actions import SIMPLE MOVEMENT
# Start the environment
env = gym super mario bros.make(’SuperMarioBros-v0’) # Generates the environment
env = JoypadSpace(env, SIMPLE MOVEMENT) # Limits the joypads moves with important moves
startGameRand(env)
If the game runs properly, you may go on with the preprocessing steps. Also, you may save your gameplay
as a video using saveGameRand from utils.py. Don’t forget to install FFmpeg to your computer. There
may be easier download methods for FFmpeg depending on your OS, so you can Google it.
# Import preprocessing wrappers
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt
# Apply the preprocessing
env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality
env = DummyVecEnv([lambda: env])
# Alternatively, you may use SubprocVecEnv for multiple CPU processors
env = VecFrameStack(env, 4, channels_order=’last’) # Stack frames
env = VecMonitor(env, "./train/TestMonitor") # Monitor your progress
3
Don’t forget to create "./train/’ directory (or some alternative name) as your CHECKPOINT DIR and
"./logs/’ directory as LOG DIR.

2.1 PPO

Import your callback function from utils.py. Please note that, each weight file keeps around 250-350
MBs, so decide your save freq accordingly, not exceeding 100000. Also, import PPO function from
Stable-Baselines3. Then, you may start training. Train for at least 1 million timesteps. Make sure
Tensorboard is logging ep rew mean and entropy loss properly.
from utils import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO
callback = SaveOnBestTrainingRewardCallback(save freq=10000, check freq=1000,
chk dir=CHECKPOINT_DIR)
model = PPO(’CnnPolicy’, env, verbose=1, tensorboard log=LOG DIR, learning rate=0.000001,
n steps=512)
model.learn(total_timesteps=4000000, callback=callback)
After training, it’s time to test it out. You may load your best model, or any model you want. Also, you
may save your gameplay as a video using saveGameModel from utils.py.
model = PPO.load(’./train/best_model’)
startGameModel(env, model)

2.2 DQN

Repeat 2.1 with DQN algorithm. Make sure Tensorboard is logging ep rew mean and loss properly.
from stable_baselines3 import DQN
model = DQN(’CnnPolicy’,
env,
batch size=192,
verbose=1,
learning starts=10000,
learning rate=5e-3,
exploration fraction=0.1,
exploration initial eps=1.0,
exploration final eps=0.1,
train freq=8,
buffer size=10000,
tensorboard log=LOG DIR
)
model.learn(total timesteps=4000000, log interval=1, callback=callback)

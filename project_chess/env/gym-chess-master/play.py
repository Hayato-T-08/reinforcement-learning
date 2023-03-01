import gym
from gym_chess import ChessEnvV1, ChessEnvV2

env1 = ChessEnvV1()
env2 = ChessEnvV2()

env1 = gym.make('ChessVsSelf-v1')
env2 = gym.make('ChessVsSelf-v2')
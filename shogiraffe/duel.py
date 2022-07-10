import time

from .agents.minimax_agent import MinimaxAgent
from .agents.random_agent import RandomAgent
from .environment.shogi_env import ShogiEnv
from .evaluation.handmade_evaluation import HandmadeEvaluator

env = ShogiEnv()
obs = env.reset()

# agent = RandomAgent()
agent1 = MinimaxAgent(HandmadeEvaluator(), depth=2)
agent2 = RandomAgent()

side = 0
for step in range(1000):
    if side == 0:
        action = agent1.play(obs)
    else:
        action = agent2.play(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    # time.sleep(0.001)
    if done:
        side = 0
        env.reset()
    else:
        side = (side + 1) % 2

# Close the env
env.close()

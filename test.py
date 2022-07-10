from shogiraffe.environment.shogi_env import ShogiEnv

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: ShogiEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

import gym

env = gym.make('Acrobot-v1')

# 環境の初期化
observation = env.reset()

# エピソードのループ
for t in range(1000):
    # ランダムなアクションの選択
    action = env.action_space.sample()
    
    # アクションの実行
    observation, reward, done, info = env.step(action)
    
    # 状態の表示
    print(observation)
    env.render()
    
    # 終了条件のチェック
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()

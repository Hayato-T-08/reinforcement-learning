# reinforcement-learning
このリポジトリは強化学習の勉強のために作ったリポジトリです。
## 使用するフレームワークとライブラリ

フレームワークはtensorflowとkerasをつかい環境はほかの方が作ったopenAI gymをベースにした環境を使います.また、keras-rl2という簡単に強化学習ができるライブラリを使います。

必要なライブラリ　tensorflow keras-rl2 h5py Pillow gym gym[atari] 等
## project_acrobot
open ai gym のacrobot環境をdqn等の強化学習のアルゴリズムを用いて学習させます。

acrobot環境の詳しい情報は'https://www.gymlibrary.dev/environments/classic_control/acrobot/'
を参照してください
### example.py
このファイルはランダムな行動を取らせacrobot環境がどのようなもの確認するためのファイルです。
### simple_dqn.py
dqnを用いてopen ai gym のacrobot環境を学習しますgpu環境で学習に30分かかります。
### double_dqn.py
double_dqnを使ってacrobot環境を学習しますコードはほとんどdqnと同じです。
### trained_model.py
学習済みモデルの重みを用いて環境をプレイします。

## project_cartpole
一番単純な環境です。詳細は'https://www.gymlibrary.dev/environments/classic_control/cart_pole/'
### keras_rl_dqn.py
keras-rl2を用いてcartpoleを学習させるコードです。
### keras_rl_trained.py
学習済みモデルの重みを用いてカートポールをプレイさせます。ほとんどのステップで報酬は最大値の200となっています。

環境の閾値が-100となっているので十分に学習できたといえます。

報酬を最大化するためにすぐに黒い線にあたろうとするのが興味深いです。


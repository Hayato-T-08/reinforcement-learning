# reinforcement-learning
このリポジトリは強化学習の勉強のために作ったリポジトリです。
## 使用するフレームワークとライブラリ

keras-rl2という簡単に強化学習ができるライブラリを使います。

必要なライブラリ　tensorflow keras-rl2 h5py Pillow gym gym[atari] 等
## project_acrobot
open ai gym のacrobot環境をdqn等の強化学習のアルゴリズムを用いて学習させます。

acrobot環境の詳しい情報は'https://www.gymlibrary.dev/environments/classic_control/acrobot/'
を参照してください
### example.py
このファイルはランダムな行動を取らせacrobot環境がどのようなもの確認するためのファイルです。
### keras_rl_dqn.py
dqnを用いてopen ai gym のacrobot環境を学習しますgpu環境で学習に30分かかります。
### keras_rl_trained.py
学習済みモデルの重みを用いて環境をプレイします。

## project_cartpole
一番単純な環境です。詳細は'https://www.gymlibrary.dev/environments/classic_control/cart_pole/'
### keras_rl_dqn.py
keras-rl2を用いてcartpoleを学習させるコードです。
### keras_rl_trained.py
学習済みモデルの重みを用いてカートポールをプレイさせます。ほとんどのステップで報酬は最大値の200となっています。

環境の閾値が-100となっているので十分に学習できたといえます。

報酬を最大化するためにすぐに黒い線にあたろうとするのが興味深いです。

## project_invaders
ネット上でSpaceInvadersをプレイするRLモデルを見つけました。'https://huggingface.co/blog/deep-rl-dqn'

そちらのコードを共有します。こちらのサイトではQ学習やDQNのアルゴリズムを勉強できるのでおすすめです。(全部英語です)

自分で作ろうと思ったのですが使用するライブラリがgymの仕様変更により使えなかったり、tesorflow 2.10.0との互換性がなくエラーが発生するので断念しました。





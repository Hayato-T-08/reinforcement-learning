# reinforcement-learning
このリポジトリは強化学習の勉強のために作ったリポジトリです。
## 使用するフレームワークとライブラリ

フレームワークはtensorflowとkerasをつかい環境はほかの方が作ったopenAI gymをベースにした環境を使います
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



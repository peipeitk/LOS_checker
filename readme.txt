LOS_cheker
google earth proで衛星とアンテナ間の視線ベクトルを表示するkmlファイルを作成する
読み込んだ仰角, 方位角ファイルと同名のディレクトリを作成し、その中にkmlファイルを衛星ごとに作成する

・実行コマンド
静止点解析
python LOS_checker.py static (rtkplotで出力した仰角、方位角テキストファイル) (観測点緯度)　(観測点経度)　(観測点楕円体高度)　(観測点ジオイド高)
移動体解析
python LOS_checker.py kinematic (rtkplotで出力した仰角、方位角テキストファイル) (観測点posファイル) (観測点ジオイド高)

・実行コマンド例
python LOS_checker.py static sample_static.txt 34.54650625015037 135.50210414238964 63.02384679485112 38.026
python LOS_checker.py kinematic sample_kinematic.txt PPK.pos 38.026


準備

・仰角、方位角テキストファイルの作成方法
rtkplot.exeでobsファイルを開く
設定で"Time Format"を"h:m:s GPST"に設定する
File→Save AZ/EL/SNR/MP...

・観測点posファイルの作成方法
rtkpost.exeでobs, navファイルからposファイルを作成する
設定で"Time Format"を"h:m:s GPST"に設定する
経緯度を出力するように設定する
そこそこの精度が必要なため、相対測位が好ましい

・ジオイド高の調べ方
https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/calc_f.html

・Google Earth Proのインストール
https://support.google.com/earth/answer/21955?hl=ja

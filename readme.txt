LOS_cheker
google earth proで衛星とアンテナ間の視線ベクトルを表示する、kmlファイルを衛星ごとに作成する
現在静止点解析のみ対応

・実行コマンド
python LOS_checker.py (rtkplotで出力した仰角、方位角テキストファイル) (観測点緯度)　(観測点経度)　(観測点楕円体高度)　(観測点ジオイド高)

・実行コマンド例
python LOS_checker.py sample.txt 34.54650625015037 135.50210414238964 63.02384679485112 38.026


準備

・仰角、方位角テキストファイルの作成方法
rtkplot.exeでobsファイルを開く
設定で"Time Format"を"h:m:s GPST"に設定する
File→Save AZ/EL/SNR/MP...

・ジオイド高の調べ方
https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/calc_f.html

・Google Earth Proのインストール
https://support.google.com/earth/answer/21955?hl=ja

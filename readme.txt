###########################################################################################################
 LOS_checker 2.0変更内容
　実行コマンドが少し変わりました.
    ・オプション機能の追加
　オプションコマンド: --buildingsで, LOS, Multipath, NLOSを色分けし, 経路を表示してくれるようになりました.
###########################################################################################################

LOS_cheker
google earth proで衛星とアンテナ間の視線ベクトルを表示するkmlファイルを作成する
読み込んだ仰角, 方位角ファイルと同名のディレクトリを作成し、その中にkmlファイルを衛星ごとに作成する

・実行コマンド
静止点解析
$ python LOS_checker.py (rtkplotで出力した仰角、方位角テキストファイル) --static (観測点緯度,観測点経度,観測点楕円体高度)　(オプション)
移動体解析
$ python LOS_checker.py (rtkplotで出力した仰角、方位角テキストファイル) --kinematic (観測点posファイル) (オプション)

オプション
--buildings :建物情報ファイルを読み込み, 信号の到来経路を計算し, LOS, Multipath, NLOSに分類して表示する. ex) --buildings OPU_buildings.txt
--elp_hgt   :計測地点の楕円体高度を設定する. デフォルトは38.031[m]. ex) --elp_hgt 39.0
--length    :信号到来線の長さを変更する. デフォルトは100[m]. ex) --length 500

・実行コマンド例
$ python LOS_checker.py sample_static.txt --static 34.54650625,135.50210414,63.0238
$ python LOS_checker.py sample_static.txt --static 34.54650625,135.50210414,63.0238 --buildings OPU_buildings.txt
$ python LOS_checker.py sample_kinematic.txt --kinematic PPK.pos

準備
・pythonにnumpy, pandas, tqdmをダウンロード
$ pip install numpy
$ pip install pandas
$ pip install tqdm

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

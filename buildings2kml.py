import math
import sys
import os

import numpy as np
import pandas as pd


def make_header():
    with open('lib/kml_header.kml', 'r') as f:
        header = f.read()
    return header


def make_los_point(tmp_buildings_d):
    buildings_point = ''
    #海抜楕円体高度
    elp_alt = 38.0285
    for buildings_info in tmp_buildings_d.values():
        #ヘッダ
        header = '<Folder>\n<name>'+ buildings_info[0] +'</name>\n'
        #建物高さ
        b12alt = buildings_info[3] - elp_alt
        b34alt = b12alt - buildings_info[2]
        #データ部分
        data = ''
        for i in range(buildings_info[1]):
            vertex_str = str(buildings_info[4+i][2])+','+str(buildings_info[4+i][1])+','+str(b12alt)
            #中身の作成
            data = data + '<Placemark>\n<styleUrl>#P3</styleUrl>\n<Point>\n<extrude>1</extrude>\n<altitudeMode>absolute</altitudeMode>\n<coordinates>' + vertex_str + '</coordinates>\n</Point>\n</Placemark>\n'
            vertex_str = str(buildings_info[4+i][4])+','+str(buildings_info[4+i][3])+','+str(b12alt)
            data = data + '<Placemark>\n<styleUrl>#P4</styleUrl>\n<Point>\n<extrude>1</extrude>\n<altitudeMode>absolute</altitudeMode>\n<coordinates>' + vertex_str + '</coordinates>\n</Point>\n</Placemark>\n'

        #フッタ
        footer = '</Folder>\n'

        building_point = header + data + footer
        buildings_point += building_point
    return buildings_point


def read_buildings_info(path):
    #建物情報の読み込み
    with open(path, "r", encoding="utf-8") as f:
        line = [s.strip() for s in f.readlines()]
    #建物情報の保存: tmp_buildings_d
    #key 建物名
    #value リストbuildings_info
    #0 建物名
    #1 読み込んだ建物面の数
    #1 建物全長[m]
    #2 建物楕円体高度[m]
    #3 建物面ごとの頂点座標を格納したリスト
    #4 建物面ごとの頂点座標を格納したリスト
    #・・・
    tmp_buildings_d = {}
    vertexes = []
    faces_counter = 0
    for s in line:
        if '#' in s:
            continue
        if '%' in s:
            buildings_info = []
            head = s.split(',')
            buildings_info.append(head[0][1:])
            buildings_info.append(int(head[1]))
            buildings_info.append(float(head[2]))
            buildings_info.append(float(head[3]))
            continue
        if '-' in s:
            vertexes.append(s[1:])
            vertexes_counter = 0
            faces_counter += 1
            continue
        if vertexes_counter < 2:
            vertex = s.split(',')
            vertexes.append(float(vertex[0]))
            vertexes.append(float(vertex[1]))
            if vertexes_counter == 1:
                buildings_info.append(vertexes)
                vertexes = []
                if faces_counter == buildings_info[1]:
                    tmp_buildings_d[buildings_info[0]] = buildings_info
                    faces_counter = 0
                continue
            vertexes_counter += 1
            continue
    return tmp_buildings_d


#kmlファイルを作成
args = sys.argv
#ヘッダ部分を作成
header_kml = make_header()
#point描画部分を作成
buildings_path = args[1]
tmp_buildings_d = read_buildings_info(buildings_path)
los_point_kml = make_los_point(tmp_buildings_d)
#フッタ部分を作成
footer_kml = '</Document>\n</kml>'


los_kml = header_kml + los_point_kml + footer_kml
with open('buildings.kml', 'w', encoding="utf-8") as f:
    f.write(los_kml)

# Google Earthで使用するkmlファイルを作成する
# 衛星との見通し線を表示することで、直接波が受信できる環境かどうかを判定する

import math
import sys
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


########################################
###########txtファイル読み込み##########
########################################

def read_add(path):
    """
    addファイルを読み込む
    """
    with open (path, "r") as f:
        lines = [s.strip() for s in f.readlines()]
    return lines


def separate_columns_add(add_lines):
    """
    spaceを区切ってデータフレームを作成
    """
    columns=['GPST', 'SAT', 'AZ', 'EL', 'SNR', 'L1_MP']
    df = pd.DataFrame(columns=columns)
    gpst = []
    sat = []
    az = []
    el = []
    snr = []
    l1_mp = []
    for i, s in enumerate(add_lines):
        if i == 0:
            continue
        gpst.append(s[0:21])
        sat.append(s[25:28])
        az.append(s[32:38])
        el.append(s[42:47])
        snr.append(s[51:59])
        l1_mp.append(s[61:71])
    df['GPST'] = gpst
    df['SAT'] = sat
    df['AZ'] = az
    df['EL'] = el
    df['SNR'] = snr
    df['L1_MP'] = l1_mp
    return df


########################################
###########posファイル読み込み##########
########################################

def read_pos(path):
    """
    posファイルを読み込みデータフレームに変換する
    """
    with open(path, "r") as f:
        lines = [s.strip() for s in f.readlines()]

    count = 0
    for s in lines:
        if '%' in s:
            count += 1
        else:
            break
    del lines[0:count-1]
    return lines


def separate_columns_pos(lines):
    """
    読み込んだファイルのリストをカラムごとにデータフレームに変換する
    """
    for s in lines:
        if '%' in s:
            columns = s.split()
            del columns[0]
            df = pd.DataFrame(columns=columns)
            continue
        data_l = s.split()
        data_l[1] = data_l[1][:-2]
        gpst = ' '.join(data_l[0:2])
        del data_l[0:2]
        data_l.insert(0, gpst)
        data_s = pd.Series(data_l, index=df.columns)
        df = df.append(data_s, ignore_index=True)
    return df


#######################################################################
############################kmlファイル作成############################
#######################################################################

def make_kml(name, df, buildings_path=None):
    #仰角がゼロの点を排除
    df = df.astype({'EL':'float16'})
    df = df[df['EL'] > 0.0]
    #建物情報の読み込み
    if not buildings_path is None:
        tmp_buildings_d = read_buildings_info(buildings_path)
        buildings_d = cal_buildings_info(tmp_buildings_d, base_llh)

    count = 0
    sat_l = list(dict.fromkeys(df['SAT'].values.tolist()))
    for sat in tqdm(sat_l):
        #print(sat, '読み込み中')
        #obsファイルの現在時刻のデータ数を出力
        row0 = count
        for i in range(row0, df.shape[0]):
            if df['SAT'].iloc[i] == sat:
                count += 1
                #行列最後の調整
                if count == df.shape[0]:
                    end_index = count
            else:
                end_index = count
                break
        #同じ衛星のデータフレームを作成
        df_copy = df[['GPST', 'AZ', 'EL']].iloc[row0:end_index].copy()

        los_line_l, nlos_line_l, multipath_line_l, los_points_l = [], [], [], []
        for i in range(df_copy.shape[0]):
            #衛星位置をllhに直す（hは標高）
            az = float(df_copy['AZ'].iloc[i])
            #azが0のときスキップ
            if az == 0:
                continue
            el = float(df_copy['EL'].iloc[i])
            az_rad = math.radians(az)
            el_rad = math.radians(el)
            d = [math.cos(el_rad)*math.sin(az_rad), math.cos(el_rad)*math.cos(az_rad), math.sin(el_rad)]
            satpos_l = plr2llh([length, az, el], base_llh)
            satpos_l[0] = round(satpos_l[0], 9)
            satpos_l[1] = round(satpos_l[1], 9)
            satpos_l[2] = round(satpos_l[2]-elp_sea_diff, 3)
            #経度、緯度、標高の順番に直し、strに変換
            satpos_str = ','.join(map(str, [satpos_l[1], satpos_l[0], satpos_l[2]]))
            if not buildings_path is None:
                #NLOSかどうかを判定
                nlos_flag = check_whether_nlos(d, buildings_d)
                if not nlos_flag:
                    X, multipath_flag = check_whether_multipath(d, buildings_d, base_llh)

                #線描画用リストの作成
                if nlos_flag:
                    nlos_line_l.append(out_base_llh_sea_str)
                    nlos_line_l.append(satpos_str)
                    #点描画用リストの作成
                    los_points_l.append([df_copy['GPST'].iloc[i], satpos_str])
                elif multipath_flag:
                    #マルチパス波用の新たな衛星位置を算出
                    satpos2 = np.array(X) + length*np.array(d)
                    satpos2 = ecef2llh(enu2ecef(satpos2, base_llh))
                    satpos2[2] = round(satpos2[2]-elp_sea_diff, 3)
                    satpos2_str = ','.join(map(str, [satpos2[1], satpos2[0], satpos2[2]]))
                    #反射位置を緯度経度高度に変換
                    reflection_point = ecef2llh(enu2ecef(X, base_llh))
                    reflection_point[2] = round(reflection_point[2]-elp_sea_diff, 3)
                    reflection_point_str = ','.join(map(str, [reflection_point[1], reflection_point[0], reflection_point[2]]))
                    #線描画用リストの作成
                    multipath_line_l.append(out_base_llh_sea_str)
                    multipath_line_l.append(reflection_point_str)
                    multipath_line_l.append(satpos2_str)
                    multipath_line_l.append(reflection_point_str)
                    #点描画用リストの作成
                    los_points_l.append([df_copy['GPST'].iloc[i], satpos2_str])
                else:
                    los_line_l.append(out_base_llh_sea_str)
                    los_line_l.append(satpos_str)
                    #点描画用リストの作成
                    los_points_l.append([df_copy['GPST'].iloc[i], satpos_str])
            else:
                los_line_l.append(out_base_llh_sea_str)
                los_line_l.append(satpos_str)
                #点描画用リストの作成
                los_points_l.append([df_copy['GPST'].iloc[i], satpos_str])

        #kmlファイルを作成
        #ヘッダ部分を作成
        header_kml = make_header()
        #line描画部分を作成
        los_line_kml = make_los_line(los_line_l, 'LOS', '00ffff')
        if len(nlos_line_l) > 0:
            nlos_line_kml = make_los_line(nlos_line_l, 'NLOS', 'ff00ff')
            los_line_kml = los_line_kml + nlos_line_kml
        if len(multipath_line_l) > 0:
            multipath_line_kml = make_los_line(multipath_line_l, 'Multipath', 'ffff00')
            los_line_kml = los_line_kml + multipath_line_kml
        #point描画部分を作成
        los_point_kml = make_los_point(los_points_l)
        #フッタ部分を作成
        footer_kml = '</Document>\n</kml>'


        los_kml = header_kml + los_line_kml + los_point_kml + footer_kml
        with open(name + '/'+ sat + '.kml', 'w') as f:
            f.write(los_kml)


def make_kml_kinematic(name, df_sat, df_pos, buildings_path):
    #建物情報の読み込み
    if not buildings_path is None:
        tmp_buildings_d = read_buildings_info(buildings_path)

    count = 0
    sat_l = list(dict.fromkeys(df_sat['SAT'].values.tolist()))
    for sat in tqdm(sat_l):
        #obsファイルの現在時刻のデータ数を出力
        row0 = count
        for i in range(row0, df_sat.shape[0]):
            if df_sat['SAT'].iloc[i] == sat:
                count += 1
                #行列最後の調整
                if count == df_sat.shape[0]:
                    end_index = count
            else:
                end_index = count
                break
        #同じ衛星のデータフレームを作成
        df_sat_copy = df_sat[['GPST', 'AZ', 'EL']].iloc[row0:end_index].copy()

        los_line_l, nlos_line_l, multipath_line_l, los_points_l = [], [], [], []
        for i in range(df_pos.shape[0]):
            if i == 0:
                add_counter = 0
            #posファイルの時刻がaddファイルに合っているか確認
            for j in range(df_sat_copy.shape[0]):
                if not df_pos['GPST'].iloc[i] == df_sat_copy['GPST'].iloc[add_counter]:
                    if add_counter >= df_sat_copy.shape[0]-1:
                        break
                    add_counter += 1
                else:
                    break
            pos_l = [float(df_pos['latitude(deg)'].iloc[i]), float(df_pos['longitude(deg)'].iloc[i]),\
                     float(df_pos['height(m)'].iloc[i])]
            pos_sea_l = [pos_l[1], pos_l[0], float(pos_l[2])-elp_sea_diff]
            pos_sea_str = ','.join(map(str, pos_sea_l))
            los_line_l.append(pos_sea_str)
            #建物の法線ベクトルの計算
            if not buildings_path is None:
                buildings_d = cal_buildings_info(tmp_buildings_d, pos_l)
            #衛星位置をllhに直す（hは標高）
            az = float(df_sat_copy['AZ'].iloc[add_counter])
            #azが0のときスキップ
            if az == 0:
                continue
            el = float(df_sat_copy['EL'].iloc[add_counter])
            az_rad = math.radians(az)
            el_rad = math.radians(el)
            d = [math.cos(el_rad)*math.sin(az_rad), math.cos(el_rad)*math.cos(az_rad), math.sin(el_rad)]
            satpos_l = plr2llh([length, az, el], pos_l)
            satpos_l[0] = round(satpos_l[0], 9)
            satpos_l[1] = round(satpos_l[1], 9)
            satpos_l[2] = round(satpos_l[2]-elp_sea_diff, 3)
            #経度、緯度、標高の順番に直し、strに変換
            satpos_str = ','.join(map(str, [satpos_l[1], satpos_l[0], satpos_l[2]]))
            if not buildings_path is None:
                #NLOSかどうかを判定
                nlos_flag = check_whether_nlos(d, buildings_d)
                if not nlos_flag:
                    X, multipath_flag = check_whether_multipath(d, buildings_d, pos_l)

                #線描画用リストの作成
                if nlos_flag:
                    nlos_line_l.append(satpos_str)
                    nlos_line_l.append(pos_sea_str)
                    #点描画用リストの作成
                    los_points_l.append([df_pos['GPST'].iloc[i], pos_sea_str])
                elif multipath_flag:
                    #マルチパス波用の新たな衛星位置を算出
                    satpos2 = np.array(X) + length*np.array(d)
                    satpos2 = ecef2llh(enu2ecef(satpos2, pos_l))
                    satpos2[2] = round(satpos2[2]-elp_sea_diff, 3)
                    satpos2_str = ','.join(map(str, [satpos2[1], satpos2[0], satpos2[2]]))
                    #反射位置を緯度経度高度に変換
                    reflection_point = ecef2llh(enu2ecef(X, pos_l))
                    reflection_point[2] = round(reflection_point[2]-elp_sea_diff, 3)
                    reflection_point_str = ','.join(map(str, [reflection_point[1], reflection_point[0], reflection_point[2]]))
                    #線描画用リストの作成
                    multipath_line_l.append(pos_sea_str)
                    multipath_line_l.append(reflection_point_str)
                    multipath_line_l.append(satpos2_str)
                    multipath_line_l.append(reflection_point_str)
                    #点描画用リストの作成
                    los_points_l.append([df_pos['GPST'].iloc[i], pos_sea_str])
                else:
                    los_line_l.append(satpos_str)
                    los_line_l.append(pos_sea_str)
                    #点描画用リストの作成
                    los_points_l.append([df_pos['GPST'].iloc[i], pos_sea_str])
            else:
                los_line_l.append(satpos_str)
                los_line_l.append(pos_sea_str)
                #点描画用リストの作成
                los_points_l.append([df_pos['GPST'].iloc[i], pos_sea_str])

        #kmlファイルを作成
        #ヘッダ部分を作成
        header_kml = make_header()
        #line描画部分を作成
        los_line_kml = make_los_line(los_line_l, 'LOS', '00ffff')
        if len(nlos_line_l) > 0:
            nlos_line_kml = make_los_line(nlos_line_l, 'NLOS', 'ff00ff')
            los_line_kml = los_line_kml + nlos_line_kml
        if len(multipath_line_l) > 0:
            multipath_line_kml = make_los_line(multipath_line_l, 'Multipath', 'ffff00')
            los_line_kml = los_line_kml + multipath_line_kml
        #point描画部分を作成
        los_point_kml = make_los_point(los_points_l)
        #フッタ部分を作成
        footer_kml = '</Document>\n</kml>'


        los_kml = header_kml + los_line_kml + los_point_kml + footer_kml
        with open(name + '/'+ sat + '.kml', 'w') as f:
            f.write(los_kml)


def make_header():
    with open('lib/kml_header.kml', 'r') as f:
        header = f.read()
    return header


def make_los_line(los_line_l, signal_type, color):
    header = '<Placemark>\n<name>'+ signal_type + '_line</name>\n<Style>\n<LineStyle>\n<color>ff' + color + '</color>\n</LineStyle>\n</Style>\n<LineString>\n<altitudeMode>absolute</altitudeMode>\n<coordinates>\n'
    footer = '\n</coordinates>\n</LineString>\n</Placemark>\n'
    data = '\n'.join(map(str, los_line_l))
    los_line = header + data + footer
    return los_line


def make_los_point(los_points_l):
    #ヘッダ
    header = '<Folder>\n<name>LOS_Point</name>\n'
    #データ部分
    data = ''
    for los_point_l in los_points_l:
        #時刻の変換
        gpst = los_point_l[0]
        year = gpst[0:4]
        month = gpst[5:7]
        date = gpst[8:10]
        time = gpst[11:]
        time_kml = year + '-' + month + '-' + date + 'T' + time + 'Z'
        #中身の作成
        data = data + '<Placemark>\n<styleUrl>#P3</styleUrl>\n<TimeStamp><when>' + time_kml + '</when></TimeStamp>\n<Point>\n<extrude>1</extrude>\n<altitudeMode>absolute</altitudeMode>\n<coordinates>' + los_point_l[1] + '</coordinates>\n</Point>\n</Placemark>\n'

    #フッタ
    footer = '</Folder>\n'

    los_point = header + data + footer
    return los_point


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


def cal_buildings_info(tmp_buildings_d, base_llh):
    #建物ごとに計算
    buildings_d = {}
    for building in tmp_buildings_d.values():
        buildings_info = [building[1]]
        #建物面ごとに計算
        for i in range(building[1]):
            vertexes = building[4+i]
            #建物面の頂点座標を定義
            b1 = [vertexes[1], vertexes[2], building[3]]
            b2 = [vertexes[3], vertexes[4], building[3]]
            b3 = [vertexes[3], vertexes[4], building[3]-building[2]]
            b4 = [vertexes[1], vertexes[2], building[3]-building[2]]
            #受信位置を基準にENU変換
            b1 = np.array(LLH2ENU(base_llh, b1))
            b2 = np.array(LLH2ENU(base_llh, b2))
            b3 = np.array(LLH2ENU(base_llh, b3))
            b4 = np.array(LLH2ENU(base_llh, b4))
            #単位法線ベクトルnを計算
            N = np.cross(b3-b4, b1-b4)
            n = N / np.linalg.norm(N, ord=2)
            buildings_info += [[n, b1, b2, b3, b4]]
            if i == building[1]-1:
                buildings_d[building[0]] = buildings_info
    return buildings_d


def check_whether_nlos(d, buildings_d, skip_building=None):
    for j, building in enumerate(buildings_d.values()):
        if list(buildings_d.keys())[j] == skip_building:
            continue
        for i in range(building[0]):
            n = building[i+1][0]
            b1 = building[i+1][1]
            #計算する必要のない建物面をスキップ
            if np.dot(-np.array(b1),np.array(n)) <= 0:
                continue
            b2 = building[i+1][2]
            b3 = building[i+1][3]
            b4 = building[i+1][4]
            A = np.matrix([[1/d[0], -1/d[1], 0], [0, 1/d[1], -1/d[2]], [n[0], n[1], n[2]]])
            Y = np.matrix([[0], [0], [n[0]*b1[0] + n[1]*b1[1] + n[2]*b1[2]]])
            x = np.linalg.inv(A)*Y
            if min([b1[0], b2[0], b3[0], b4[0]]) <= x[0] <= max([b1[0], b2[0], b3[0], b4[0]]):
                if min([b1[1], b2[1], b3[1], b4[1]]) <= x[1] <= max([b1[1], b2[1], b3[1], b4[1]]):
                    if min([b1[2], b2[2], b3[2], b4[2]]) <= x[2] <= max([b1[2], b2[2], b3[2], b4[2]]):
                        #print(ecef2llh(enu2ecef(x, base_llh)))
                        return True
    return False


def check_whether_multipath(d, buildings_d, base_llh):

    for j, building in enumerate(buildings_d.values()):
        for i in range(building[0]):
            n = building[i+1][0]
            b1 = building[i+1][1]
            #計算する必要のない建物面をスキップ
            if np.dot(-np.array(b1),np.array(n)) <= 0:
                continue
            b2 = building[i+1][2]
            b3 = building[i+1][3]
            b4 = building[i+1][4]
            A_dash = np.matrix([[1-2*n[0]**2, -2*n[0]*n[1], -2*n[2]*n[0]],\
                                [-2*n[0]*n[1], 1-2*n[1]**2, -2*n[1]*n[2]],\
                                [-2*n[2]*n[0], -2*n[1]*n[2], 1-2*n[2]**2]])
            Y_dash = np.matrix([[d[0]], [d[1]], [d[2]]])
            X_dash = np.linalg.inv(A_dash)*Y_dash
            A = np.matrix([[np.array(X_dash)[0][0], b4[0]-b1[0], b4[0]-b3[0]],\
                           [np.array(X_dash)[1][0], b4[1]-b1[1], b4[1]-b3[1]],\
                           [np.array(X_dash)[2][0], b4[2]-b1[2], b4[2]-b3[2]]])
            Y = np.matrix([[b4[0]], [b4[1]], [b4[2]]])
            T = np.linalg.inv(A)*Y
            X = [(1-np.array(T)[1][0]-np.array(T)[2][0])*b4[0] + np.array(T)[1][0]*b1[0] + np.array(T)[2][0]*b3[0],\
                 (1-np.array(T)[1][0]-np.array(T)[2][0])*b4[1] + np.array(T)[1][0]*b1[1] + np.array(T)[2][0]*b3[1],\
                 (1-np.array(T)[1][0]-np.array(T)[2][0])*b4[2] + np.array(T)[1][0]*b1[2] + np.array(T)[2][0]*b3[2]]

            if min([b1[0], b2[0], b3[0], b4[0]]) <= X[0] <= max([b1[0], b2[0], b3[0], b4[0]]):
                if min([b1[1], b2[1], b3[1], b4[1]]) <= X[1] <= max([b1[1], b2[1], b3[1], b4[1]]):
                    if min([b1[2], b2[2], b3[2], b4[2]]) <= X[2] <= max([b1[2], b2[2], b3[2], b4[2]]):
                        #TODO二点以上反射点がある場合
                        #print(ecef2llh(enu2ecef(X, base_llh)))
                        #新たな反射点と受信位置との線が建物にはばまれていないか確認
                        if check_whether_nlos(X, buildings_d, list(buildings_d.keys())[j]):
                            return None, False
                        return X, True
    return None, False


############################################################################
################################座標変換####################################
############################################################################

#定数の定義
a = 6378137.0
f = 1/298.257223563
b = a*(1-f)
e2 = f*(2-f)
e2d = (a**2 - b**2)/b**2


def LLH2ECEF(llh):
    lat = math.radians(llh[0])
    lon = math.radians(llh[1])
    h = llh[2]
    new = a/math.sqrt(1-e2*(math.sin(lat))**2)
    x = (new + h)*math.cos(lat)*math.cos(lon)
    y = (new + h)*math.cos(lat)*math.sin(lon)
    z = (h + new*(1 - e2))*math.sin(lat)
    return [x, y, z]


def cal_R_ENU(llh):
    phi = math.radians(llh[0])
    lam = math.radians(llh[1])
    e11 = -math.sin(lam)
    e12 = math.cos(lam)
    e13 = 0
    e21 = -math.sin(phi)*math.cos(lam)
    e22 = -math.sin(phi)*math.sin(lam)
    e23 = math.cos(phi)
    e31 = math.cos(phi)*math.cos(lam)
    e32 = math.cos(phi)*math.sin(lam)
    e33 = math.sin(phi)

    R_ENU = np.array([[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]])
    return R_ENU


def LLH2ENU(llh0, llh):
    x = LLH2ECEF(llh)
    R_ENU = cal_R_ENU(llh)
    x0 = np.array(LLH2ECEF(llh0))
    x = np.array(x)
    pos_diff = x - x0
    x_ENU = np.dot(R_ENU, pos_diff)
    return x_ENU


def ENU2PLR(enu):
    r  = math.sqrt(enu[0]**2 + enu[1]**2 + enu[2]**2)
    az = math.atan2(enu[0], enu[1])
    el = math.atan2(enu[2], math.sqrt(enu[0]**2 + enu[1]**2))
    return [r, math.degrees(az), math.degrees(el)]


def plr2enu(plr):
    az = math.radians(plr[1])
    el = math.radians(plr[2])
    length = plr[0]
    e = length*math.cos(el)*math.sin(az)
    n = length*math.cos(el)*math.cos(az)
    u = length*math.sin(el)
    return [e, n, u]


def enu2ecef(enu, llh0):
    enu = np.array(enu)
    enu = enu.reshape(3, 1)
    R_ENU = cal_R_ENU(llh0)
    r0 = LLH2ECEF(llh0)
    r0 = np.array(r0)
    r0 = r0.reshape(3, 1)
    R_ENU_inv = np.linalg.inv(R_ENU)
    ecef_out = np.dot(R_ENU_inv, np.array(enu)) + r0
    return ecef_out.reshape(3, )


def ecef2llh(ecef):
    p = math.sqrt(ecef[0]**2 + ecef[1]**2)
    theta = math.atan2((ecef[2]*a), (p*b))

    lat = math.atan2((ecef[2]+e2d*b*(math.sin(theta))**3), p-e2*a*(math.cos(theta))**3)
    lon = math.atan2(ecef[1], ecef[0])
    v = a/math.sqrt(1-e2*(math.sin(lat))**2)
    h = p/math.cos(lat) - v
    return [math.degrees(lat), math.degrees(lon), h]


def plr2llh(plr, llh0):
    enu = plr2enu(plr)
    ecef = enu2ecef(enu, llh0)
    llh = ecef2llh(ecef)
    return llh



parser = argparse.ArgumentParser()
parser.add_argument("ELAZ_path", help='rtkplotで出力した仰角方位角ファイルのパス')
parser.add_argument("--static", help='静止点測位での観測位置(緯度経度高度)')
parser.add_argument("--elp_hgt", help='観測地点の楕円体高. デフォルト38.031[m]')
parser.add_argument("--kinematic", help='移動体の場合の、rtkposで出力した観測地点posファイルのパス')
parser.add_argument("--buildings", help='建物情報から衛星地点を分類するための建物情報ファイルのパス')
parser.add_argument("--length", help='グーグルアースに表示するLOS直線の長さを変更. デフォルト100[m]')
args = parser.parse_args()

#仰角方位角ファイルの読み込み
add_path = args.ELAZ_path
#建物情報の読み込み
if args.buildings:
    buildings_path = args.buildings
else:
    buildings_path = None
#LOS直線の長さを設定[m]
if args.length:
    length = float(args.length)
else:
    length = 100

#kmlファイルを収納するディレクトリを作成
if not os.path.exists(add_path[:-4]):
    os.mkdir(add_path[:-4])


if args.static:
    obs_llh = args.static.split(',')
    base_llh = [float(obs_llh[0]), float(obs_llh[1]), float(obs_llh[2])]
    #海抜の楕円体高
    if args.elp_hgt:
        elp_sea_diff = float(args.elp_hgt)
    else:
        elp_sea_diff = 38.031
    base_llh_sea = [base_llh[0], base_llh[1], round(base_llh[2]-elp_sea_diff, 3)]
    #経度、緯度、標高の順番に直し、strに変換
    out_base_llh_sea_str = ','.join(map(str, [base_llh_sea[1], base_llh_sea[0], base_llh_sea[2]]))

    lines = read_add(add_path)
    df = separate_columns_add(lines)

    make_kml(add_path[:-4], df, buildings_path)

elif args.kinematic:
    df_pos = separate_columns_pos(read_pos(args.kinematic))
    df_sat = separate_columns_add(read_add(add_path))
    if args.elp_hgt:
        elp_sea_diff = float(args.elp_hgt)
    else:
        elp_sea_diff = 38.031
    make_kml_kinematic(add_path[:-4], df_sat, df_pos, buildings_path)

else:
    print('--static or --kinematicをコマンドに入力してください')
    sys.exit()

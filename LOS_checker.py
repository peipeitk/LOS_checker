# Google Earthで使用するkmlファイルを作成する
# 衛星との見通し線を表示することで、直接波が受信できる環境かどうかを判定する

import numpy as np
import pandas as pd
import math
import sys
import os


########################################
###########txtファイル読み込み###########
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


#######################################################################
############################kmlファイル作成############################
#######################################################################

def make_kml(name, df):
    count = 0
    sat_l = list(dict.fromkeys(df['SAT'].values.tolist()))
    for sat in sat_l:
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

        los_line_l, los_points_l = [], []
        for i in range(df_copy.shape[0]):
            #衛星位置をllhに直す（hは標高）
            az = float(df_copy['AZ'].iloc[i])
            el = float(df_copy['EL'].iloc[i])
            satpos_l = plr2llh([length, az, el], base_llh)
            satpos_l[0] = round(satpos_l[0], 9)
            satpos_l[1] = round(satpos_l[1], 9)
            satpos_l[2] = round(satpos_l[2]-elp_sea_diff, 3)
            #経度、緯度、標高の順番に直し、strに変換
            satpos_str = ','.join(map(str, [satpos_l[1], satpos_l[0], satpos_l[2]]))

            #線描画用リストの作成
            los_line_l.append(out_base_llh_sea_str)
            los_line_l.append(satpos_str)
            #点描画用リストの作成
            los_points_l.append([df_copy['GPST'].iloc[i], satpos_str])

        #kmlファイルを作成
        #ヘッダ部分を作成
        header_kml = make_header()
        #line描画部分を作成
        los_line_kml = make_los_line(los_line_l)
        #point描画部分を作成
        los_point_kml = make_los_point(los_points_l)
        #フッタ部分を作成
        footer_kml = '</Document>\n</kml>'


        los_kml = header_kml + los_line_kml + los_point_kml + footer_kml
        with open(name + '/'+ sat + '.kml', 'w') as f:
            f.write(los_kml)


def make_header():
    with open('data/lib/kml_header.kml', 'r') as f:
        header = f.read()
    return header


def make_los_line(los_line_l):
    header = '<Placemark>\n<name>LOS_line</name>\n<Style>\n<LineStyle>\n<color>ff00ffff</color>\n</LineStyle>\n</Style>\n<LineString>\n<altitudeMode>absolute</altitudeMode>\n<coordinates>\n'
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


def ECEF2ENU(llh0, x):
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
    R_ENU = cal_R_ENU(llh0)
    r0 = LLH2ECEF(llh0)
    R_ENU_inv = np.linalg.inv(R_ENU)
    return np.dot(R_ENU_inv, np.array(enu)) + r0


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


args = sys.argv

base_llh = [float(args[2]), float(args[3]), float(args[4])]
#海抜の楕円体高
elp_sea_diff = float(args[5])
base_llh_sea = [base_llh[0], base_llh[1], round(base_llh[2]-elp_sea_diff, 3)]
#経度、緯度、標高の順番に直し、strに変換
out_base_llh_sea_str = ','.join(map(str, [base_llh_sea[1], base_llh_sea[0], base_llh_sea[2]]))
length = 100

path = args[1]
lines = read_add(path)
df = separate_columns_add(lines)

#kmlファイルを収納するディレクトリを作成
if not os.path.exists(path[:-4]):
    os.mkdir(path[:-4])

make_kml(path[:-4], df)

#%% 기본 실행
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
from PIL import Image
import math
import os
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import multiprocessing

def sorting_txt_data(path):
    return path.split('tracks/')[1].split('_')[0]

def path_keyword(path):
    return path.split('tracks/')[1][0]

def sorting_image_data(path):
    return path.split('track_background/')[1].split('.')[0]

#횡단보도랑 가장 가까운 축
def area_speed(area, point):
    point_ = abs(point - area)
    
    return point_

def point_to_tuple(p):
    return (p.x, p.y)

def remove_outliers(data):
    Q1 = np.percentile(data, 25)  # 1사분위수
    Q3 = np.percentile(data, 75)  # 3사분위수
    IQR = Q3 - Q1  # IQR 계산
    lower_bound = Q1 - 1.5 * IQR  # 이상치 하한값
    upper_bound = Q3 + 1.5 * IQR  # 이상치 상한값
    
    # 이상치 제거 후 데이터 반환
    return [x for x in data if lower_bound <= x <= upper_bound]

#%% 추가 데이터 생성
# spot_features = pd.DataFrame({'place':       ['spot_A',    'spot_B',     'spot_C',   'spot_D',   'spot_E',   'spot_F',    'spot_G',   'spot_H',    'spot_I'],
#                               'size' :       [(1920,1080),(1920,1080),  (1920,1080),(1280,720), (1280,720), (1280,720),  (1280,720),  (1280,720), (1920,1080)],
#                               'image_1':     [(632,201),  (745,153),    (494,56),   (661,11),   (257,61),   (629,59),    (333,115),   (21,153),   (13,17)],
#                               'image_2':     [(826,199),  (1217,171),   (789,49),   (955,43),   (631,69),   (1083,75),   (685,105),   (299,83),   (533,13)],
#                               'image_3':     [(1760,925), (2437,1019),  (2361,665), (1037,667), (1661,711), (1757,1083), (1623,717),  (1611,399), (2405,688)],
#                               'image_4':     [(240,1423), (7,1103),     (0,1080),   (0,391),    (0,669),    (1,321),     (74,720),    (439,1193), (343,1440)],
#                               'crosswalk_1': [(588,731),  (733,489),    (443,457),  (415,191),  (305,349),  (253,275),   (403,195),   (191,257),  (445,590)],
#                               'crosswalk_2': [(1272,663), (1575,517),   (1431,421), (801,259),  (873,311),  (1061,243),  (669,186),   (533,185),  (1283,424)],
#                               'crosswalk_3': [(1590,969), (1920,857),   (1765,641), (731,437),  (1389,683), (1103,787),  (769,273),   (809,273),  (1915,682)],
#                               'crosswalk_4': [(552,1313), (611,885),    (365,757),  (231,317),  (313,657),  (91,351),    (385,290),   (329,381),  (743,1104)],
#                               'walk_left_1 : [(543,501),  (748,158),    (502,63),   (663,16),   (258,72),   (626,63),    (336,122),   (31,154),   (28,26)],
#                               'walk_left_2 : [(593,499),  (841,163),    (563,61),   (708,21),   (389,73),   (695,63),    (421,124),   (77,143),   (110,24)],
#                               'walk_left_3 : [(537,1305), (522,1073),   (257,1032), (120,409),  (322,666),  (77,350),    (371,321),   (394,468),  (699,1070)],
#                               'walk_left_4 : [(257,1395), (30,1090),    (13,1076),  (21,385),   (26,653),   (19,325),    (210,420),   (237,671),  (290,1175)],
#                               'walk_right_1: [(815,223),  (1163,176),   (738,58),   (898,39),   (560,73),   (1055,82),   (601,115),   (243,101),  (379,24)],
#                               'walk_right_2: [(833,215),  (1215,179),   (793,56),   (947,46),   (622,74),   (1076,83),   (688,111),   (301,87),   (539,22)],
#                               'walk_right_3: [(1746,926), (2405,1011),  (1931,514), (1025,659), (1615,699), (1490,721),  (1003,321),  (1280,327), (2123,597)],
#                               'walk_right_4: [(1610,970), (2131,1021),  (1931,705), (694,568),  (1419,692), (1112,723),  (806,288),   (1279,426), (1907,667)]
#                               })

spot_features = pd.DataFrame({'place':       ['spot_A',    'spot_I'],
                              'size' :       [(1920,1080)  ,(1920,1080)],
                              'image_1':     [(632,201),   (13,17)],
                              'image_2':     [(826,199),     (533,13)],
                              'image_3':     [(1760,925),   (2405,688)],
                              'image_4':     [(240,1423),    (343,1440)],
                              'crosswalk_1': [(609,729),    (445,590)],
                              'crosswalk_2': [(1264,664), (1283,424)],
                              'crosswalk_3': [(1574,976),  (1915,682)],
                              'crosswalk_4': [(588,1298),  (743,1104)],
                              'walk_left_1': [(543,501), (28,26)],
                              'walk_left_2': [(593,499), (110,24)],
                              'walk_left_3': [(537,1305), (699,1070)],
                              'walk_left_4': [(257,1395), (290,1175)],
                              'walk_right_1': [(815,223), (379,24)],
                              'walk_right_2': [(833,215), (539,22)],
                              'walk_right_3': [(1746,926), (2123,597)],
                              'walk_right_4': [(1610,970), (1907,667)],
                              })

only_car_total_list_A = []
crosswalk_total_list_A = []
walk_total_list_A = []
only_car_total_list_I = []
crosswalk_total_list_I = []
walk_total_list_I = []
spot_distance_onlycar = {'A':0, 'B':0}
spot_distance_crosswalk = {'A':0, 'B':0}
spot_distance_walk = {'A':0, 'B':0}
A_area = [680.5,1361,2041.5,2722,2944]
I_area = [629.5,1259,1888.5,2518,2939]


#%% DATA
data_middle_path = []
text_name = []
data_path = []
car_data = pd.DataFrame()

exp_path = os.listdir("project/a_i_track")    # exp, exp2....

for i in range(len(exp_path)):
    data_middle_path.append("project/a_i_track/" + exp_path[i] + "/tracks/")

    # 텍스트 파일명 불러오기
for j in data_middle_path:
    text = os.listdir(j)
    text_name.append(text)


    # 데이터 경로 불러오기
for i in range(len(text_name)):
    data_path.append("{}".format(data_middle_path[i] + text_name[i][0]))

    # 데이터 경로 정렬
data_path = sorted(data_path, key=sorting_txt_data)

# 전체 데이터 경로 불러오기 및 정렬
spot_A_data = []
# spot_B_data = []
# spot_C_data = []
# spot_D_data = []
# spot_E_data = []
# spot_F_data = []
# spot_G_data = []
# spot_H_data = []
spot_I_data = []

for path in data_path:
    character = path.split('tracks/')[1][0]
    
    if character == 'A':
        spot_A_data.append(path)
    # if character == 'B':
    #     spot_B_data.append(path)
    # if character == 'C':
    #     spot_C_data.append(path)
    # if character == 'D':
    #     spot_D_data.append(path)
    # if character == 'E':
    #     spot_E_data.append(path)
    # if character == 'F':
    #     spot_F_data.append(path)
    # if character == 'G':
    #     spot_G_data.append(path)
    # if character == 'H':
    #     spot_H_data.append(path)
    if character == 'I':
        spot_I_data.append(path)
spot_data = [spot_A_data, spot_I_data]
# spot_data = [spot_A_data, spot_B_data, spot_C_data, spot_D_data, spot_E_data, spot_F_data, spot_G_data, spot_H_data, spot_I_data]



#%% IMAGE
image_path = []
image_middle_path = os.listdir("project/A_I_background")

for i in range(len(image_middle_path)):
    image_path.append("project/A_I_background/" + image_middle_path[i])
# image_path = sorted(image_path, key=sorting_image_data)

#%% 전체적인 전처리1
for _path in spot_data:  #하나의 SPOT 끝날 때까지 반복

    distance = 0
    crosswalk_distance = 0
    walk_distance = 0 
    only_car_total_list = []
    crosswalk_total_list = []
    walk_total_list = []
    area_1 = []
    area_2 = []
    area_3 = []
    area_4 = []
    area_5 = []
    
    if "A" in _path[0]:
        image_path_one = 'project/A_I_background/spot_A.jpg'
        spot_feat = spot_features.iloc[0]       
        now_spot = "A"
    # if "B" in _path[0]:
    #     image_path_one = 'project/track_background/spot_B.jpg'
    #     spot_feat = spot_features.iloc[1] 
    #     now_spot = "B"
    # if "C" in _path[0]:
    #     image_path_one = 'project/track_background/spot_C.jpg'
    #     spot_feat = spot_features.iloc[2]
    #     now_spot = "C"
    # if "D" in _path[0]:
    #     image_path_one = 'project/track_background/spot_D.jpg'
    #     spot_feat = spot_features.iloc[3]
    #     now_spot = "D"
    # if "E" in _path[0]:
    #     image_path_one = 'project/track_background/spot_E.jpg'
    #     spot_feat = spot_features.iloc[4] 
    #     now_spot = "E"     
    # if "F" in _path[0]:
    #     image_path_one = 'project/track_background/spot_F.jpg'
    #     spot_feat = spot_features.iloc[5]
    #     now_spot = "F"
    # if "G" in _path[0]:
    #     image_path_one = 'project/track_background/spot_G.jpg'
    #     spot_feat = spot_features.iloc[6]
    #     now_spot = "G"
    # if "H" in _path[0]:
    #     image_path_one = 'project/track_background/spot_H.jpg'
    #     spot_feat = spot_features.iloc[7]
    #     now_spot = "H"
    if "I" in _path[0]:
        image_path_one = 'project/A_I_background/spot_I.jpg'
        spot_feat = spot_features.iloc[1]
        #spot_feat = spot_features.iloc[8]
        now_spot = "I"
    
    image = cv2.imread(image_path_one, cv2.IMREAD_COLOR)


    for path in _path:   #하나의 파일이 끝날 때까지 반복
        data = pd.read_csv(path, sep=" ", header=None)
        
        
        width = spot_features['size'][0][0]
        height = spot_features['size'][0][1]
        image = cv2.resize(image, (width, height))
        
        data.columns = ["frame", "id", "x", "y", "width", "height", "column7", "column8", "column9", "class_number", "column11", "blank"]
        selected_columns = ["frame", "id", "class_number", "x", "y", "width", "height"]
        data = data[selected_columns]

        data.rename(columns={"x":"x1", "y":"y1"}, inplace = True)
    #     break
    # break
#%% #####직사각형 그릴때만 사용
        # data = data.assign(x2 = data["x1"] + data["width"]) 
        # data = data.assign(y2 = data["y1"] + data["height"])
#%% 전체적인 전처리2
        # under_point = []
        # for i in range(len(data)):
        #     point = (data['x1'][i] + data['width'][i]/2, data['y1'][i] + data['height'][i])
        #     under_point.append(point)
        # data["under_point"] = under_point     
        data["under_point"] = data.apply(lambda row: (row['x1'] + row['width'] / 2, row['y1'] + row['height']), axis=1)


        # ID / class_number unique
        unique_ids = data["id"].unique()
        unique_class = data["class_number"].unique()
        unique_frame = data["frame"].unique()      

#%% #####직사각형 출력 -> 사용 X
# for class_num in unique_class:
#         class_data_person = filtered_data[filtered_data["class_number"] == 0]
#         class_data_person = class_data_person.reset_index()
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         for p in range(0, len(class_data_person)//4, 100):
#             plt.imshow(cv2.rectangle(image, (class_data_person['x1'][p], class_data_person['y1'][p]),
#                                      (class_data_person['x2'][p], class_data_person['y2'][p]), color, thickness=5))
        
#         class_data_bicycle = filtered_data[filtered_data["class_number"] == 1]
#         class_data_bicycle = class_data_bicycle.reset_index()
#         color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
#         for b in range(0,len(class_data_bicycle)):
#              plt.imshow(cv2.rectangle(image, (class_data_bicycle['x1'][p], class_data_bicycle['y1'][p]),
#                                      (class_data_bicycle['x2'][p], class_data_bicycle['y2'][p]), color, thickness=5))
        
#         class_data_car = filtered_data[filtered_data["class_number"] == 2]
#         class_data_car = class_data_car.reset_index()
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         for c in range(0, len(class_data_car)//4, 500):
#             plt.imshow(cv2.rectangle(image, (class_data_car['x1'][c], class_data_car['y1'][c]),
#                                      (class_data_car['x2'][c], class_data_car['y2'][c]), color, thickness=2))
#             plt.axis("on")
# plt.show()

#%% #####이동 경로 point로 출력
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV BGR 이미지를 RGB로 변환하여 표시
# for class_num in unique_class:
#     class_data = data[data["class_number"] == class_num]
#     class_data = class_data.reset_index(drop = True)
#     plt.scatter(class_data["under_point"][0], class_data["under_point"][1], s=50)
# plt.axis("on")
# plt.show()

#%% 행렬 변환
        origin_points = [spot_feat[2],spot_feat[3],spot_feat[4],spot_feat[5]]
        h = 3000
        w = 2000
        base_points = [(0, 0),(w, 0),(w, h),(0, h)]

        M = cv2.getPerspectiveTransform(np.array(origin_points, dtype='float32'), np.array(base_points, dtype='float32'))
#%% 이미지 행렬 변환
        cvt_image = cv2.warpPerspective(image, M, (w, h))
        cvt_image = cv2.cvtColor(cvt_image, cv2.COLOR_BGR2RGB)       

#%% POINT 행렬 변환 + 데이터 프레임에 topview_point열 추가     
        # new_data = pd.DataFrame()
        # for class_num in unique_class:
        #     data_ = data[data["class_number"] == class_num]
        #     data_ = data_.reset_index(drop = True)
        #     point = []
                
        #     for p in range(0, len(data_)):
        #         cvt_point = cv2.perspectiveTransform(np.array([[np.array((data_['under_point'][p]), dtype = 'float32'), ]]), M)
        #         cvt_point = (int(cvt_point[0][0][0]), int(cvt_point[0][0][1]))
        #         point.append(cvt_point)
        #     data_ = data_.assign(topview_point = point)
        #     new_data = pd.concat([data_,new_data],ignore_index=True)
        # data = new_data
        
        # data = data.sort_values(['frame', 'id'], ignore_index = True)
        data["under_point"] = data["under_point"].apply(lambda p: Point(p))

        data["under_point_tuple"] = data["under_point"].apply(lambda p: (p.x, p.y))
        data["topview_point"] = data["under_point_tuple"].apply(lambda p: cv2.perspectiveTransform(np.array([[np.array(p, dtype='float32')]]), M))
        data["topview_point"] = data["topview_point"].apply(lambda p: (int(p[0][0][0]), int(p[0][0][1])))

        data = data.sort_values(['frame', 'id'], ignore_index=True)
        data.drop(columns=["under_point_tuple"], inplace=True)

        print("전처리")

#%% only_car 차량만 있을 때의 차 spot별 차량 속도

        # 데이터에서 사람 있는 프레임의 행 제거
        # only_car = pd.DataFrame()
        # for i in unique_frame:
        #     mix = new_data.loc[new_data['frame'] == i]
        #     if 0 in mix['class_number'].unique(): 
        #         pass
        #     else:    
        #         only_car = pd.concat([only_car, mix])
                
        # only_car = only_car.sort_values(['id','frame'], ignore_index= True)
        # only_car.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)  
        
        only_car = data[data['class_number'] != 0].copy()
        only_car = only_car.sort_values(['id', 'frame']).drop_duplicates(subset=None, keep='first').reset_index(drop=True)

        
        # 각 id 별 거리 계산     
        id = []
        distance_list = []
        for i in range(len(only_car)):
            if only_car['id'][i] not in id:
                distance_list.append(0.0)
                id.append(only_car['id'][i])
            else:
                dist = math.sqrt((only_car['topview_point'][i][0] - only_car['topview_point'][i-1][0])**2 +
                                (only_car['topview_point'][i][1] - only_car['topview_point'][i-1][1])**2)
                dist = dist/abs(only_car['frame'][i] - only_car['frame'][i-1])
                distance_list.append(dist)
                only_car_total_list.append(dist)
        only_car = only_car.assign(distance = distance_list)
        each_mean = only_car['distance'].mean()
        
        distance = distance + each_mean
        del id
        del distance_list
        print("only_car")


           
#%% crosswalk에 따른 차량 속도
        # 600, 1200, 1800, 2400, 3000 -> 선그래프 그린 후 x축에서 저 좌표만 추출
          
        car_data = pd.DataFrame()   
        car_data = data[data.class_number != 0]
        car_data = car_data.reset_index(drop = True)
        person_data = data[data.class_number != 2]
        person_data = person_data.reset_index(drop = True)        
        crosswalk_person_frame = []
        crosswalk_point = []
        crosswalk_append = []
        crosswalk_df = pd.DataFrame()

        for i in range(6,10):        
            crosswalk_p = cv2.perspectiveTransform(np.array([[np.array((spot_feat[i]), dtype = 'float32'), ]]), M)
            crosswalk_p = (int(crosswalk_p[0][0][0]), int(crosswalk_p[0][0][1]))
            # crosswalk_p = spot_feat[i]
            crosswalk_point.append(crosswalk_p)
        
        crosswalk_polygon = Polygon(crosswalk_point)
        
        for i in range(len(person_data)):
            if crosswalk_polygon.contains(Point(person_data['topview_point'][i])):
                crosswalk_person_frame.append(person_data['frame'][i])
        
        crosswalk_person_frame = list(set(crosswalk_person_frame))

        
        for i in range(len(car_data)):
            if car_data['frame'][i] in crosswalk_person_frame:
                    crosswalk_append.append(car_data.iloc[i])
        crosswalk_df = pd.DataFrame(crosswalk_append)
        crosswalk_df = crosswalk_df.sort_values(['id','frame'])
        crosswalk_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
        unique_crosswalk_ids = crosswalk_df["id"].unique()

        
        crosswalk_id = []
        crosswalk_distance_list = []     
        for i in range(len(crosswalk_df)):
            if crosswalk_df['id'][i].item() not in crosswalk_id:
                crosswalk_distance_list.append(0)
                crosswalk_id.append(crosswalk_df['id'][i])
            else:
                dist = math.sqrt((crosswalk_df['topview_point'][i][0] - crosswalk_df['topview_point'][i-1][0])**2 +
                                (crosswalk_df['topview_point'][i][1] - crosswalk_df['topview_point'][i-1][1])**2)
                dist = dist/abs(crosswalk_df['frame'][i] - crosswalk_df['frame'][i-1])
                crosswalk_distance_list.append(dist)
                crosswalk_total_list.append(dist)
        crosswalk_df = crosswalk_df.assign(crosswalk_distance = crosswalk_distance_list)
        each_mean = crosswalk_df['crosswalk_distance'].mean()
        
        crosswalk_distance = crosswalk_distance + each_mean
        
#%% 특정 지정 구역
        for i in unique_crosswalk_ids:
            if now_spot == "A":
                dot1_list = []
                dot2_list = []
                dot3_list = []
                dot4_list = []
                dot5_list = []
                filtered_data = crosswalk_df[crosswalk_df['id'] == i]
                filtered_data = filtered_data.sort_values(['id', 'frame'])
                for j in range(len(filtered_data)):     
                    dot1 = area_speed(680.5,filtered_data.iloc[j][8][1])
                    dot1_list.append(dot1)
                filtered_data['dot1'] = dot1_list #crosswalk_df.update(filtered_data) 
                area1 = filtered_data['crosswalk_distance'][filtered_data['dot1'].idxmin()]
                    
                for j in range(len(filtered_data)):                  
                    dot2 = area_speed(1361,filtered_data.iloc[j][8][1])                    
                    dot2_list.append(dot2)
                filtered_data['dot2'] = dot2_list
                area2 = filtered_data['crosswalk_distance'][filtered_data['dot2'].idxmin()]
                    
                for j in range(len(filtered_data)):                     
                    dot3 = area_speed(2041.5,filtered_data.iloc[j][8][1])                    
                    dot3_list.append(dot3)
                filtered_data['dot3'] = dot3_list
                area3 = filtered_data['crosswalk_distance'][filtered_data['dot3'].idxmin()]
                    
                for j in range(len(filtered_data)):                      
                    dot4 = area_speed(2722,filtered_data.iloc[j][8][1])                    
                    dot4_list.append(dot4)
                filtered_data['dot4'] = dot4_list
                area4 = filtered_data['crosswalk_distance'][filtered_data['dot4'].idxmin()]
                                
                for j in range(len(filtered_data)): 
                    dot5 = area_speed(2944,filtered_data.iloc[j][8][1])                    
                    dot5_list.append(dot5)
                filtered_data['dot5'] = dot5_list
                area5 = filtered_data['crosswalk_distance'][filtered_data['dot3'].idxmin()]
                    

                
                area_1.append(area1)
                area_2.append(area2)
                area_3.append(area3)
                area_4.append(area4)
                area_5.append(area5)
                
            if now_spot == "I":                
                dot1_list = []
                dot2_list = []
                dot3_list = []
                dot4_list = []
                dot5_list = [] 

                filtered_data = filtered_data.sort_values(['id', 'frame'])
                filtered_data = crosswalk_df[crosswalk_df['id'] == i]
                for j in range(len(filtered_data)):
                    dot1 = area_speed(629.5,filtered_data.iloc[j][8][1])
                    dot1_list.append(dot1)
                filtered_data['dot1'] = dot1_list
                area1 = filtered_data['crosswalk_distance'][filtered_data['dot1'].idxmin()]
                    
                for j in range(len(filtered_data)):
                    dot2 = area_speed(1259,filtered_data.iloc[j][8][1])                    
                    dot2_list.append(dot2)
                filtered_data['dot2'] = dot2_list
                area2 = filtered_data['crosswalk_distance'][filtered_data['dot2'].idxmin()]
                    
                for j in range(len(filtered_data)):
                    dot3 = area_speed(1888.5,filtered_data.iloc[j][8][1])                    
                    dot3_list.append(dot3)
                filtered_data['dot3'] = dot3_list
                area3 = filtered_data['crosswalk_distance'][filtered_data['dot3'].idxmin()]
                    
                for j in range(len(filtered_data)):
                    dot4 = area_speed(2518,filtered_data.iloc[j][8][1])                    
                    dot4_list.append(dot4)
                filtered_data['dot4'] = dot4_list
                area4 = filtered_data['crosswalk_distance'][filtered_data['dot4'].idxmin()]
                    
                for j in range(len(filtered_data)):                    
                    dot5 = area_speed(2939,filtered_data.iloc[j][8][1])                    
                    dot5_list.append(dot5)
                filtered_data['dot5'] = dot5_list
                area5 = filtered_data['crosswalk_distance'][filtered_data['dot5'].idxmin()]
                
                area_1.append(area1)
                area_2.append(area2)
                area_3.append(area3)
                area_4.append(area4)
                area_5.append(area5)
               
            del filtered_data  
        del car_data
        del crosswalk_point
        del crosswalk_person_frame
        del crosswalk_append
        del crosswalk_df
        del person_data
        print("crosswalk")
#%% 바로 위의 셀 구간 별 속도 지정을 어떻게 하지? 구간을 나눠서 평균? 
# crosswalk_person_frame
# walk_person_frame
# person_frame = walk_person_frame + crosswalk_person_frame
# person_frame = list(set(person_frame))

# y1 = 600
# y2 = 1200
# y3 = 1800
# y4 = 2400

#%% walk에 따른 차량 속도
        car_data = pd.DataFrame()
        person_data = data[data.class_number != 2]
        person_data = person_data.reset_index(drop = True)     
        car_data = data[data.class_number != 0]
        car_data = car_data.reset_index(drop = True)
        walk_person_frame = [] #사람들이 도보에 있는 프레임
        left_walk_point = [] #왼쪽 도보 변환 포인트
        right_walk_point = [] #오른쪽 도보 변환 포인트
        walk_append = [] #최종 데이터프레임에 합칠 데이터
        walk_df = pd.DataFrame() #최종 데이터프레임
        for i in range(10,14):        
            left_walk_p = cv2.perspectiveTransform(np.array([[np.array((spot_feat[i]), dtype = 'float32'), ]]), M)
            left_walk_p = (int(left_walk_p[0][0][0]), int(left_walk_p[0][0][1]))
            # crosswalk_p = spot_feat[i]
            left_walk_point.append(left_walk_p)
        
        left_walk_polygon = Polygon(left_walk_point)
        
        for i in range(len(person_data)):
            if left_walk_polygon.contains(Point(person_data['topview_point'][i])):
                walk_person_frame.append(person_data['frame'][i])
                
        
        for i in range(14,18):        
            right_walk_p = cv2.perspectiveTransform(np.array([[np.array((spot_feat[i]), dtype = 'float32'), ]]), M)
            right_walk_p = (int(right_walk_p[0][0][0]), int(right_walk_p[0][0][1]))
            # crosswalk_p = spot_feat[i]
            right_walk_point.append(right_walk_p)
        
        right_walk_polygon = Polygon(right_walk_point)
        
        for i in range(len(person_data)):
            if right_walk_polygon.contains(Point(person_data['topview_point'][i])):
                walk_person_frame.append(person_data['frame'][i])
        
        walk_person_frame = list(set(walk_person_frame))
        
        for i in range(len(car_data)):
            if car_data['frame'][i] in walk_person_frame:
                walk_append.append(car_data.iloc[i])
        
        if walk_append != []:   
            walk_df = pd.DataFrame(walk_append)
            walk_df = walk_df.sort_values(['id','frame'])
            walk_df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)


            
            walk_id = []
            walk_distance_list = []     
            for i in range(len(walk_df)):
                if walk_df['id'][i].item() not in walk_id:
                    walk_distance_list.append(0)
                    walk_id.append(walk_df['id'][i])
                else:
                    dist_ = math.sqrt((walk_df['topview_point'][i][0] - walk_df['topview_point'][i-1][0])**2 +
                                    (walk_df['topview_point'][i][1] - walk_df['topview_point'][i-1][1])**2)
                    dist_ = dist_/abs(walk_df['frame'][i] - walk_df['frame'][i-1])
                    walk_distance_list.append(dist_)
                    walk_total_list.append(dist_)
            walk_df = walk_df.assign(walk_distance = walk_distance_list)
            each_mean = walk_df['walk_distance'].mean()
            
            walk_distance = walk_distance + each_mean
            
        print("walk")
        
                
        print("{}".format(path))
        
        
        
        del walk_person_frame
        del left_walk_point
        del right_walk_point
        del walk_df
        del walk_append
        del person_data
#%% 구간 별 차량 속도
        # id_ = []
        # area1 = [] #500
        
        # for i in range(len(car_data)):
        #     if car_data['topview_point'][i] is near y=500

        
#%% SPOT 다 돌고 딕셔너리에 평균 거리 추가
    if now_spot == "A":
        distance_mean_A = distance / len(_path)
        crosswalk_distance_mean_A = crosswalk_distance / len(_path)
        walk_distance_mean_A = walk_distance / len(_path)
        only_car_total_list_A = only_car_total_list
        crosswalk_total_list_A = crosswalk_total_list
        walk_total_list_A = walk_total_list
        only_car_total_list_I = []
        crosswalk_total_list_I = []
        walk_total_list_I = []
    # if now_spot == "B":
    #     distance_mean_B = distance / len(_path)
    #     crosswalk_distance_mean_B = crosswalk_distance / len(_path)
    #     walk_distance_mean_B = walk_distance / len(_path)
    # if now_spot == "C":
    #     distance_mean_C = distance / len(_path)
    #     crosswalk_distance_mean_C = crosswalk_distance / len(_path)
    #     walk_distance_mean_C = walk_distance / len(_path)
    # if now_spot == "D":
    #     distance_mean_D = distance / len(_path)
    #     crosswalk_distance_mean_D = crosswalk_distance / len(_path)
    #     walk_distance_mean_D = walk_distance / len(_path)
    # if now_spot == "E":
    #     distance_mean_D = distance / len(_path)
    #     crosswalk_distance_mean_D = crosswalk_distance / len(_path)
    #     walk_distance_mean_D = walk_distance / len(_path)
    # if now_spot == "F":
    #     distance_mean_F = distance / len(_path)
    #     crosswalk_distance_mean_F = crosswalk_distance / len(_path)
    #     walk_distance_mean_F = walk_distance / len(_path)
    # if now_spot == "G":
    #     distance_mean_G = distance / len(_path)
    #     crosswalk_distance_mean_G = crosswalk_distance / len(_path)
    #     walk_distance_mean_G = walk_distance / len(_path)
    # if now_spot == "H":
    #     distance_mean_H = distance / len(_path)
    #     crosswalk_distance_mean_H = crosswalk_distance / len(_path)
    #     walk_distance_mean_H = walk_distance / len(_path)
    if now_spot == "I":
        distance_mean_I = distance / len(_path)
        crosswalk_distance_mean_I = crosswalk_distance / len(_path)
        walk_distance_mean_I = walk_distance / len(_path)
        only_car_total_list_I = only_car_total_list
        crosswalk_total_list_I = crosswalk_total_list
        walk_total_list_I = walk_total_list
        
    del only_car_total_list 
    del crosswalk_total_list
    del walk_total_list  
        # 함수 찍어봐야겠다.
    if now_spot == "A": 
        no_outlier_area_1 = remove_outliers(area_1)
        no_outlier_area_2 = remove_outliers(area_2)
        no_outlier_area_3 = remove_outliers(area_3)
        no_outlier_area_4 = remove_outliers(area_4)
        no_outlier_area_5 = remove_outliers(area_5)
        
        mean1 = np.mean(no_outlier_area_1) 
        mean2 = np.mean(no_outlier_area_2) 
        mean3 = np.mean(no_outlier_area_3)
        mean4 = np.mean(no_outlier_area_4) 
        mean5 = np.mean(no_outlier_area_5) 
        mean1_shifted = (A_area[0],mean1)
        mean2_shifted = (A_area[1],mean2)
        mean3_shifted = (A_area[2],mean3)
        mean4_shifted = (A_area[3],mean4)
        mean5_shifted = (A_area[4],mean5)
 
        plt.scatter(*mean1_shifted, color='black', label='List 1')
        plt.scatter(*mean2_shifted, color='black', label='List 2')
        plt.scatter(*mean3_shifted, color='black', label='List 3')
        plt.scatter(*mean4_shifted, color='black', label='List 4')
        plt.scatter(*mean5_shifted, color='black', label='List 5')
        plt.title("SPOT A")

                
    if now_spot == "I":
        no_outlier_area_1 = remove_outliers(area_1)
        no_outlier_area_2 = remove_outliers(area_2)
        no_outlier_area_3 = remove_outliers(area_3)
        no_outlier_area_4 = remove_outliers(area_4)
        no_outlier_area_5 = remove_outliers(area_5)
        
        mean1 = np.mean(no_outlier_area_1) 
        mean2 = np.mean(no_outlier_area_2) 
        mean3 = np.mean(no_outlier_area_3)
        mean4 = np.mean(no_outlier_area_4) 
        mean5 = np.mean(no_outlier_area_5) 
        mean1_shifted = (I_area[0],mean1)
        mean2_shifted = (I_area[1],mean2)
        mean3_shifted = (I_area[2],mean3)
        mean4_shifted = (I_area[3],mean4)
        mean5_shifted = (I_area[4],mean5)
        
        # plt.xlim(-4,1)
        plt.scatter(*mean1_shifted, color='black', label='List 1')
        plt.scatter(*mean2_shifted, color='black', label='List 2')
        plt.scatter(*mean3_shifted, color='black', label='List 3')
        plt.scatter(*mean4_shifted, color='black', label='List 4')
        plt.scatter(*mean5_shifted, color='black', label='List 5')
        plt.title("SPOT B")
 
    






#%% 차량만 있을 때 spot별 차량 속도 text 파일에 저장  
spot_distance_onlycar = {'A': distance_mean_A, 'B': distance_mean_I}
spot_distance_crosswalk = {'A': crosswalk_distance_mean_A, 'B': crosswalk_distance_mean_I}
spot_distance_walk = {'A': walk_distance_mean_A, 'B': walk_distance_mean_I}
# spot_distance_onlycar = {'A': distance_mean_A, 'B': distance_mean_B, 'C': distance_mean_C, 'D': distance_mean_D, 'E': distance_mean_E, 'F': distance_mean_F, 'G': distance_mean_G, 'H': distance_mean_H, 'I': distance_mean_I}
# spot_distance_crosswalk = {'A': crosswalk_distance_mean_A, 'B': crosswalk_distance_mean_B, 'C': crosswalk_distance_mean_C, 'D': crosswalk_distance_mean_D, 'E': crosswalk_distance_mean_E, 'F': crosswalk_distance_mean_F, 'G': crosswalk_distance_mean_G, 'H': crosswalk_distance_mean_H, 'I': crosswalk_distance_mean_I}
# spot_distance_walk = {'A': walk_distance_mean_A, 'B': walk_distance_mean_B, 'C': walk_distance_mean_C, 'D': walk_distance_mean_D, 'E': walk_distance_mean_E, 'F': walk_distance_mean_F, 'G': walk_distance_mean_G, 'H': walk_distance_mean_H, 'I': walk_distance_mean_I}



file_path = "onlycar.txt"
with open(file_path, 'w') as file:
    for spot, distance in spot_distance_onlycar.items():
        file.write(f"{spot}: {distance}\n")

file_path = "crosswalk.txt"
with open(file_path, 'w') as file:
    for spot, distance in spot_distance_crosswalk.items():
        file.write(f"{spot}: {distance}\n")

file_path = "walk.txt"
with open(file_path, 'w') as file:
    for spot, distance in spot_distance_walk.items():
        file.write(f"{spot}: {distance}\n")
            

        


# ##########################################################################
#%% only_car
# categories = list(spot_distance_onlycar.keys())
# values = list(spot_distance_onlycar.values())

# # 그래프 그리기
# plt.bar(categories, values)
# plt.xlabel('SPOT')
# plt.ylabel('PIXEL DISTANCE PER FRAME')
# plt.title('only_car')
# plt.grid(True)
# plt.show()

# #%% crosswalk
# categories = list(spot_distance_crosswalk.keys())
# values = list(spot_distance_crosswalk.values())

# # 그래프 그리기
# plt.bar(categories, values)
# plt.xlabel('SPOT')
# plt.ylabel('PIXEL DISTANCE PER FRAME')
# plt.title('crosswalk')
# plt.grid(True)
# plt.show()

#%% walk
# categories = list(spot_distance_walk.keys())
# values = list(spot_distance_walk.values())

# # 그래프 그리기
# plt.bar(categories, values)
# plt.xlabel('SPOT')
# plt.ylabel('PIXEL DISTANCE PER FRAME')
# plt.title('sidewalk')
# plt.grid(True)
# plt.show()    


#%% 박스플롯 (이상치 제거한 채)
no_outlier_only_car_total_list_A = remove_outliers(only_car_total_list_A)
no_outlier_only_car_total_list_I = remove_outliers(only_car_total_list_I)

no_outlier_crosswalk_total_list_A = remove_outliers(crosswalk_total_list_A)
no_outlier_crosswalk_total_list_I = remove_outliers(crosswalk_total_list_I)

no_outlier_walk_total_list_A = remove_outliers(walk_total_list_A)
no_outlier_walk_total_list_I = remove_outliers(walk_total_list_I)


plt.boxplot([no_outlier_only_car_total_list_A, no_outlier_only_car_total_list_I], showfliers=False, showmeans=True, meanline=True, medianprops={'visible': False})
plt.xticks([1, 2], ['A', 'B'])  # X 축 레이블 설정
plt.ylabel('Pixel Distance Per Frame')
plt.title('ONLY CAR DATA')
plt.grid(True)
plt.show()
print("ONLY CAR MEAN \"A\" Spot : {}".format(sum(only_car_total_list_A)/len(only_car_total_list_A)))
print("ONLY CAR MEAN \"B\" Spot : {}".format(sum(only_car_total_list_I)/len(only_car_total_list_I)))


plt.boxplot([no_outlier_crosswalk_total_list_A, no_outlier_crosswalk_total_list_I], showfliers=False, showmeans=True, meanline=True, medianprops={'visible': False})
plt.xticks([1, 2], ['A', 'B'])  # X 축 레이블 설정
plt.ylabel('Pixel Distance Per Frame')
plt.title('CROSSWALK DATA')
plt.grid(True)
plt.show()
print("CROSSWALK CAR MEAN \"A\" Spot : {}".format(sum(crosswalk_total_list_A)/len(crosswalk_total_list_A)))
print("CROSSWALK CAR MEAN \"B\" Spot : {}".format(sum(crosswalk_total_list_I)/len(crosswalk_total_list_I)))


plt.boxplot([no_outlier_walk_total_list_A, no_outlier_walk_total_list_I], showfliers=False, showmeans=True, meanline=True, medianprops={'visible': False})
plt.xticks([1, 2], ['A', 'B'])  # X 축 레이블 설정
plt.ylabel('Pixel Distance Per Frame')
plt.title('SIDEWALK DATA')
plt.grid(True)
plt.show()
print("WALK CAR MEAN \"A\" Spot : {}".format(sum(walk_total_list_A)/len(walk_total_list_A)))
print("WALK CAR MEAN \"B\" Spot : {}".format(sum(walk_total_list_I)/len(walk_total_list_I)))

#%% A구간만, I구간만
plt.boxplot([only_car_total_list_A, crosswalk_total_list_A, walk_total_list_A], showfliers=False, showmeans=True, meanline=True, medianprops={'visible': False})
plt.xticks([1, 2, 3], ['ONLY CAR', 'CROSSWALK', 'WALK'])  # X 축 레이블 설정
plt.ylabel('Pixel Distance Per Frame')
plt.title('SPOT A')
plt.grid(True)
plt.show()

plt.boxplot([only_car_total_list_I, crosswalk_total_list_I, walk_total_list_I], showfliers=False, showmeans=True, meanline=True, medianprops={'visible': False})
plt.xticks([1, 2, 3], ['ONLY CAR', 'CROSSWALK', 'WALK'])  # X 축 레이블 설정
plt.ylabel('Pixel Distance Per Frame')
plt.title('SPOT B')
plt.grid(True)
plt.show()

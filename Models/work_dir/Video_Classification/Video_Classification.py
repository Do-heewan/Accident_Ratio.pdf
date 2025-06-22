import numpy as np
import json
import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

video_name = "" # video name
point_of_view = "1인칭"

real_categories_ids_1st = {
    0 : "직선 도로", 1 : "사거리 교차로(신호등 없음)", 2 : "사거리 교차로(신호등 있음)", 3 : "T자형 교차로",
    4 : "차도와 차도가 아닌 장소", 5 : "주차장(또는 차도가 아닌 장소)", 6 : "회전교차로", 7 : "횡단보도(신호등 없음)",
    8 : "횡단보도(신호등 있음)", 9 : "횡단보도 없음", 10 : "횡단보도(신호등 없음) 부근", 11 : "횡단보도(신호등 있음) 부근",
    12 : "육교 및 지하도 부근", 13 : "고속도로(자동차 전용도로)포함", 14 : "자전거 도로"
}

real_categories_ids_2nd = {
    0 : "추돌 사고", 1 : "차로 감소 도로 (합류)", 2 : "열린 문 접촉사고", 3 : "역주행 사고(중앙선 침범)",
    4 : "이면도로 교행 사고", 5 : "추월 사고", 6 : "차로변경(진로변경)", 7 : "안전지대 통과 사고",
    8 : "정차 후 출발 사고", 9 : "긴급자동차 사고", 10 : "동일폭 도로", 11 : "대로와 소로", 12 : "일시정지 표시가 한쪽방향에만 있음",
    13 : "일방통행 표지가 한쪽방향에만 있음", 14 : "교차로 내 진로변경", 15 : "2개 차량이 나란히 통행 가능한 차로폭",
    16 : "좌/우회전 각도가 90도 미만", 17 : "2개 차로 동시 우회전", 18 : "상대 차량이 측면 방향에서 진입",
    19 : "일시정지 표지가 한쪽방향에만 있음", 20 : "차도가 아닌 장소에서 차도로 진입", 21 : "차도에서 차도가 아닌 장소로 진입",
    22 : "주차구역과 통로", 23 : "회전차로 1차로형", 24 : "회전차로 2차로형", 25 : "직선 도로", 26 : "자동차 교차로 통과 후",
    27 : "자동차 교차로 통과 전", 28 : "교차로(대로와 소로)", 29 : "교차로(동일폭 도로)", 30 : "단일로(중앙선 없음)",
    31 : "보도와 차도", 32 : "보도와 차도(구분 없음)", 33 : "보행자 전용도로", 34 : "차도가 아닌 장소", 
    35 : "자동차 횡단보도 통과 후", 36 : "자동차 횡단보도 통과 전", 37 : "주행차로와 추월차로", 38 : "주행차로와 주행차로",
    39 : "합류", 40 : "주 (정)차", 41 : "추돌", 42 : "낙하물", 43 : "보행자", 44 : "갓길 진로변경", 45 : "마주보는 이륜차와 자동차간의 사고",
    46 : "자동차와 이륜차가 나란히 통행 가능한 차로폭", 47 : "정체도로", 48 : "유턴구역", 49 : "상대차량이 맞은편 방향에서 진입",
    50 : "노면 표시 위반사고", 51 : "인도에서 차도가 아닌 장소로 진입", 52 : "마주보는 자전거와 자동차간의 사고",
    53 : "선행차량과 후행차량", 54 : "자전거 횡단도로", 55 : "자동차와 자전거가 나란히 통행 가능한 차로폭",
    56 : "자전거 전용도로", 57 : "자전거 전용차로", 58 : "자전거 우선도로", 59 : "비보호좌회전표지있음"
}

real_categories_ids_3rd = {
    0 : "선행자동차(1차사고차량)를 추돌", 1 : "후행 추돌", 2 : "주(정)차", 3 : "본선에서 직진",
    4 : "직진", 5 : "[마주보며] 직진", 6 : "선행 직진", 7 : "중앙선 침범 추월(후방)",
    8 : "실선 추월", 9 : "후행 직진", 10 : "동시 차로변경(진로변경)", 11 : "정체차로에서 대기 중 진로변경(측면 충돌)", 12 : "후행 직진(안전지대 벗어나기 전)",
    13 : "후행 직진(안전지대 벗어난 후)", 14 : "정차 후 출발", 15 : "선행 진로변경",
    16 : "오른쪽에서 직진", 17 : "오른쪽에서 직진(후진입)", 18 : "오른쪽에서 직진(선진입)",
    19 : "왼쪽 도로에서 직진", 20 : "오른쪽 도로에서 직진", 21 : "우회전",
    22 : "우회전(후진입)", 23 : "우회전(선진입)", 24 : "오른쪽 도로에서 좌회전", 25 : "대로에서 직진", 26 : "대로에서 직진(후진입)",
    27 : "대로에서 직진(선진입)", 28 : "소로에서 직진(좌측도로)", 29 : "소로에서 직진(우측도로)", 30 : "소로에서 우회전",
    31 : "소로에서 우회전(후진입)", 32 : "소로에서 우회전(선진입)", 33 : "대로에서 우회전", 34 : "대로에서 우회전(후진입)", 
    35 : "대로에서 우회전(선진입)", 36 : "소로에서 좌회전", 37 : "표지가 없는 도로에서 직진", 38 : "일시정지 위반 직진(좌측도로)",
    39 : "일시정지 위반 직진(우측도로)", 40 : "일시정지 위반 우회전", 41 : "일시정지 위반 직진", 42 : "일시정지 위반 좌회전", 43 : "직진(교차로 내 진로변경)", 44 : "후행 직진(차로 좌측)", 45 : "후행 직진(차로 우측)",
    46 : "우회전(오른쪽 차로)", 47 : "[녹색신호] 직진", 48 : "[녹색신호] 직진, [적색신호] 충돌", 49 : "[황색신호] 직진, [적색신호] 충돌",
    50 : "[황색신호] 직진", 51 : "[적색신호] 직진", 52 : "[녹색좌회전신호] 좌회전, [적색신호] 충돌",
    53 : "[황색신호] 좌회전, [적색신호] 충돌", 54 : "소로에서 직진", 55 : "대로에서 좌회전",
    56 : "차도에서 직진", 57 : "통로 직진", 58 : "회전(회전 2차로)", 59 : "회전교차로 진입(1차로 → 회전 2차로)", 60 : "회전(회전 1차로)",
    61 : "진로변경(회전 1차로 → 회전 2차로)", 62 : "횡단보도 횡단", 63 : "[적색신호] 횡단 시작, [적색신호] 충격", 64 : "[적색신호] 횡단 시작, 녹색(녹색점멸 포함) 신호 충격",
    65 : "[녹색신호] 횡단 시작, 녹색(녹색점멸 포함) 신호 충격", 66 : "녹색(녹색점멸) 신호에 횡단 시작, [적색신호] 충격", 67 : "[적색신호] 횡단 시작, [녹색신호] 충격",
    68 : "[녹색신호] 횡단 시작, [적색신호] 충격", 69 : "[녹색점멸신호] 횡단 시작, [적색신호] 충격", 70 : "[녹색점멸신호] 횡단 시작, [녹색점멸신호] 충격",
    71 : "소로 횡단", 72 : "대로 횡단", 73 : "횡단", 74 : "보도 보행", 75 : "차도 보행(인도에서 공사 또는 퇴적토 등)", 76 : "차도 보행(이유없이 보, 차도 경계 1m 이내)",
    77 : "차도 보행(이유없이 보, 차도 경계 1m 이상)", 78 : "차도에서 놀이, 누워있음 또는 유사한 행위", 79 : "차도의 가장자리로 보행",
    80 : "차도의 중앙부분 보행", 81 : "보행자 전용도로에서 보행", 82 : "후진하는 차 뒷부분의 3m 이내 거리에서 횡단", 83 : "후진하는 차 뒷부분의 3m 이상 거리에서 횡단",
    84 : "횡단보도 10m 이내에서 횡단", 85 : "[녹색신호] 횡단보도 10m 이내에서 횡단", 86 : "[적색신호] 횡단보도 10m 이내에서 횡단",
    87 : "육교 및 지하도 10m 이내에서 횡단", 88 : "추월차로에서 직진", 89 : "차로에서 주 (정)차한 차량을 추돌", 90 : "갓길에서 주(정)차한 차량을 추돌",
    91 : "선행 차량 추돌", 92 : "낙화물에 의해 충격,회피중", 93 : "갓길로 진로변경", 94 :"중앙선을 침범하여 반대차로 진행",
    95 : "후행 추월(추월 금지 장소)", 96 : "급접거리 추월(점선 중앙선)", 97 : "진로변경", 98 : "직진(측면 충돌)", 99 : "유턴",
    100 : "후행 직전", 101 : "추월", 102 : "오른쪽에서 직진(동시진입)", 103 : "왼쪽에서 직진(동시진입)", 104 : "왼쪽에서 직진(선진입)",
    105 : "왼쪽에서 직진(후진입)", 106 : "[마주보며] 좌회전", 107 : "왼쪽에서 직진", 108 : "오른쪽에서 좌회전", 109 : "왼쪽에서 좌회전",
    110 : "오른쪽에서 우회전(동시진입)", 111 : "오른쪽에서 우회전(선진입)", 112 : "오른쪽에서 우회전(후진입)", 113 : "대로에서 직진(동시진입)",
    114 : "소로에서 직진(동시진입)", 115 : "소로에서 직진(선진입)", 116 : "소로에서 직진(후진입)", 117 : "소로에서 우회전(동시진입)",
    118 : "대로에서 우회전(동시진입)", 119 : "표지가 없는 도로에서 좌회전", 120 : "표지가 없는 도로에서 우회전", 121 : "일방통행 위반 직진",
    122 : "동일차로에서 후행 직진", 123 : "동일차로에서 선행 직진", 124 : "동일차로에서 선행 우회전", 125 : "동일차로에서 선행 좌회전",
    126 : "동일차로에서 추월 우회전", 127 : "동일차로에서 추월 직진", 128 : "차량좌측에서 후행 직진(차량 우측에서 후행 직진)", 129 : "차량우측에서 선행 좌회전(차량 좌측에서 선행 우회전)",
    130 : "정체도로 사이에서 직진", 131 : "[적색신호] 좌회전", 132 : "[황색신호] 좌회전", 133 : "[녹색신호] 직진(교차로내 진로 변경)",
    134 : "중앙선 침범 추월", 135 : "[녹색좌회전신호] 좌회전", 136 : "상시유턴구역에서 유턴", 137 : "신호에 따른 유턴", 138 : "우회전(좌측도로)",
    139 : "유턴(선행)", 140 : "동시 유턴(선행)", 141 : "급 유턴(후행)", 142 : "동시 유턴(후행)", 143 : "맞은편 우회전", 144 : "[녹색직진.좌회전 신호] 후행 직진(좌회전 노면표시차로)",
    145 : "[녹색직진.좌회전 신호] 선행 좌회전(직진좌회전 노면표시차로)", 146 : "좌회전(직진 노면표시차로)", 147 : "직진(직진.좌회전 노면표시차로)",
    148 : "추월 우회전(직진 노면표시차로)", 149 : "직진(직진.우회전 노면표시차로)", 150 : "차도가 아닌 장소에서 차도로 진입", 151 : "차도에서 차도가 아닌 장소로 진입",
    152 : "인도에서 차도가 아닌 장소로 진행", 153 : "교차로 내 회전", 154 : "회전교차로 진입", 155 : "회전교차로 진입(2차로 → 회전 2차로)",
    156 : "회전교차로 대진입", 157 : "차도 횡단", 158 : "[자동차를 마주보며] 역주행", 159 : "자전거 횡단도로 횡단", 160 : "[자동차를 마주보며] 직진",
    161 : "[자동차를 마주보며] 좌회전", 162 : "오른쪽에서 우회전", 163 : "자동차 좌측에서 대우회전", 164 : "자동차 좌측에서 소우회전"
}

real_categories_ids_4th = {
    0 : "선행 자동차(1차사고차량)", 1 : "선행 직진", 2 : "후행 추돌", 3 : "차로 감소 도로에서 본선으로 합류",
    4 : "선행 자동차(정차중 문열림)", 5 : "중앙선 침범 직진", 6 : "[마주보며] 직진", 7 : "추월(실선 중앙선)",
    8 : "급접거리 추월(점선 중앙선)", 9 : "중앙선 침범 추월(전방)", 10 : "선행 진로변경", 11 : "동시 차로변경(진로변경)", 12 : "직진(측면 충돌)",
    13 : "추월", 14 : "중앙선 왼쪽 통행(긴급자동차)", 15 : "선행 진로변경(긴급자동차)",
    16 : "추월(긴급자동차)", 17 : "후행 직진(긴급자동차)", 18 : "왼쪽에서 직진",
    19 : "왼쪽에서 직진(선진입)", 20 : "왼쪽에서 직진(후진입)", 21 : "[마주보며] 좌회전",
    22 : "오른쪽 도로에서 좌회전", 23 : "왼쪽 도로에서 좌회전", 24 : "직진", 25 : "직진(선진입)", 26 : "직진(후진입)",
    27 : "소로에서 직진", 28 : "소로에서 좌회전", 29 : "소로에서 직진(선진입)", 30 : "소로에서 직진(후진입)",
    31 : "대로에서 좌회전(우측도로)", 32 : "대로에서 좌회전(좌측도로)", 33 : "대로에서 직진", 34 : "대로에서 직진(선진입)", 
    35 : "대로에서 직진(후진입)", 36 : "대로에서 좌회전", 37 : "일시정지 위반 직진", 38 : "일시정지 위반 좌회전",
    39 : "표지가 없는 도로에서 좌회전(우측도로)", 40 : "표지가 없는 도로에서 좌회전(좌측도로)", 41 : "표지가 없는 도로에서 직진", 42 : "표지가 없는 도로에서 우회전", 43 : "표지가 없는 도로에서 좌회전", 44 : "일방통행 위반 직진", 45 : "우회전",
    46 : "선행 좌회전(차로 우측)", 47 : "선행 우회전(차로 좌측)", 48 : "선행 좌회전", 49 : "선행 우회전",
    50 : "우회전(왼쪽 차로)", 51 : "소로에서 진입(긴급자동차)", 52 : "[적색신호] 직진",
    53 : "[녹색신호] 직진", 54 : "[적색신호] 좌회전", 55 : "[녹색좌회전신호] 좌회전",
    56 : "[황색신호] 좌회전", 57 : "소로에서 우회전", 58 : "대로에서 우회전", 59 : "일시정지 위반 우회전", 60 : "표지가 없는 도로에서 좌회전(좌회전)",
    61 : "차도가 아닌 장소에서 중앙선 침범 진입", 62 : "차도가 아닌 장소에서 우회전 진입", 63 : "차도가 아닌 장소로 중앙선 침범 진입", 64 : "주차구역에서 직진 출자",
    65 : "주차구역에서 후진 출자", 66 : "교차로 내 회전", 67 : "진로변경(회전 1차로 → 회전 2차로)",
    68 : "회전교차로 진입(2차로 → 회전 2차로)", 69 : "회전교차로 대진입", 70 : "회전교차로 진입",
    71 : "이륜차 횡단보도 횡단", 72 : "[녹색직진신호] 직진", 73 : "[녹색우회전신호] 우회전", 74 : "[황색신호] 직진", 75 : "[황색우회전신호] 우회전", 76 : "[적색우회전신호] 우회전",
    77 : "소로 주행<좌(우)회전 동일>", 78 : "대로 주행<좌(우)회전 동일>", 79 : "교차로 주행<좌(우)회전 동일>",
    80 : "보도 침범", 81 : "차도 주행", 82 : "보행자 전용도로로 주행", 83 : "후진",
    84 : "[적색신호] 직진 또는 좌회전", 85 : "[황색신호] 직진 또는 좌회전", 86 : "[녹색신호] 직진 또는 [녹색좌회전신호] 좌회전",
    87 : "주행차로에서 추월차로로 진로변경", 88 : "추월차로에서 주행차로로 진로변경", 89 : "주행차로에서 주행차로로 변경", 90 : "본선으로 합류",
    91 : "차로에서 주(정)차", 92 : "갓길에서 주(정)차", 93 : "적재물 등의 낙하", 94 :"이유 없는 고속도로 보행",
    95 : "이유 없는 고속도로 보행(고장,사무, 공무 등)", 96 : "갓길 직진", 97 : "중앙선을 침범하여 반대차로 진행", 98 : "좌회전", 99 : "후행 추월(추월 금지 장소)",
    100 : "진로변경", 101 : "후행 직진", 102 : "정제차로에서 대기 중 진로변경(측면 충돌)", 103 : "후행 직진(안전지대 벗어나기 전)", 104 : "후행 직진(안전지대 벗어난 후)",
    105 : "주정차", 106 : "왼쪽 문 열림", 107 : "오른쪽 문 열림", 108 : "정차 후 출발", 109 : "왼쪽에서 직진(동시진입)",
    110 : "오른쪽에서 직진(동시진입)", 111 : "오른쪽에서 우회전(동시진입)", 112 : "오른쪽에서 직진(후진입)", 113 : "오른쪽에서 우회전(후진입)",
    114 : "오른쪽에서 직진(선진입)", 115 : "오른쪽에서 우회전(선진입)", 116 : "오른쪽에서 좌회전", 117 : "왼쪽에서 좌회전",
    118 : "오른쪽에서 직진", 119 : "소로에서 우회전(동시진입)", 120 : "소로에서 우회전(후진입)", 121 : "소로에서 우회전(선진입)",
    122 : "대로에서 직진(동시진입)", 123 : "대로에서 우회전(동시진입)", 124 : "대로에서 우회전(후진입)", 125 : "대로에서 우회전(선진입)",
    126 : "소로에서 직진(동시진입)", 127 : "직진(교차로 내 진로변경)", 128 : "동일차로에서 선행 좌회전", 129 : "동일차로에서 선행 우회전",
    130 : "동일차로에서 후행 좌회전", 131 : "동일차로에서 후행 우회전", 132 : "동일차로에서 후행 직진", 133 : "동일차로에서 추월 직진",
    134 : "동일차로에서 선행 직진", 135 : "이륜차 우측에서 선행 좌회전(이륜차 좌측에서 선행 우회전)", 136 : "이륜차 좌측에서 후행 직진(이륜차 우측에서 후행 직진)", 137 : "[녹색신호] 직진(교차로내진로변경)", 138 : "중앙선 침범 추월",
    139 : "상시유턴구역에서 유턴", 140 : "신호에 따른 유턴", 141 : "우회전(좌측도로)", 142 : "급 유턴(후행)", 143 : "동시 유턴(후행)", 144 : "유턴(선행)",
    145 : "동시 유턴(선행)", 146 : "맞은편 우회전", 147 : "[녹색직진.좌회전 신호] 선행 좌회전(직진좌회전 노면표시차로)",
    148 : "[녹색직진.좌회전 신호] 후행 직진(좌회전 노면표시차로)", 149 : "직진(직진.좌회전 노면표시차로)", 150 : "좌회전(직진 노면표시차로)", 151 : "직진(직진.우회전 노면표시차로)"
}

def directory_classifier_restrict_1(data_dir, num_class = 15):
    txt_constrict = 'Classifier_restrict_vector_1st.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        cluster = line_split[0]
        #Key generation
        key = cluster
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[1:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def directory_classifier_restrict_2(data_dir, num_class = 60):
    txt_constrict = 'Classifier_restrict_vector_2nd.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        #Key generation
        key = '{}{}'.format(line_split[0],line_split[1])
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[2:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def directory_classifier_restrict_3(data_dir, num_class = 165):
    txt_constrict = 'Classifier_restrict_vector_3rd.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        #Key generation
        key = '{}{}{}'.format(line_split[0],line_split[1],line_split[2])
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[3:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def directory_classifier_restrict_4(data_dir, num_class = 152):
    txt_constrict = 'Classifier_restrict_vector_4th.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        #Key generation
        key = '{}{}{}{}'.format(line_split[0],line_split[1],line_split[2],line_split[3])
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[4:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def inference_recognizer(model,
                        video_path,
                        use_frames=False,
                        outputs=None,
                        as_tensor=True):
    if isinstance(outputs, str):
        outputs = (outputs, )
    assert outputs is None or isinstance(outputs, (tuple, list))

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)

    modality = cfg.data.test.get('modality', 'RGB')

    num_frames = len(os.listdir(video_path))

    data = dict(
        frame_dir=video_path,
        total_frames=num_frames,
        label=-1,
        start_index=0,
        filename_tmpl=None,
        modality=modality)

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    imgs = data['imgs']
    # forward the model
    with torch.no_grad():
        scores, scores_pro = model.forward_test_with_constraint(imgs)
    return scores, scores_pro

work_dir = "" # work directory path
video_path = work_dir + "datasets/Video_Data/" + video_name + "/" # Video path

cluster_path = work_dir + "video_classification/classifier/" # Classifier path

model_path =  work_dir + "video_classification/models/" # Model path

if (point_of_view == "1인칭"):
    # Classifier 1
    config_path_1 = model_path + "/For1_view1/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_1 = model_path + '/For1_view1/for1_view1.pth'

    # Classifier 2
    config_path_2 = model_path + "/For2_view1/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_2 = model_path + '/For2_view1/for2_view1.pth'

    # Classifier 3
    config_path_3 = model_path + "/For3_view1/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_3 = model_path + '/For3_view1/for3_view1.pth'

    # Classifier 4
    config_path_4 = model_path + "/For4_view1/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_4 = model_path + '/For4_view1/for4_view1.pth'

else:  # 3인칭
    # Classifier 1
    config_path_1 = model_path + "/For1_view3/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_1 = model_path + '/For1_view3/for1_view3.pth'

    # Classifier 2
    config_path_2 = model_path + "/For2_view3/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_2 = model_path + '/For2_view3/for2_view3.pth'

    # Classifier 3
    config_path_3 = model_path + "/For3_view3/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_3 = model_path + '/For3_view3/for3_view3.pth'

    # Classifier 4
    config_path_4 = model_path + "/For4_view3/slowfast_r50_4x16x1_256e_kinetics400_rgb.py"
    checkpoint_path_4 = model_path + '/For4_view3/for4_view3.pth'

# 분류기 별 클러스터 지정
cluster_dict_1 = directory_classifier_restrict_1(cluster_path) # Classifier 1
cluster_dict_2 = directory_classifier_restrict_2(cluster_path) # Classifier 2
cluster_dict_3 = directory_classifier_restrict_3(cluster_path) # Classifier 3
cluster_dict_4 = directory_classifier_restrict_4(cluster_path) # Classifier 4

# 모델 초기화
model_1 = init_recognizer(config_path_1, checkpoint_path_1, device='cpu', use_frames=True)
model_2 = init_recognizer(config_path_2, checkpoint_path_2, device='cpu', use_frames=True)
model_3 = init_recognizer(config_path_3, checkpoint_path_3, device='cpu', use_frames=True)
model_4 = init_recognizer(config_path_4, checkpoint_path_4, device='cpu', use_frames=True)

# 사고 분류 추론
# 1st 사고 장소 분류
cls_scores_1, cls_scores_softmax_1 = inference_recognizer(model_1, video_path, outputs='cls_score', as_tensor=True)
pred_class_id_1 = np.argmax(cls_scores_softmax_1[0])  # 제약 없이 가장 높은 확률의 클래스 선택

# 2nd 사고 특성 분류
key_2 = f"{0}{pred_class_id_1}" 
constraint_2 = cluster_dict_2.get(key_2, np.ones(60))  # 첫 번째 결과에 기반한 제약조건

cls_scores_2, cls_scores_softmax_2 = inference_recognizer(model_2, video_path, outputs='cls_score', as_tensor=True)
constrained_scores_2 = cls_scores_softmax_2 * constraint_2
pred_class_id_2 = np.argmax(constrained_scores_2[0])  # 제약조건이 적용된 예측

# 3rd 객체 A 행동 분류
key_3 = f"{0}{pred_class_id_1}{pred_class_id_2}"
constraint_3 = cluster_dict_3.get(key_3, np.ones(165))

cls_scores_3, cls_scores_softmax_3 = inference_recognizer(model_3, video_path, outputs='cls_score', as_tensor=True)
constrained_scores_3 = cls_scores_softmax_3 * constraint_3
pred_class_id_3 = np.argmax(constrained_scores_3[0])

# 4th 객체 B 행동 분류
key_4 = f"{0}{pred_class_id_1}{pred_class_id_2}{pred_class_id_3}"
constraint_4 = cluster_dict_4.get(key_4, np.ones(152))

cls_scores_4, cls_scores_softmax_4 = inference_recognizer(model_4, video_path, outputs='cls_score', as_tensor=True)
constrained_scores_4 = cls_scores_softmax_4 * constraint_4
pred_class_id_4 = np.argmax(constrained_scores_4[0])

# 결과 매핑
result_1st = real_categories_ids_1st[pred_class_id_1]
result_2nd = real_categories_ids_2nd[pred_class_id_2]
result_3rd = real_categories_ids_3rd[pred_class_id_3]
result_4th = real_categories_ids_4th[pred_class_id_4]

# 결과 저장 및 출력 확장
results = []
results.append({
    'video_path': video_path,
    'video_name': video_name,
    'accident_place': result_1st,
    'accident_place_feature': result_2nd,
    'object_A': result_3rd,
    'object_B': result_4th,
})

print(f"1번 결과 : {pred_class_id_1}, 2번 결과 : {pred_class_id_2}, 3번 결과 : {pred_class_id_3}, 4번 결과 : {pred_class_id_4}")
print(f"1번 결과 : {result_1st}, 2번 결과 : {result_2nd}, 3번 결과 : {result_3rd}, 4번 결과 : {result_4th}")

# 결과 출력 형식 개선
print(f"\n===== 분석 결과: {video_name} =====")
print(f"사고 장소: {result_1st} (신뢰도: {cls_scores_softmax_1[0][pred_class_id_1]})")
print(f"사고 특성: {result_2nd} (신뢰도: {constrained_scores_2[0][pred_class_id_2]})")
print(f"객체 A 행동: {result_3rd} (신뢰도: {constrained_scores_3[0][pred_class_id_3]})")
print(f"객체 B 행동: {result_4th} (신뢰도: {constrained_scores_4[0][pred_class_id_4]})")
print("================================\n")
# print(f"1번 신뢰도 : {cls_scores_softmax_1}")
# print(f"2번 신뢰도 : {cls_scores_softmax_2}")
# print(f"3번 신뢰도 : {cls_scores_softmax_3}")
# print(f"4번 신뢰도 : {cls_scores_softmax_4}")

print(f"{video_name} : {results[0]['accident_place']}, {results[0]['accident_place_feature']}, {results[0]['object_A']}, {results[0]['object_B']}")

# Create output directory if it doesn't exist
output_dir = os.path.join(work_dir, "datasets/Results")
os.makedirs(output_dir, exist_ok=True)

# Save results to a JSON file
json_filename = os.path.join(output_dir, f"{video_name}_classification.json")
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {json_filename}")
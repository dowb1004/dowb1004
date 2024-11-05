import streamlit as st
import torch
from PIL import Image
import os
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import models, transforms
import pandas as pd
from openai import OpenAI
from matplotlib import pyplot as plt
import shutil
import json


st.set_page_config(
    layout="wide",
    page_title="두피케어 제품 추천 서비스",
    page_icon=".data/images/monsterball.png"
)

st.markdown("""
<style>
img { 
    max-height: 300px;
}
.streamlit-expanderContent div {
    display: flex;
    justify-content: center;
    font-size: 20px;
}
[data-testid="stExpanderToggleIcon"] {
    visibility: hidden;
}
.streamlit-expanderHeader {
    pointer-events: none;
}
[data-testid="StyledFullScreenButton"] {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


st.subheader("🤠 딥러닝 통한 두피진단과 두피케어 제품 추천 서비스")
st.text("2024.11.22, DeepRoot, 김성환, 김준호, 이혜진, 전민정")


scalp_example = {
    "name": "고민남",
    "sex": "남",
    "age": "30대",
    "symptom": ["탈모"],
    "location": "정수리(TH)",
    "bindo": "1일 1회",
    "pum": "4~6회/연",
    "yaumsac": "4~6회/연",
    "mobal": ["염색 모발"],
    "product": ["샴푸", "린스"],
    "hope": "예",
    "variety": ["샴푸", "린스/컨디셔너", "샴푸바/드라이샴푸"],
    "gorau": "세정력"
}

sex_emoji_dict = {
    "남": "⚪",
    "여": "✊"
}

age_emoji_dict = {
    "0대": "⚪",
    "10대": "✊",
    "20대": "🧚",
    "30대": "⚪",
    "40대": "✊",
    "50대": "🧚",
    "60대": "⚪",
    "70대": "✊",
    "80대": "🧚",
    "90대": "⚪",
    "100대": "✊"
}

symptom_emoji_dict = {
    "비듬": "🐲",
    "미세각질": "🤖",
    "모낭사이홍반": "🧚",
    "모낭홍반농포": "🍃",
    "피지과다": "🔮",
    "탈모": "❄️"
}

location_emoji_dict = {
    "정수리(TH)": "⚪",
    "좌측두(LH)": "✊",
    "우측두(RH)": "🧚",
    "후두부(BH)": "🧚"
}


bindo_emoji_dict = {
    "1일 1회": "⚪",
    "1일 2회": "✊",
    "2일 1회": "🧚"
}

pum_emoji_dict = {
    "하지않음": "⚪",
    "1~3회/연": "✊",
    "4~6회/연": "🧚",
    "7회 이상/연": "🧚"
}

yaumsac_emoji_dict = {
    "하지않음": "⚪",
    "1~3회/연": "✊",
    "4~6회/연": "🧚",
    "7회 이상/연": "🧚"
}

mobal_emoji_dict = {
    "염색 모발": "⚪",
    "가발 사용(붙임머리 포함)": "✊",
    "모발이식/시술": "🧚",
    "기타": "🧚"
}

product_emoji_dict = {
    "샴푸": "⚪",
    "트리트먼트": "✊",
    "헤어에센스": "🧚",
    "린스": "🧚",
    "헤어스타일링제": "🧚",
    "두피스케일링제": "🧚",
    "두피세럼": "🧚"
}

hope_emoji_dict = {
    "예": "⚪",
    "아니오": "✊"
}

variety_emoji_dict = {
    "샴푸": "🐲",
    "린스/컨디셔너": "🤖",
    "샴푸바/드라이샴푸": "🧚",
    "헤어오일/헤어세럼": "👨‍🚒",
    "헤어워터": "🦹",
    "두피팩/스케일러": "🦔",
    "헤어토닉/두피토닉": "🐯"
}

gorau_emoji_dict = {
    "세정력": "⚪",
    "두피자극": "✊",
    "머리결": "✊",
    "향": "✊",
    "헹굼후느낌": "✊",
    "가격": "✊"
}

initial_scalp = [
    {
        "name": "",
        "sex": [""],
        "age": [""],
        "symptom": [""],
        "location": [""],
        "bindo": [""],
        "pum": [""],
        "yaumsac": [""],
        "mobal": [""],
        "product": [""],
        "hope": [""],
        "variety": ["샴푸", "린스/컨디셔너", "샴푸바/드라이샴푸"],
        "gorau": [""],
        "bidum_state": "",
        "gakzil_state": "",
        "hongban_state": "",
        "nongpo_state": "",
        "pizy_state": "",
        "talmo_state": ""
    }
]

initial_upload = {
    "session": 0,
    "filepath": ""
}



example_scalps_img = [
    {
        "name": "정상",
        "url": "./data/images/nomal.jpg"
    },
    {
        "name": "비듬",
        "url": "./data/images/bidum.jpg"
    },
    {
        "name": "미세각질",
        "url": "./data/images/gakzil.jpg"
    },
    {
        "name": "미세사이홍반",
        "url": "./data/images/hongban.jpg"
    },
    {
        "name": "미세홍반농포",
        "url": "./data/images/nongpo.jpg"
    },
    {
        "name": "피지과다",
        "url": "./data/images/pizy.jpg"
    },
    {
        "name": "탈모",
        "url": "./data/images/talmo.jpg"
    },
]


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def load_survey():
    # 데이터 불러오기
    df_survey = pd.read_csv("./data/meta_data.csv", encoding="utf-8")

    return df_survey

def make_value_of_graph(df_survey):

    # 각 입력 값 지역변수로 입력하기
    name = st.session_state.scalp[0]["name"]
    sex = ",".join(st.session_state.scalp[0]["sex"])
    age = ",".join(st.session_state.scalp[0]["age"])
    symptom = ",".join(st.session_state.scalp[0]["symptom"])
    location1 = ",".join(st.session_state.scalp[0]["location"]).split("(")
    location2 = location1[1].split(")")
    location = location2[0]
    bindo = ",".join(st.session_state.scalp[0]["bindo"])
    pum = ",".join(st.session_state.scalp[0]["pum"])
    yaumsac = ",".join(st.session_state.scalp[0]["yaumsac"])
    mobal = ",".join(st.session_state.scalp[0]["mobal"])
    product = ",".join(st.session_state.scalp[0]["product"])
    hope = ",".join(st.session_state.scalp[0]["hope"])
    variety = ",".join(st.session_state.scalp[0]["variety"])
    gorau = ",".join(st.session_state.scalp[0]["gorau"])

    # print(name, sex, age, symptom, location, bindo, pum, yaumsac, mobal, product, hope, variety, gorau)

    # 그래프 데이터 만들기
    tmp_data = []
    for i in range(len(df_survey)):
        if df_survey.iloc[i]["성별"] == sex and df_survey.iloc[i]["나이"] == age and df_survey.iloc[i]["사진_위치"] == location and \
                df_survey.iloc[i]["샴푸_사용_빈도"] == bindo and df_survey.iloc[i]["펌_주기"] == pum and df_survey.iloc[i][
            "염색_주기"] == yaumsac and df_survey.iloc[i]["현재_모발_상태"] == mobal and df_survey.iloc[i][
            "사용_중인_두피모발_제품"] == product and df_survey.iloc[i]["두피케어제품_사용_희망"] == hope and df_survey.iloc[i][
            "샴푸_구매시_중요_고려_사항"] == gorau:
            tmp_data.append(df_survey.iloc[i])
    df_result = pd.DataFrame(tmp_data, columns=['Unnamed: 0', 'unique_id', '성별', '나이', '사진_위치', '샴푸_사용_빈도', '펌_주기',
                                                '염색_주기', '현재_모발_상태', '사용_중인_두피모발_제품', '두피케어제품_사용_희망', '샴푸_구매시_중요_고려_사항',
                                                '각질', '피지', '홍반', '농포', '비듬', '탈모'])
    # st.write(df_result)

    return df_result

def count_value_of_graph(df_result):
    gakgil_count = 0
    pizy_count = 0
    hongban_count = 0
    nongpo_count = 0
    bidum_count = 0
    talmo_count = 0

    for j in range(len(df_result)):
        if df_result.iloc[j]["각질"] == 1:
            gakgil_count += 1
        if df_result.iloc[j]["피지"] == 1:
            pizy_count += 1
        if df_result.iloc[j]["홍반"] == 1:
            hongban_count += 1
        if df_result.iloc[j]["농포"] == 1:
            nongpo_count += 1
        if df_result.iloc[j]["비듬"] == 1:
            bidum_count += 1
        if df_result.iloc[j]["탈모"] == 1:
            talmo_count += 1

    tmp_dat2 = []
    tmp_data2 = [[gakgil_count, pizy_count, hongban_count, nongpo_count, bidum_count, talmo_count]]
    df_count = pd.DataFrame(tmp_data2, columns=['미세각질', '피지과다', '미세사이홍반', '미세홍반농포', '비듬', '탈모'])
    df_count = df_count.transpose()
    df_count.columns = ['개수']
    # st.write(df_count)

    return df_count

def draw_graph(df_count):

    plt.figure(figsize=(12, 6))
    plt.rc("font", family="Malgun Gothic")
    ax = df_count.plot(
        kind="bar",
        figsize=(12, 6)
    )
    _ = plt.xticks(size=10, rotation=0, ha="center")

    plt.xlabel("증상")
    plt.ylabel("개수")

    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        # print(p.get_bbox().bounds)
        # ax.annotate("%0.5f%% (%i명/18,154명)" % (height / 18154 * 100, height), (left + width / 2, height * 1.01), ha='center', size=8)
        ax.annotate("%i명/18,154명" % (height), (left + width / 2, height * 1.01), ha='center', size=8)
    st.pyplot(ax.figure)

def write_values(df_survey, df_count):

    gakgil_count = df_count.iloc[0]["개수"]
    pizy_count = df_count.iloc[1]["개수"]
    hongban_count = df_count.iloc[2]["개수"]
    nongpo_count = df_count.iloc[3]["개수"]
    bidum_count = df_count.iloc[4]["개수"]
    talmo_count = df_count.iloc[5]["개수"]

    all_count = "{:,}".format(len(df_survey))
    if gakgil_count > 0:
        view_gakgil_count = "{:,}".format(gakgil_count)
        st.markdown(
            f"- **미세각질**이 **{round(gakgil_count / 18154 * 100, 5)}%** 확률로 생길 수 있으며 기존 데이터에서 **{all_count}**명 중 **{view_gakgil_count}**명이 존재합니다.")
    if pizy_count > 0:
        view_pizy_count = "{:,}".format(pizy_count)
        st.markdown(
            f"- **피지과다**가 **{round(pizy_count / 18154 * 100, 5)}%** 확률로 생길 수 있으며 기존 데이터에서 **{all_count}**명 중 **{view_pizy_count}**명이 존재합니다.")
    if hongban_count > 0:
        view_hongban_count = "{:,}".format(hongban_count)
        st.markdown(
            f"- **미세사이홍반**이 **{round(hongban_count / 18154 * 100, 5)}%** 확률로 생길 수 있으며 기존 데이터에서 **{all_count}**명 중 **{view_hongban_count}**명이 존재합니다.")
    if nongpo_count > 0:
        view_nongpo_count = "{:,}".format(nongpo_count)
        st.markdown(
            f"- **미세홍반농포**가 **{round(nongpo_count / 18154 * 100, 5)}%** 확률로 생길 수 있으며 기존 데이터에서 **{all_count}**명 중 **{view_nongpo_count}**명이 존재합니다.")
    if bidum_count > 0:
        view_bidum_count = "{:,}".format(bidum_count)
        st.markdown(
            f"- **비듬**이 **{round(bidum_count / 18154 * 100, 5)}%** 확률로 생길 수 있으며 기존 데이터에서 **{all_count}**명 중 **{view_bidum_count}**명이 존재합니다.")
    if talmo_count > 0:
        view_talmo_count = "{:,}".format(talmo_count)
        st.markdown(
            f"- **탈모**가 **{round(talmo_count / 18154 * 100, 5)}%** 확률로 생길 수 있으며 기존 데이터에서 **{all_count}**명 중 **{view_talmo_count}**명이 존재합니다.")

# def draw_graphs():
#     그래프 그리기
#     col12, col13, col14 = st.columns(3)
#     with col12:
#         bins = np.arange(0, 25, 1)
#         plt.figure(figsize=(10, 5))
#         plt.hist(
#             df[DATE_COLUMN].dt.hour,
#             bins=bins,
#             label="count",
#             width=0.8,
#         )
#         plt.legend()
#         plt.xlim(0, 24)
#         plt.xticks(bins, fontsize=8)
#         st.pyplot(plt)
#     with col13:
#         bins = np.arange(0, 25, 1)
#         plt.figure(figsize=(10, 5))
#         plt.hist(
#             df[DATE_COLUMN].dt.hour,
#             bins=bins,
#             label="count",
#             width=0.8,
#         )
#         plt.legend()
#         plt.xlim(0, 24)
#         plt.xticks(bins, fontsize=8)
#         st.pyplot(plt)
#     with col14:
#         bins = np.arange(0, 25, 1)
#         plt.figure(figsize=(10, 5))
#         plt.hist(
#             df[DATE_COLUMN].dt.hour,
#             bins=bins,
#             label="count",
#             width=0.8,
#         )
#         plt.legend()
#         plt.xlim(0, 24)
#         plt.xticks(bins, fontsize=8)
#         st.pyplot(plt)


# 이미지 크기 조정 함수 추가
def resize_with_aspect_ratio(image, target_size):
    w, h = image.size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)
    else:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    return image.resize((new_w, new_h), Image.BICUBIC)

def load_models():
    #사전 학습된 모델 불러오기 (24.10.30)
    model1 = torch.load('./data/models/bidum_model_label3_92.7.pt', map_location=torch.device('cpu'))
    model1.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model2 = torch.load('./data/models/gakzil_model_label3_84%.pt', map_location=torch.device('cpu'))
    model2.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model3 = torch.load('./data/models/hongban_label3_93.2%.pt', map_location=torch.device('cpu'))
    model3.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model4 = torch.load('./data/models/nongpo_model_label3_89.5.pt', map_location=torch.device('cpu'))
    model4.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model5 = torch.load('./data/models/pizy_model_92.6%.pt', map_location=torch.device('cpu'))
    model5.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)
    model6 = torch.load('./data/models/talmo_model_93.48%.pt', map_location=torch.device('cpu'))
    model6.eval()  # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 비활성화)

    return [model1, model2, model3, model4, model5, model6]

def load_image(image_path):

    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect_ratio(img, target_size=240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')  # 이미지를 RGB로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)

    return image

# 이미지를 모델에 통과시켜 예측하는 함수
def predict_image(image_path):

    class_names = ['class1', 'class2', 'class3']

    models = load_models()
    model1 = models[0]
    model2 = models[1]
    model3 = models[2]
    model4 = models[3]
    model5 = models[4]
    model6 = models[5]

    # 장치 설정 (GPU 사용 가능 시 GPU로 이동)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)
    model6 = model6.to(device)


    image_tensor = load_image(image_path)  # 이미지를 전처리하여 텐서로 변환
    image_tensor = image_tensor.to(device)  # 모델과 동일한 장치로 이동 (CPU/GPU)

    with torch.no_grad():  # 예측 시 기울기 계산을 하지 않음
        outputs1 = model1(image_tensor)  # 모델의 예측값 (로짓)
        outputs2 = model2(image_tensor)  # 모델의 예측값 (로짓)
        outputs3 = model3(image_tensor)  # 모델의 예측값 (로짓)
        outputs4 = model4(image_tensor)  # 모델의 예측값 (로짓)
        outputs5 = model5(image_tensor)  # 모델의 예측값 (로짓)
        outputs6 = model6(image_tensor)  # 모델의 예측값 (로짓)
        _, preds1 = torch.max(outputs1, 1)  # 가장 높은 확률의 클래스 선택
        _, preds2 = torch.max(outputs2, 1)  # 가장 높은 확률의 클래스 선택
        _, preds3 = torch.max(outputs3, 1)  # 가장 높은 확률의 클래스 선택
        _, preds4 = torch.max(outputs4, 1)  # 가장 높은 확률의 클래스 선택
        _, preds5 = torch.max(outputs5, 1)  # 가장 높은 확률의 클래스 선택
        _, preds6 = torch.max(outputs6, 1)  # 가장 높은 확률의 클래스 선택

        probabilities1 = torch.nn.functional.softmax(outputs1, dim=1)
        probabilities2 = torch.nn.functional.softmax(outputs2, dim=1)
        probabilities3 = torch.nn.functional.softmax(outputs3, dim=1)
        probabilities4 = torch.nn.functional.softmax(outputs4, dim=1)
        probabilities5 = torch.nn.functional.softmax(outputs5, dim=1)
        probabilities6 = torch.nn.functional.softmax(outputs6, dim=1)

        # 각 이미지에 대한 클래스별 확률값 저장
        prob_values1 = probabilities1.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values2 = probabilities2.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values3 = probabilities3.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values4 = probabilities4.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values5 = probabilities5.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환
        prob_values6 = probabilities6.cpu().numpy()[0]  # 확률값을 numpy 배열로 변환

        top_probability1 = prob_values1[0]  # 첫 번째 확률
        top_probability2 = prob_values2[0]  # 첫 번째 확률
        top_probability3 = prob_values3[0]  # 첫 번째 확률
        top_probability4 = prob_values4[0]  # 첫 번째 확률
        top_probability5 = prob_values5[0]  # 첫 번째 확률
        top_probability6 = prob_values6[0]  # 첫 번째 확률

        second_probability1 = prob_values1[1]  # 두 번째 확률
        second_probability2 = prob_values2[1]  # 두 번째 확률
        second_probability3 = prob_values3[1]  # 두 번째 확률
        second_probability4 = prob_values4[1]  # 두 번째 확률
        second_probability5 = prob_values5[1]  # 두 번째 확률
        second_probability6 = prob_values6[1]  # 두 번째 확률

        third_probability1 = prob_values1[2]  # 세 번째 확률
        third_probability2 = prob_values2[2]  # 세 번째 확률
        third_probability3 = prob_values3[2]  # 세 번째 확률
        third_probability4 = prob_values4[2]  # 세 번째 확률
        third_probability5 = prob_values5[2]  # 세 번째 확률
        third_probability6 = prob_values6[2]  # 세 번째 확률

        # 상위 확률 및 해당 클래스 찾기
        # top_two_indices1 = prob_values1.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices2 = prob_values2.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices3 = prob_values3.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices4 = prob_values4.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices5 = prob_values5.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)
        # top_two_indices6 = prob_values6.argsort()[-3:][::-1]  # 상위 2개의 인덱스 (내림차순)

        # top_class1 = class_names[top_two_indices1[0]]  # 첫 번째 클래스
        # top_class2 = class_names[top_two_indices2[0]]  # 첫 번째 클래스
        # top_class3 = class_names[top_two_indices3[0]]  # 첫 번째 클래스
        # top_class4 = class_names[top_two_indices4[0]]  # 첫 번째 클래스
        # top_class5 = class_names[top_two_indices5[0]]  # 첫 번째 클래스
        # top_class6 = class_names[top_two_indices6[0]]  # 첫 번째 클래스

        # top_probability1 = prob_values1[top_two_indices1[0]]  # 첫 번째 확률
        # top_probability2 = prob_values2[top_two_indices2[0]]  # 첫 번째 확률
        # top_probability3 = prob_values3[top_two_indices3[0]]  # 첫 번째 확률
        # top_probability4 = prob_values4[top_two_indices4[0]]  # 첫 번째 확률
        # top_probability5 = prob_values5[top_two_indices5[0]]  # 첫 번째 확률
        # top_probability6 = prob_values6[top_two_indices6[0]]  # 첫 번째 확률
        #
        # second_class1 = class_names[top_two_indices1[1]]  # 두 번째 클래스
        # second_class2 = class_names[top_two_indices2[1]]  # 두 번째 클래스
        # second_class3 = class_names[top_two_indices3[1]]  # 두 번째 클래스
        # second_class4 = class_names[top_two_indices4[1]]  # 두 번째 클래스
        # second_class5 = class_names[top_two_indices5[1]]  # 두 번째 클래스
        # second_class6 = class_names[top_two_indices6[1]]  # 두 번째 클래스
        #
        # second_probability1 = prob_values1[top_two_indices1[1]]  # 두 번째 확률
        # second_probability2 = prob_values2[top_two_indices2[1]]  # 두 번째 확률
        # second_probability3 = prob_values3[top_two_indices3[1]]  # 두 번째 확률
        # second_probability4 = prob_values4[top_two_indices4[1]]  # 두 번째 확률
        # second_probability5 = prob_values5[top_two_indices5[1]]  # 두 번째 확률
        # second_probability6 = prob_values6[top_two_indices6[1]]  # 두 번째 확률
        #
        # third_class1 = class_names[top_two_indices1[2]]  # 세 번째 클래스
        # third_class2 = class_names[top_two_indices2[2]]  # 세 번째 클래스
        # third_class3 = class_names[top_two_indices3[2]]  # 세 번째 클래스
        # third_class4 = class_names[top_two_indices4[2]]  # 세 번째 클래스
        # third_class5 = class_names[top_two_indices5[2]]  # 세 번째 클래스
        # third_class6 = class_names[top_two_indices6[2]]  # 세 번째 클래스
        #
        # third_probability1 = prob_values1[top_two_indices1[2]]  # 세 번째 확률
        # third_probability2 = prob_values2[top_two_indices2[2]]  # 세 번째 확률
        # third_probability3 = prob_values3[top_two_indices3[2]]  # 세 번째 확률
        # third_probability4 = prob_values4[top_two_indices4[2]]  # 세 번째 확률
        # third_probability5 = prob_values5[top_two_indices5[2]]  # 세 번째 확률
        # third_probability6 = prob_values6[top_two_indices6[2]]  # 세 번째 확률


    # return [[preds1.item(), top_class1, top_probability1, second_class1, second_probability1, third_class1, third_probability1],
    #         [preds2.item(), top_class2, top_probability2, second_class2, second_probability2, third_class2, third_probability2],
    #         [preds3.item(), top_class3, top_probability3, second_class3, second_probability3, third_class3, third_probability3],
    #         [preds4.item(), top_class4, top_probability4, second_class4, second_probability4, third_class4, third_probability4],
    #         [preds5.item(), top_class5, top_probability5, second_class5, second_probability5, third_class5, third_probability5],
    #         [preds6.item(), top_class6, top_probability6, second_class6, second_probability6, third_class6, third_probability6]]  # 예측된 클래스 반환

    return [[preds1.item(), top_probability1, second_probability1, third_probability1],
            [preds2.item(), top_probability2, second_probability2, third_probability2],
            [preds3.item(), top_probability3, second_probability3, third_probability3],
            [preds4.item(), top_probability4, second_probability4, third_probability4],
            [preds5.item(), top_probability5, second_probability5, third_probability5],
            [preds6.item(), top_probability6, second_probability6, third_probability6]]  # 예측된 클래스 반환

def save_meta(file_name):
    # 저장할 경로 설정
    SAVE_FOLDER = './data/uploaded_images/meta/'

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # 각 입력 값 지역변수로 입력하기
    name = st.session_state.scalp[0]["name"]
    sex = ",".join(st.session_state.scalp[0]["sex"])
    age = ",".join(st.session_state.scalp[0]["age"])
    symptom = ",".join(st.session_state.scalp[0]["symptom"])
    location1 = ",".join(st.session_state.scalp[0]["location"]).split("(")
    location2 = location1[1].split(")")
    location = location2[0]
    bindo = ",".join(st.session_state.scalp[0]["bindo"])
    pum = ",".join(st.session_state.scalp[0]["pum"])
    yaumsac = ",".join(st.session_state.scalp[0]["yaumsac"])
    mobal = ",".join(st.session_state.scalp[0]["mobal"])
    product = ",".join(st.session_state.scalp[0]["product"])
    hope = ",".join(st.session_state.scalp[0]["hope"])
    variety = ",".join(st.session_state.scalp[0]["variety"])
    gorau = ",".join(st.session_state.scalp[0]["gorau"])


    # JSON으로 저장할 데이터 (Python 딕셔너리 형식)
    data = {
        "gender": sex,
        "age": age,
        "location": location,
        "question1":"샴푸 사용 빈도",
        "answers1":bindo,
        "question2":"펌 주기",
        "answers2":pum,
        "question3":"염색 주기 (자가 염색 포함)",
        "answers3":yaumsac,
        "question4":"현재 모발 상태",
        "answers4":mobal,
        "question5":"현재 사용하고 있는 두피모발용 제품",
        "answers5":product,
        "question6":"맞춤두피케어 제품사용을 희망(선호)하시나요",
        "answers6":hope,
        "question7":"샴푸 구매시 중요시 고려하는 부분은 무엇인가요?",
        "answers7":gorau
    }

    file_name = file_name.split(".")[0]

    # JSON 파일 생성 및 데이터 기록
    with open(f"{SAVE_FOLDER+file_name}_META.json", "w", encoding="utf-8-sig") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)  # indent=4는 파일을 읽기 좋게 정렬

def save_image(file_name):
    # 저장할 경로 설정
    SAVE_FOLDER = './data/uploaded_images/'

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    hongban_data_folder_of_class1 = SAVE_FOLDER + "test/[원천]모낭사이홍반_0.양호/"
    hongban_data_folder_of_class2 = SAVE_FOLDER + "test/[원천]모낭사이홍반_1.경증/"
    hongban_data_folder_of_class3 = SAVE_FOLDER + "test/[원천]모낭사이홍반_3.중증/"

    nongpo_data_folder_of_class1 = SAVE_FOLDER + "test/[원천]모낭홍반농포_0.양호/"
    nongpo_data_folder_of_class2 = SAVE_FOLDER + "test/[원천]모낭홍반농포_1.경증/"
    nongpo_data_folder_of_class3 = SAVE_FOLDER + "test/[원천]모낭홍반농포_3.중증/"

    gakzil_data_folder_of_class1 = SAVE_FOLDER + "test/[원천]미세각질_0.양호/"
    gakzil_data_folder_of_class2 = SAVE_FOLDER + "test/[원천]미세각질_1.경증/"
    gakzil_data_folder_of_class3 = SAVE_FOLDER + "test/[원천]미세각질_3.중증/"

    bidum_data_folder_of_class1 = SAVE_FOLDER + "test/[원천]비듬_0.양호/"
    bidum_data_folder_of_class2 = SAVE_FOLDER + "test/[원천]비듬_1.경증/"
    bidum_data_folder_of_class3 = SAVE_FOLDER + "test/[원천]비듬_3.중증/"

    talmo_data_folder_of_class1 = SAVE_FOLDER + "test/[원천]탈모_0.양호/"
    talmo_data_folder_of_class2 = SAVE_FOLDER + "test/[원천]탈모_1.경증/"
    talmo_data_folder_of_class3 = SAVE_FOLDER + "test/[원천]탈모_3.중증/"

    pizy_data_folder_of_class1 = SAVE_FOLDER + "test/[원천]피지과다_0.양호/"
    pizy_data_folder_of_class2 = SAVE_FOLDER + "test/[원천]피지과다_1.경증/"
    pizy_data_folder_of_class3 = SAVE_FOLDER + "test/[원천]피지과다_3.중증/"
    
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(hongban_data_folder_of_class1):
        os.makedirs(hongban_data_folder_of_class1)
    if not os.path.exists(hongban_data_folder_of_class2):
        os.makedirs(hongban_data_folder_of_class2)
    if not os.path.exists(hongban_data_folder_of_class3):
        os.makedirs(hongban_data_folder_of_class3)

    if not os.path.exists(nongpo_data_folder_of_class1):
        os.makedirs(nongpo_data_folder_of_class1)
    if not os.path.exists(nongpo_data_folder_of_class2):
        os.makedirs(nongpo_data_folder_of_class2)
    if not os.path.exists(nongpo_data_folder_of_class3):
        os.makedirs(nongpo_data_folder_of_class3)

    if not os.path.exists(gakzil_data_folder_of_class1):
        os.makedirs(gakzil_data_folder_of_class1)
    if not os.path.exists(gakzil_data_folder_of_class2):
        os.makedirs(gakzil_data_folder_of_class2)
    if not os.path.exists(gakzil_data_folder_of_class3):
        os.makedirs(gakzil_data_folder_of_class3)

    if not os.path.exists(bidum_data_folder_of_class1):
        os.makedirs(bidum_data_folder_of_class1)
    if not os.path.exists(bidum_data_folder_of_class2):
        os.makedirs(bidum_data_folder_of_class2)
    if not os.path.exists(bidum_data_folder_of_class3):
        os.makedirs(bidum_data_folder_of_class3)

    if not os.path.exists(talmo_data_folder_of_class1):
        os.makedirs(talmo_data_folder_of_class1)
    if not os.path.exists(talmo_data_folder_of_class2):
        os.makedirs(talmo_data_folder_of_class2)
    if not os.path.exists(talmo_data_folder_of_class3):
        os.makedirs(talmo_data_folder_of_class3)

    if not os.path.exists(pizy_data_folder_of_class1):
        os.makedirs(pizy_data_folder_of_class1)
    if not os.path.exists(pizy_data_folder_of_class2):
        os.makedirs(pizy_data_folder_of_class2)
    if not os.path.exists(pizy_data_folder_of_class3):
        os.makedirs(pizy_data_folder_of_class3)

    # 입력값 내부 변수에 저장
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]

    # 클래스 이름 정의
    class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

    if hongban_state == class_names[0]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = hongban_data_folder_of_class1 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    elif hongban_state == class_names[1]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = hongban_data_folder_of_class2 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    else:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = hongban_data_folder_of_class3 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    if nongpo_state == class_names[0]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = nongpo_data_folder_of_class1 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    elif nongpo_state == class_names[1]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = nongpo_data_folder_of_class2 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    else:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = nongpo_data_folder_of_class3 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    if gakzil_state == class_names[0]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = gakzil_data_folder_of_class1 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    elif gakzil_state == class_names[1]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = gakzil_data_folder_of_class2 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    else:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = gakzil_data_folder_of_class3 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    if bidum_state == class_names[0]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = bidum_data_folder_of_class1 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    elif bidum_state == class_names[1]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = bidum_data_folder_of_class2 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    else:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = bidum_data_folder_of_class3 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    if talmo_state == class_names[0]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = talmo_data_folder_of_class1 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    elif talmo_state == class_names[1]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = talmo_data_folder_of_class2 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    else:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = talmo_data_folder_of_class3 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)


    if pizy_state == class_names[0]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = pizy_data_folder_of_class1 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    elif pizy_state == class_names[1]:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = pizy_data_folder_of_class2 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

    else:
        # 원본 이미지 경로와 복사할 경로 지정
        source_path = SAVE_FOLDER + file_name
        destination_path = pizy_data_folder_of_class3 + file_name

        # 이미지 파일 복사
        shutil.copy(source_path, destination_path)

def save_json(file_name):

    # 저장할 경로 설정
    SAVE_FOLDER = './data/uploaded_images/'

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    hongban_label_folder_of_class1 = SAVE_FOLDER + "test/[라벨]모낭사이홍반_0.양호/"
    hongban_label_folder_of_class2 = SAVE_FOLDER + "test/[라벨]모낭사이홍반_1.경증/"
    hongban_label_folder_of_class3 = SAVE_FOLDER + "test/[라벨]모낭사이홍반_3.중증/"


    nongpo_label_folder_of_class1 = SAVE_FOLDER + "test/[라벨]모낭홍반농포_0.양호/"
    nongpo_label_folder_of_class2 = SAVE_FOLDER + "test/[라벨]모낭홍반농포_1.경증/"
    nongpo_label_folder_of_class3 = SAVE_FOLDER + "test/[라벨]모낭홍반농포_3.중증/"


    gakzil_label_folder_of_class1 = SAVE_FOLDER + "test/[라벨]미세각질_0.양호/"
    gakzil_label_folder_of_class2 = SAVE_FOLDER + "test/[라벨]미세각질_1.경증/"
    gakzil_label_folder_of_class3 = SAVE_FOLDER + "test/[라벨]미세각질_3.중증/"


    bidum_label_folder_of_class1 = SAVE_FOLDER + "test/[라벨]비듬_0.양호/"
    bidum_label_folder_of_class2 = SAVE_FOLDER + "test/[라벨]비듬_1.경증/"
    bidum_label_folder_of_class3 = SAVE_FOLDER + "test/[라벨]비듬_3.중증/"


    talmo_label_folder_of_class1 = SAVE_FOLDER + "test/[라벨]탈모_0.양호/"
    talmo_label_folder_of_class2 = SAVE_FOLDER + "test/[라벨]탈모_1.경증/"
    talmo_label_folder_of_class3 = SAVE_FOLDER + "test/[라벨]탈모_3.중증"


    pizy_label_folder_of_class1 = SAVE_FOLDER + "test/[라벨]피지과다_0.양호/"
    pizy_label_folder_of_class2 = SAVE_FOLDER + "test/[라벨]피지과다_1.경증/"
    pizy_label_folder_of_class3 = SAVE_FOLDER + "test/[라벨]피지과다_3.중증/"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(hongban_label_folder_of_class1):
        os.makedirs(hongban_label_folder_of_class1)
    if not os.path.exists(hongban_label_folder_of_class2):
        os.makedirs(hongban_label_folder_of_class2)
    if not os.path.exists(hongban_label_folder_of_class3):
        os.makedirs(hongban_label_folder_of_class3)


    if not os.path.exists(nongpo_label_folder_of_class1):
        os.makedirs(nongpo_label_folder_of_class1)
    if not os.path.exists(nongpo_label_folder_of_class2):
        os.makedirs(nongpo_label_folder_of_class2)
    if not os.path.exists(nongpo_label_folder_of_class3):
        os.makedirs(nongpo_label_folder_of_class3)


    if not os.path.exists(gakzil_label_folder_of_class1):
        os.makedirs(gakzil_label_folder_of_class1)
    if not os.path.exists(gakzil_label_folder_of_class2):
        os.makedirs(gakzil_label_folder_of_class2)
    if not os.path.exists(gakzil_label_folder_of_class3):
        os.makedirs(gakzil_label_folder_of_class3)


    if not os.path.exists(bidum_label_folder_of_class1):
        os.makedirs(bidum_label_folder_of_class1)
    if not os.path.exists(bidum_label_folder_of_class2):
        os.makedirs(bidum_label_folder_of_class2)
    if not os.path.exists(bidum_label_folder_of_class3):
        os.makedirs(bidum_label_folder_of_class3)


    if not os.path.exists(talmo_label_folder_of_class1):
        os.makedirs(talmo_label_folder_of_class1)
    if not os.path.exists(talmo_label_folder_of_class2):
        os.makedirs(talmo_label_folder_of_class2)
    if not os.path.exists(talmo_label_folder_of_class3):
        os.makedirs(talmo_label_folder_of_class3)


    if not os.path.exists(pizy_label_folder_of_class1):
        os.makedirs(pizy_label_folder_of_class1)
    if not os.path.exists(pizy_label_folder_of_class2):
        os.makedirs(pizy_label_folder_of_class2)
    if not os.path.exists(pizy_label_folder_of_class3):
        os.makedirs(pizy_label_folder_of_class3)

    image_id = file_name.split(".")[0]

    # 클래스 이름 정의
    class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

    # 입력값 내부 변수에 저장
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]

    # 미세각질 (0 : 양호, 1 : 경증, 2 : 중증)
    if gakzil_state == class_names[0]:
        value_1 = 0
    elif gakzil_state == class_names[1]:
        value_1 = 1
    else:
        value_1 = 2

    # 피지과다 (0 : 양호, 1 : 경증, 2 : 중증)
    if pizy_state == class_names[0]:
        value_2 = 0
    elif pizy_state == class_names[1]:
        value_2 = 1
    else:
        value_2 = 2

    # 모낭사이홍반 (0 : 양호, 1 : 경증, 2 : 중증)
    if hongban_state == class_names[0]:
        value_3 = 0
    elif hongban_state == class_names[1]:
        value_3 = 1
    else:
        value_3 = 2

    # 모낭홍반농포 (0 : 양호, 1 : 경증, 2 : 중증)
    if nongpo_state == class_names[0]:
        value_4 = 0
    elif nongpo_state == class_names[1]:
        value_4 = 1
    else:
        value_4 = 2

    # 비듬 (0 : 양호, 1 : 경증, 2 : 중증)
    if bidum_state == class_names[0]:
        value_5 = 0
    elif bidum_state == class_names[1]:
        value_5 = 1
    else:
        value_5 = 2

    # 탈모 (0 : 양호, 1 : 경증, 2 : 중증)
    if talmo_state == class_names[0]:
        value_6 = 0
    elif talmo_state == class_names[1]:
        value_6 = 1
    else:
        value_6 = 2

    # JSON으로 저장할 데이터 (Python 딕셔너리 형식)
    data = {
        "image_id": image_id,
        "image_file_name": file_name,
        "value_1": value_1,
        "value_2": value_2,
        "value_3": value_3,
        "value_4": value_4,
        "value_5": value_5,
        "value_6": value_6
    }

    # 미세각질
    if gakzil_state == class_names[0]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{gakzil_label_folder_of_class1+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    elif gakzil_state == class_names[1]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{gakzil_label_folder_of_class2+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{gakzil_label_folder_of_class3+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    # 피지과다
    if pizy_state == class_names[0]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{pizy_label_folder_of_class1+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    elif pizy_state == class_names[1]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{pizy_label_folder_of_class2+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{pizy_label_folder_of_class3+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    # 모낭사이홍반
    if hongban_state == class_names[0]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{hongban_label_folder_of_class1+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    elif hongban_state == class_names[1]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{hongban_label_folder_of_class2+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{hongban_label_folder_of_class3+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    # 모낭홍반농포
    if nongpo_state == class_names[0]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{nongpo_label_folder_of_class1+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    elif nongpo_state == class_names[1]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{nongpo_label_folder_of_class2+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{nongpo_label_folder_of_class3+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    # 비듬
    if bidum_state == class_names[0]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{bidum_label_folder_of_class1+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    elif bidum_state == class_names[1]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{bidum_label_folder_of_class2+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{bidum_label_folder_of_class3+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    # 탈모
    if talmo_state == class_names[0]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{talmo_label_folder_of_class1+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    elif talmo_state == class_names[1]:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{talmo_label_folder_of_class2+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # JSON 파일 생성 및 데이터 기록
        with open(f"{talmo_label_folder_of_class3+image_id}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)


def request_chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 두피 전문가 입니다."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    return response


def generate_prompt(scalp_type):
    prompt = f"""
{scalp_type}의 원인과 특징 그리고 관리방안을 생성해주세요.
각 타입별 원인, 특징, 관리 방안을 각각 1문장으로 나타내 주세요.
반드시 키워드를 포함해야 합니다.
이모지를 사용하세요.
글씨 크기는 일정하게 해주세요.
---
두피 타입: {scalp_type}
---
""".strip()
    return prompt


def print_streaming_response(response):
    message = ""
    placeholder = st.empty()
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            message += delta.content
            placeholder.markdown(message + "▌")
    placeholder.markdown(message)


# cache 적용 후 새로고침하면 그대로 데이터 로딩 과정을 건너 뜀
@st.cache_data
def load_data(variety):
    if variety == "shampoo":
        df = pd.read_csv("./data/crowlings/shampoo_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "rinse":
        df = pd.read_csv("./data/crowlings/rinse_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "bar":
        df = pd.read_csv("./data/crowlings/bar_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "hairoil":
        df = pd.read_csv("./data/crowlings/hairoil_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "hairwater":
        df = pd.read_csv("./data/crowlings/hairwater_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "scaler":
        df = pd.read_csv("./data/crowlings/scaler_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    elif variety == "tonic":
        df = pd.read_csv("./data/crowlings/tonic_ingredient_data2_add_type.csv", encoding="utf-8-sig")
    return df

def product_recommend(df):
    # 클래스 이름 정의
    class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

    # 상품 추천 목록 보기
    # st.write("")
    # st.markdown('**1. 비듬 관련 상품 추천**')
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    if bidum_state == class_names[1] or bidum_state == class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "건성" or row[j] == "비듬":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"], df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"]])
                        break
        bidum_sorted1 = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)[:5]
        # st.text(f"1) {bidum_sorted1[0][0]}({bidum_sorted1[0][1]})")
        # # st.text(f"2) {bidum_sorted1[1][0]}({bidum_sorted1[1][1]})")
        # # st.text(f"3) {bidum_sorted1[2][0]}({bidum_sorted1[2][1]})")
        # st.text("---------------------------------------------")
        # bidum_sorted2 = sorted(data, key=lambda x: x[2], reverse=True)[:3]
        # st.text(f"1) {bidum_sorted2[0][0]}({bidum_sorted2[0][2]})")
        # # st.text(f"2) {bidum_sorted2[1][0]}({bidum_sorted2[1][2]})")
        # # st.text(f"3) {bidum_sorted2[2][0]}({bidum_sorted2[2][2]})")
    else:
        bidum_sorted1 = []
        # bidum_sorted2 = []

    # st.write("")
    # st.markdown('**2. 미세각질 관련 상품 추천**')
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    if gakzil_state == class_names[1] or gakzil_state == class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "건성" or row[j] == "비듬":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"], df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"]])
                        break
        gakzil_sorted1 = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)[:5]
        # st.text(f"1) {gakzil_sorted1[0][0]}({gakzil_sorted1[0][1]})")
        # st.text(f"2) {gakzil_sorted1[1][0]}({gakzil_sorted1[1][1]})")
        # st.text(f"3) {gakzil_sorted1[2][0]}({gakzil_sorted1[2][1]})")
        # st.text("---------------------------------------------")
        # gakzil_sorted2 = sorted(data, key=lambda x: x[2], reverse=True)[:3]
        # st.text(f"1) {gakzil_sorted2[0][0]}({gakzil_sorted2[0][2]})")
        # st.text(f"2) {gakzil_sorted2[1][0]}({gakzil_sorted2[1][2]})")
        # st.text(f"3) {gakzil_sorted2[2][0]}({gakzil_sorted2[2][2]})")
    else:
        gakzil_sorted1 = []
        # gakzil_sorted2 = []

        # st.write("")
    # st.markdown('**3. 미세사이홍반 관련 상품 추천**')
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    if hongban_state == class_names[1] or hongban_state == class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "지성":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"], df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"]])
                        break
        hongban_sorted1 = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)[:5]
        # st.text(f"1) {hongban_sorted1[0][0]}({hongban_sorted1[0][1]})")
        # st.text(f"2) {hongban_sorted1[1][0]}({hongban_sorted1[1][1]})")
        # st.text(f"3) {hongban_sorted1[2][0]}({hongban_sorted1[2][1]})")
        # st.text("---------------------------------------------")
        # hongban_sorted2 = sorted(data, key=lambda x: x[2], reverse=True)[:3]
        # st.text(f"1) {hongban_sorted2[0][0]}({hongban_sorted2[0][2]})")
        # st.text(f"2) {hongban_sorted2[1][0]}({hongban_sorted2[1][2]})")
        # st.text(f"3) {hongban_sorted2[2][0]}({hongban_sorted2[2][2]})")
    else:
        hongban_sorted1 = []
        # hongban_sorted2 = []

    # st.write("")
    # st.markdown('**4. 미세홍반농포 관련 상품 추천**')
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    if nongpo_state == class_names[1] or nongpo_state == class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "지루성":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"], df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"]])
                        break
        nongpo_sorted1 = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)[:5]
        # st.text(f"1) {nongpo_sorted1[0][0]}({nongpo_sorted1[0][1]})")
        # st.text(f"2) {nongpo_sorted1[1][0]}({nongpo_sorted1[1][1]})")
        # st.text(f"3) {nongpo_sorted1[2][0]}({nongpo_sorted1[2][1]})")
        # st.text("---------------------------------------------")
        # nongpo_sorted2 = sorted(data, key=lambda x: x[2], reverse=True)[:3]
        # st.text(f"1) {nongpo_sorted2[0][0]}({nongpo_sorted2[0][2]})")
        # st.text(f"2) {nongpo_sorted2[1][0]}({nongpo_sorted2[1][2]})")
        # st.text(f"3) {nongpo_sorted2[2][0]}({nongpo_sorted2[2][2]})")
    else:
        nongpo_sorted1 = []
        # nongpo_sorted2 = []

    # st.write("")
    # st.markdown('**5. 피지과다 관련 상품 추천**')
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    if pizy_state == class_names[1] or pizy_state == class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "지성" or row[j] == "지루성":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"], df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"]])
                        break
        pizy_sorted1 = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)[:5]
        # st.text(f"1) {pizy_sorted1[0][0]}({pizy_sorted1[0][1]})")
        # st.text(f"2) {pizy_sorted1[1][0]}({pizy_sorted1[1][1]})")
        # st.text(f"3) {pizy_sorted1[2][0]}({pizy_sorted1[2][1]})")
        # st.text("---------------------------------------------")
        # pizy_sorted2 = sorted(data, key=lambda x: x[2], reverse=True)[:3]
        # st.text(f"1) {pizy_sorted2[0][0]}({pizy_sorted2[0][2]})")
        # st.text(f"2) {pizy_sorted2[1][0]}({pizy_sorted2[1][2]})")
        # st.text(f"3) {pizy_sorted2[2][0]}({pizy_sorted2[2][2]})")
    else:
        pizy_sorted1 = []
        # pizy_sorted2 = []

    # st.write("")
    # st.markdown('**6. 탈모 관련 상품 추천**')
    talmo_state = st.session_state.scalp[0]["talmo_state"]
    if talmo_state == class_names[1] or talmo_state == class_names[2]:
        data = []
        for i in range(len(df)):
            type_line = str(df.iloc[i]["type"])
            if type_line:
                row = type_line.split(",")
                for j in range(len(row)):
                    if row[j] == "탈모":
                        data.append([df.iloc[i]["product_link"], df.iloc[i]["img_link"], df.iloc[i]["brand_name"], df.iloc[i]["product_name"],
                                     df.iloc[i]["star"], df.iloc[i]["review_count"]])
                        break
        talmo_sorted1 = sorted(data, key=lambda x: (x[4], x[5]), reverse=True)[:5]
        # st.text(f"1) {talmo_sorted1[0][0]}({talmo_sorted1[0][1]})")
        # st.text(f"2) {talmo_sorted1[1][0]}({talmo_sorted1[1][1]})")
        # st.text(f"3) {talmo_sorted1[2][0]}({talmo_sorted1[2][1]})")
        # st.text("---------------------------------------------")
        # talmo_sorted2 = sorted(data, key=lambda x: x[2], reverse=True)[:3]
        # st.text(f"1) {talmo_sorted2[0][0]}({talmo_sorted2[0][2]})")
        # st.text(f"2) {talmo_sorted2[1][0]}({talmo_sorted2[1][2]})")
        # st.text(f"3) {talmo_sorted2[2][0]}({talmo_sorted2[2][2]})")
    else:
        talmo_sorted1 = []
        # talmo_sorted2 = []

    tmp = []
    if bidum_sorted1:
        tmp += bidum_sorted1
    if gakzil_sorted1:
        tmp += gakzil_sorted1
    if hongban_sorted1:
        tmp += hongban_sorted1
    if nongpo_sorted1:
        tmp += nongpo_sorted1
    if pizy_sorted1:
        tmp += pizy_sorted1
    if talmo_sorted1:
        tmp += talmo_sorted1

    # 중복 제거
    tmp_list = []
    tmp2 = []
    for k in range(len(tmp)):
        if tmp[k][2] not in tmp_list:
            tmp_list.append(tmp[k][2])
            tmp2.append(tmp[k])
    all_type_result = sorted(tmp2, key=lambda x: (x[3], x[4]), reverse=True)
    # print(all_type_star)

    return all_type_result


def product_view(result):
    tmp = []
    cols = st.columns(6)
    for l in range(3):
        with cols[l]:
            product_link = result[l][0]
            img_link = result[l][1]
            brand_name = result[l][2]
            product_name = result[l][3]
            if brand_name in tmp:
                continue
            else:
                tmp.append(brand_name)

            with st.expander(label=f"**{l + 1}. {product_name}**", expanded=True):
                # st.markdown(f"{l+1}. {product_name}")
                st.markdown(f'''
                    <a href="{product_link}" target="_blank">
                        <img src="{img_link}" alt="image" style="width: 200px;">
                    </a>
                    ''', unsafe_allow_html=True)
    for m in range(3):
        with cols[m]:
            st.write("")
                
if "scalp" not in st.session_state:
    st.session_state.scalp = initial_scalp


if 'page' not in st.session_state:
    st.session_state.page = 0

if 'upload' not in st.session_state:
    st.session_state.upload = initial_upload


if 'survey' not in st.session_state:
    st.session_state.survey = 0

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page = max(0, st.session_state.page - 1)

def prev_main_page():
    st.session_state.upload["session"] = 0
    st.session_state.page = max(0, st.session_state.page - 1)

def home_page():
    st.experimental_js("location.reload()")

if st.session_state.page == 0:
    ############################ 1. 사용자 두피 이미지 업로드  ############################
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔥 사용자 두피 이미지 업로드**")
        uploaded_file = st.file_uploader("자신의 두피이미지를 업로드 해 주세요!!!", type=["jpg", "png", "jpeg"])
        # 저장할 경로 설정
        SAVE_FOLDER = './data/uploaded_images/'

        # 폴더가 존재하지 않으면 생성
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)

        # 파일이 업로드된 경우 처리
        if uploaded_file is not None:
            ############################ 2. 사용자 두피 이미지 결과  ############################
            # 업로드된 파일을 PIL 이미지로 열기
            image = Image.open(uploaded_file)

            # 파일 이름을 가져옴
            file_name = uploaded_file.name

            # 저장할 경로 생성
            file_path = os.path.join(SAVE_FOLDER, file_name)

            # 이미지 파일을 지정한 경로에 저장
            image.save(file_path)
            st.session_state.upload["session"] = 1
            st.session_state.upload["filepath"] = file_path

    with col2:
        if st.session_state.upload["session"] == 1:
            st.markdown("**🔥 사용자 두피 이미지 보기**")
            with st.expander(label="※ 사용자의 이미지", expanded=True):
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write("Image uploaded successfully! 😍")

    st.button("Home", on_click=home_page)
    col3, col4 = st.columns(2)
    with col3:
        st.write("")
    with col4:
        if st.session_state.upload["session"] == 1:
            st.button("Next", on_click=next_page)

elif st.session_state.page == 1:
    ############################ 3. 사용자 정보 입력하기 ############################
    st.markdown("**🔥 사용자 정보 입력**")
    st.text("* 본 개인정보는 시스템에 기록됨을 알려드립니다.")
    auto_complete = st.toggle("예시 데이터로 채우기")
    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(
                label="◾ 이름",
                value=scalp_example["name"] if auto_complete else ""
            )
        with col2:
            sex = st.multiselect(
                label="◾ 성별",
                options=list(sex_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["sex"] if auto_complete else []
            )
        with col3:
            age = st.multiselect(
                label="◾ 나이",
                options=list(age_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["age"] if auto_complete else []
            )
        col4, col5 = st.columns(2)
        with col4:
            symptom = st.multiselect(
                label="◾ 고민 증상",
                options=list(symptom_emoji_dict.keys()),
                max_selections=6,
                default=scalp_example["symptom"] if auto_complete else []
            )
        with col5:
            location = st.multiselect(
                label="◾ 사진 위치",
                options=list(location_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["location"] if auto_complete else []
            )
        col6, col7 = st.columns(2)
        with col6:
            bindo = st.multiselect(
                label="◾ 샴푸 사용 빈도",
                options=list(bindo_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["bindo"] if auto_complete else []
            )
        with col7:
            pum = st.multiselect(
                label="◾ 펌주기",
                options=list(pum_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["pum"] if auto_complete else []
            )
        col8, col9 = st.columns(2)
        with col8:
            yaumsac = st.multiselect(
                label="◾ 염색주기",
                options=list(yaumsac_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["yaumsac"] if auto_complete else []
            )
        with col9:
            mobal = st.multiselect(
                label="◾ 현재 모발 상태",
                options=list(mobal_emoji_dict.keys()),
                max_selections=4,
                default=scalp_example["mobal"] if auto_complete else []
            )
        col10, col11 = st.columns(2)
        with col10:
            product = st.multiselect(
                label="◾ 사용 제품",
                options=list(product_emoji_dict.keys()),
                max_selections=7,
                default=scalp_example["product"] if auto_complete else []
            )
        with col11:
            hope = st.multiselect(
                label="◾ 맞춤 두피 케어 제품 사용 유무",
                options=list(hope_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["hope"] if auto_complete else []
            )
        col12, col13 = st.columns(2)
        with col12:
            variety = st.multiselect(
                label="◾ 추천 제품",
                options=list(variety_emoji_dict.keys()),
                max_selections=7,
                default=scalp_example["variety"] if auto_complete else []
            )
        with col13:
            gorau = st.multiselect(
                label="◾ 샴푸 구매시 고려 부분",
                options=list(gorau_emoji_dict.keys()),
                max_selections=1,
                default=scalp_example["gorau"] if auto_complete else []
            )

        submit = st.form_submit_button(label="Submit")
        if submit:
            if not name:
                st.error("◾ 이름을 입력해주세요.")
            elif len(sex) == 0:
                st.error("◾ 성별을 선택해주세요.")
            elif len(age) == 0:
                st.error("◾ 나이를 선택해주세요.")
            elif len(symptom) == 0:
                st.error("◾ 고민 증상을 한개 이상 선택해주세요.")
            elif len(location) == 0:
                st.error("◾ 올려주실 사진의 위치를 선택해주세요.")
            elif len(bindo) == 0:
                st.error("◾ 샴푸 사용 빈도를 선택해주세요.")
            elif len(pum) == 0:
                st.error("◾ 펌주기를 선택해주세요.")
            elif len(yaumsac) == 0:
                st.error("◾ 염색주기를 선택해주세요.")
            elif len(mobal) == 0:
                st.error("◾ 현재 모발의 상태를 선택해주세요.")
            elif len(product) == 0:
                st.error("◾ 사용하고 있는 두피 모발용 제품을 선택해주세요.")
            elif len(hope) == 0:
                st.error("◾ 맞춤 두피 케어 제품 사용 유무를 선택해주세요.")
            elif len(variety) == 0:
                st.error("◾ 추천 받고 싶은 제품의 종류를 선택해주세요.")
            elif len(gorau) == 0:
                st.error("◾ 샴푸 구매시 중요시 고려하는 부분을 선택해주세요.")
            else:
                st.success("개인 정보 업로드 성공!!!")
                st.session_state.scalp = [{
                    "name": name,
                    "sex": sex,
                    "age": age,
                    "symptom": symptom,
                    "location": location,
                    "bindo": bindo,
                    "pum": pum,
                    "yaumsac": yaumsac,
                    "mobal": mobal,
                    "product": product,
                    "hope": hope,
                    "variety": variety,
                    "gorau": gorau,
                    "bidum_state": "",
                    "gakzil_state": "",
                    "hongban_state": "",
                    "nongpo_state": "",
                    "pizy_state": "",
                    "talmo_state": ""
                }]

    if submit:
        st.session_state.survey == 1

        # df_survey = load_survey()
        # df_result = make_value_of_graph(df_survey)
        # df_count = count_value_of_graph(df_result)
        #
        # st.markdown("**🔥 결과**")
        # if len(df_count) > 0:
        #     st.text("* 결과를 그래프와 텍스트로 보여드립니다.")
        #     draw_graph(df_count)
        #     write_values(df_survey, df_count)
        # else:
        #     st.text("* 당신과 같은 타입의 사용자가 존재하지 않습니다.")

    col14, col15 = st.columns(2)
    with col14:
        st.button("Prev", on_click=prev_main_page)
    with col15:
        st.button("Next", on_click=next_page)

elif st.session_state.page == 2:
    # 파일이 업로드된 경우 처리
    if st.session_state.upload["session"] == 1:

        ############################ 4. 예제 두피 이미지 보여주기 ############################
        st.markdown("**🔥 예제 두피 이미지**")
        st.text("")

        # for i in range(1, len(example_scalps_img), 3):
        #     row_scalps = example_scalps_img[i:i+3]
        #     cols = st.columns(3)
        #     for j in range(len(row_scalps)):
        #         with cols[j]:
        #             scalp = row_scalps[j]
        #             with st.expander(label=f"**{i+j}. {scalp['name']}**", expanded=True):
        #                 st.image(scalp["url"])

        cols = st.columns(6)
        for j in range(1, len(example_scalps_img)):
            with cols[j - 1]:
                scalp = example_scalps_img[j]
                with st.expander(label=f"**{j}. {scalp['name']}**", expanded=True):
                    st.image(scalp["url"])

        ############################ 5. 사용자의 두피 상태 결과 보여주기 ############################
        st.markdown("**🔥 사용자의 두피 상태 결과**")

        # 클래스 이름 정의
        class_names = ["👻 양호", "💧 경증", "😈 중증"]  # 클래스

        # 예측 수행 및 결과 출력
        file_path = st.session_state.upload["filepath"]
        pred_class = predict_image(file_path)

        st.session_state.scalp[0]["bidum_state"] = class_names[pred_class[0][0]]
        st.session_state.scalp[0]["gakzil_state"] = class_names[pred_class[1][0]]
        st.session_state.scalp[0]["hongban_state"] = class_names[pred_class[2][0]]
        st.session_state.scalp[0]["nongpo_state"] = class_names[pred_class[3][0]]
        # st.session_state.scalp[0]["pizy_state"] = class_names[pred_class[4][0]]
        st.session_state.scalp[0]["talmo_state"] = class_names[pred_class[5][0]]
        if class_names[pred_class[1][0]] == "👻 양호":
            st.session_state.scalp[0]["pizy_state"] = "😈 중증"
        elif class_names[pred_class[1][0]] == "💧 경증":
            st.session_state.scalp[0]["pizy_state"] = "💧 경증"
        else:
            st.session_state.scalp[0]["pizy_state"] = "👻 양호"

        with st.container():
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"1. **비듬** : **{class_names[pred_class[0][0]]}**")
                st.markdown(f"[**양호**({round(pred_class[0][1]*100)}%), **경증**({round(pred_class[0][2]*100)}%), **중증**({round(pred_class[0][3]*100)}%)]")
                st.markdown(f"2. **모낭사이홍반** : **{class_names[pred_class[2][0]]}**")
                st.markdown(f"[**양호**({round(pred_class[2][1]*100)}%), **경증**({round(pred_class[2][2]*100)}%), **중증**({round(pred_class[2][3]*100)}%)]")
                st.markdown(f"3. **모낭홍반농포** : **{class_names[pred_class[3][0]]}**")
                st.markdown(f"[**양호**({round(pred_class[3][1]*100)}%), **경증**({round(pred_class[3][2]*100)}%), **중증**({round(pred_class[3][3]*100)}%)]")
            with col4:
                st.markdown(f"4. **탈모** : **{class_names[pred_class[5][0]]}**")
                st.markdown(f"[**양호**({round(pred_class[5][1]*100)}%), **경증**({round(pred_class[5][2]*100)}%), **중증**({round(pred_class[5][3]*100)}%)]")
                st.markdown(f"5. **미세각질** : **{class_names[pred_class[1][0]]}**")
                st.markdown(f"[**양호**({round(pred_class[1][1] * 100)}%), **경증**({round(pred_class[1][2] * 100)}%), **중증**({round(pred_class[1][3] * 100)}%)]")
                st.markdown(f"6. **피지과다** : **{st.session_state.scalp[0]["pizy_state"]}**")
                # st.markdown(f"[**양호**({round(pred_class[4][1]*100)}%), **경증**({round(pred_class[4][2]*100)}%), **중증**({round(pred_class[4][3]*100)}%)]")

        # save_meta(file_name)
        # save_image(file_name)
        # save_json(file_name)

        col5, col6 = st.columns(2)
        with col5:
            st.button("Prev", on_click=prev_page)
        with col6:
            st.button("Next", on_click=next_page)

elif st.session_state.page == 3:
    ############################ 6 두피 타입별 원인과 특징 그리고 관리방안을 보여주기 ############################
    st.write("")
    st.markdown("**🔥 두피 타입별 원인과 특징 그리고 관리방안**")

    # 입력값 내부 변수에 저장
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]


    scalp_type = []
    if bidum_state != "👻 양호":
        scalp_type.append("비듬")
    if gakzil_state != "👻 양호":
        scalp_type.append("미세각질")
    if hongban_state != "👻 양호":
        scalp_type.append("모낭사이홍반")
    if nongpo_state != "👻 양호":
        scalp_type.append("모낭홍반농포")
    if pizy_state != "👻 양호":
        scalp_type.append("피지과다")
    if talmo_state != "👻 양호":
        scalp_type.append("탈모")

    with st.spinner('두피 타입의 원인과 특징 그리고 관리방안을 보여 주고 있습니다...'):
        prompt = generate_prompt(','.join(scalp_type))
        response = request_chat_completion(prompt)
    print_streaming_response(response)


    col1, col2 = st.columns(2)
    with col1:
            st.button("Prev", on_click=prev_page)
    with col2:
            st.button("Next", on_click=next_page)

elif st.session_state.page == 4:
    ############################ 7. 추천 제품 목록 보여주기  ############################

    # 입력값 내부 변수에 저장
    bidum_state = st.session_state.scalp[0]["bidum_state"]
    gakzil_state = st.session_state.scalp[0]["gakzil_state"]
    hongban_state = st.session_state.scalp[0]["hongban_state"]
    nongpo_state = st.session_state.scalp[0]["nongpo_state"]
    pizy_state = st.session_state.scalp[0]["pizy_state"]
    talmo_state = st.session_state.scalp[0]["talmo_state"]

    variety = st.session_state.scalp[0]["variety"]

    st.write("")
    st.markdown("**🔥 추천 제품 목록**")

    for v in variety:
        if v == "샴푸":
            st.write("")
            st.text("* 샴푸를 추천해드리겠습니다.")
            df_shampoo = load_data(variety="shampoo")
            result_shampoo = product_recommend(df_shampoo)
            product_view(result_shampoo)
        if v == "린스/컨디셔너":
            st.write("")
            st.text("* 린스/컨디셔너를 추천해드리겠습니다.")
            df_rinse = load_data(variety = "rinse")
            result_rinse = product_recommend(df_rinse)
            product_view(result_rinse)
        if v == "샴푸바/드라이샴푸":
            st.write("")
            st.text("* 샴푸바/드라이샴푸를 추천해드리겠습니다.")
            df_bar = load_data(variety="bar")
            result_bar = product_recommend(df_bar)
            product_view(result_bar)
        if v == "헤어오일/헤어세럼":
            st.write("")
            st.text("* 헤어오일/헤어세럼을 추천해드리겠습니다.")
            df_bar = load_data(variety="hairoil")
            result_bar = product_recommend(df_bar)
            product_view(result_bar)
        if v == "헤어워터":
            st.write("")
            st.text("* 헤어워터를 추천해드리겠습니다.")
            df_bar = load_data(variety="hairwater")
            result_bar = product_recommend(df_bar)
            product_view(result_bar)
        if v == "두피팩/스케일러":
            st.write("")
            st.text("* 두피팩/스케일러를 추천해드리겠습니다.")
            df_bar = load_data(variety="scaler")
            result_bar = product_recommend(df_bar)
            product_view(result_bar)
        if v == "헤어토닉/두피토닉":
            st.write("")
            st.text("* 헤어토닉/두피토닉을 추천해드리겠습니다.")
            df_bar = load_data(variety="tonic")
            result_bar = product_recommend(df_bar)
            product_view(result_bar)


    col1, col2 = st.columns(2)
    with col1:
        st.button("Prev", on_click=prev_page)
    with col2:
        st.write("")

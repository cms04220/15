import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 파일 경로
file_path = r'C:\Users\82103\OneDrive\문서\GitHub\15\data\2023_weather_accident.csv'

# 데이터 불러오기
try:
    data = pd.read_csv(file_path, encoding='cp949')
    print("데이터 불러오기 성공!")
    print("데이터 컬럼 목록:", data.columns)
except Exception as e:
    print("오류 발생:", e)
    exit()

# 전처리: 결측치 처리 (결측값이 있는 경우 처리 방법 선택)
data = data.dropna()  # 결측치가 있는 행을 제거 (선택사항)

# '사고발생여부' 컬럼 추가
if '소계' in data.columns:
    data['사고발생여부'] = np.where(data['소계'] > 0, 1, 0)
else:
    raise KeyError("'소계' 컬럼이 존재하지 않습니다. CSV 파일을 확인해주세요.")

# 독립변수(X)와 종속변수(y) 설정
X = data[['맑음', '흐림', '비', '안개', '눈', '기타/불명']]
y = data['사고발생여부']

# 데이터 시각화 (날씨별 사고 발생 여부)
weather_sums = data.groupby('사고발생여부')[X.columns].sum()
weather_sums.T.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('날씨 조건별 사고 발생 여부')
plt.xlabel('날씨 조건')
plt.ylabel('발생 건수')
plt.legend(['미발생(0)', '발생(1)'])
plt.savefig('weather_condition_accident.png')  # 날씨별 사고 발생 여부 시각화 저장
plt.close()

# 데이터 분할 (훈련 데이터: 70%, 테스트 데이터: 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습 (RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n모델 정확도: {accuracy:.2f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # 레이블을 명시적으로 지정
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['미발생(0)', '발생(1)'])
disp.plot(cmap='Blues')
plt.title('혼동 행렬')
plt.savefig('confusion_matrix.png')  # 혼동 행렬 저장
plt.close()

# 특성 중요도 시각화
importances = model.feature_importances_
features = X.columns
print("특성 중요도", importances)

# 특성 중요도 내림차순으로 정렬하여 시각화
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(features[indices], importances[indices], color='lightgreen')
plt.title('특성 중요도')
plt.xlabel('중요도')
plt.ylabel('날씨 조건')
plt.gca().invert_yaxis()  # 상위 중요도부터 위로 정렬
plt.savefig('feature_importance.png')  # 특성 중요도 저장
plt.close()

print("시각화 결과가 이미지로 저장되었습니다.")
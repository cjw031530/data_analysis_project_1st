# TabICLv2 분류 실험

저장소 최상단에 원본 CSV를 놓고 Python 3.12 가상환경을 만든 다음 실행한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe TabICL\train.py
```

설정은 [`conf/train.yaml`](conf/train.yaml)에 있다. 첫 실행에서는 공식 사전학습 체크포인트를
Hugging Face 캐시로 내려받는다. `model.device=null`은 사용 가능한 장치를 자동 선택한다.
CPU에서도 실행할 수 있지만 큰 문맥과 많은 예측 행은 오래 걸릴 수 있다. 크기가 큰 실험은
CUDA 지원 PyTorch와 GPU 사용을 권장한다. `requirements.txt` 기본 설치는 환경에 따라
CPU 빌드의 PyTorch를 설치할 수 있다.

기본 실험은 `preprocess/feature_types.csv`의 유형을 사용한다. XGBoost gain 중요도를
202306~202401 학습 행에서만 산출하여 상위 100개 변수를 고른다. `no_month`는 그 100개를,
`month_index`는 상위 99개와 연속 월 인덱스를 사용한다. 범주형과 불리언을 pandas 범주형으로
전달하고 수치형의 결측은 TabICL의 내장 처리를 사용한다. `TARGET`과 원본 `LNMON`은
그대로 입력 변수에 넣지 않는다.

TabICL은 예측할 때 정답이 있는 학습 문맥이 필요하다. 기본 `context.max_rows=10000`은
각 기간의 월과 정답 비율을 보존하는 층화 표본이다. 모든 정답 행을 자동으로 사용하지 않는다.
행 수와 변수 수, 앙상블 수는 설정 인자로 명시적으로 바꿀 수 있다.

```powershell
.\.venv\Scripts\python.exe TabICL\train.py context.max_rows=2000 model.n_estimators=2 features.top_k=50 paths.output_dir=TabICL/outputs/experiment_1
.\.venv\Scripts\python.exe TabICL\train.py --cfg job
```

202306~202401로 입력을 정하고, 202402~202403에서 두 변형과 0/1 판정 임계값을
선택한다. 202404~202405는 선택 후 한 번 테스트한다. 그 뒤 202405까지 정답으로
변수를 다시 선택하고 최종 문맥을 만든다. `metrics.json`의 테스트 점수는 최종 문맥을
만들기 **전** 모델의 점수다. 모델은 원본 자료로 다시 만들 수 있어 대용량 문맥 pickle을
저장하지 않는다. `decision.json`에는 최종 변수와 설정, 원본 데이터 및 유형표의 해시를 남긴다.

정답이 없는 CSV 예측:

```powershell
.\.venv\Scripts\python.exe TabICL\evaluate.py paths.input=unseen.csv
```

예측 시에는 학습에 사용한 원본 CSV가 최상단에 그대로 있어야 한다. 다른 실험을 사용하려면
`paths.model_dir=TabICL/outputs/experiment_1`을 지정한다. 결과는 기본적으로 모델 출력
디렉터리의 `predictions.csv`에 저장된다. 언더샘플된 자료의 원시 예측확률은 모집단의
실제 발생 확률로 해석하지 않는다. 미공개 평가 기간의 가중 점수는 계산하지 않는다.

`TabICL/outputs/`는 Git에서 제외된다. 사전학습 체크포인트도 저장소에 넣지 않는다.

# XGBoost 분류 모델

원본 CSV를 저장소 최상단에 놓고 `requirements.txt`를 설치한 뒤 저장소 최상단에서 실행한다.

```powershell
.\.venv\Scripts\python.exe XGBoost\train.py
```

설정은 [`conf/train.yaml`](conf/train.yaml)에 있다. Hydra 인자를 사용하면 파일 수정 없이
실험 설정을 바꿀 수 있다.

```powershell
.\.venv\Scripts\python.exe XGBoost\train.py tuning.trials=20 tuning.rounds=1000 cpu.jobs=12 paths.output_dir=XGBoost/outputs/experiment_1
.\.venv\Scripts\python.exe XGBoost\train.py tuning.search.max_depth.high=12 paths.output_dir=XGBoost/outputs/depth_12
.\.venv\Scripts\python.exe XGBoost\train.py --cfg job
```

`cpu.jobs=0`은 감지된 논리 CPU 코어 전부를 각 XGBoost 학습에 사용한다. Optuna trial은
CPU 경합을 피하도록 순차 실행한다. tqdm에 전처리 fold와 trial 진행도, 현재 fold,
최고 오분류율이 표시된다.

기본값은 월 변수 제외·포함 모델 각각 250 trial과 4개 시간 순서 fold다.
202306~202401 내부 fold로 튜닝하고, 202402~202403에서 모델과 0/1 판정 임계값을
선택한다. 선택 모델을 202404~202405에서 한 번 테스트한 뒤 202405까지의 전체
정답으로 재학습한다. 보고되는 테스트 점수는 **재학습 전 모델**의 점수다.

`paths.output_dir`에는 재개 가능한 Optuna SQLite study, `metrics.json`,
`decision.json`, `feature_schema.json`, `model.json`이 저장된다. Hydra 설정 기록도
같은 출력 디렉터리의 `hydra/` 아래에 생성된다. 모두 Git에서 제외한다. 같은 설정과
출력 디렉터리로 다시 실행하면 완료한 trial 수를 기준으로 재개한다. fold, 탐색 범위,
seed, 라이브러리 버전 등을 바꿀 때는 새 `paths.output_dir`를 지정한다.

정답 열이 없는 CSV 예측:

```powershell
.\.venv\Scripts\python.exe XGBoost\evaluate.py paths.input=unseen.csv
```

예측 설정은 [`conf/evaluate.yaml`](conf/evaluate.yaml)에 있다. 다른 모델을 쓰려면
`paths.model_dir=XGBoost/outputs/experiment_1`, 출력 경로를 바꾸려면
`paths.output=predictions.csv`를 지정한다. 미공개 평가 기간의 월별 가중 오분류
점수는 계산하지 않는다. 원본은 언더샘플된 자료이므로 원시 예측확률을 모집단의
실제 발생 확률로 해석하지 않는다.

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

## Trial checkpoint와 재개

각 trial의 시간 순서 fold 모델은 `paths.output_dir/checkpoints/<variant>/trial_<번호>/fold_<번호>.ubj`에
저장된다. 같은 디렉터리의 `manifest.json`에는 모델 해시, hyperparameter, fold별 최적 반복 횟수와
평가 결과가 기록된다. 중단된 trial은 저장된 fold 모델을 읽고, 아직 완료되지 않은 fold부터 다시 학습한다.
fold 학습 도중에 중단되었다면 해당 fold는 처음부터 다시 학습한다.

학습을 중단한 다음 같은 명령과 같은 `paths.output_dir`로 재실행하면 SQLite study의 완료된 trial을
건너뛰고 진행 중이던 trial을 재개한다. 이전 버전에서 완료되어 모델 파일이 없는 trial은 재실행 시
모델 파일을 한 번 생성한다. 이전 버전에서 실패로 표시된 trial은 같은 hyperparameter로 다시 실행한다.
한 출력 디렉터리에는 학습 프로세스를 하나만 실행한다. 이미 실행 중인 Python 프로세스에는 코드 변경이
즉시 적용되지 않으므로 새 저장 방식은 그 프로세스를 종료하고 다시 실행할 때부터 적용된다.
모든 trial의 fold 모델을 보관하므로 디스크 사용량이 커질 수 있다.

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

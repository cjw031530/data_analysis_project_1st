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
같은 출력 디렉터리의 `hydra/` 아래에 생성된다. 최종 결과 JSON은 Git에 포함하고,
trial checkpoint, SQLite study, Hydra 기록은 제외한다. 같은 설정과
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

## 전처리 7가지 비교

```powershell
.\.venv\Scripts\python.exe -X utf8 -m XGBoost.preprocessing_experiment
```

설정은 `conf/preprocessing_experiment.yaml`에 있다. 기본값은 NVIDIA GPU이며,
CPU에서는 `device=cpu`를 지정한다. 원형 최소 변환, 현재 전처리, 결측률 50% 초과
열 삭제, One-hot, 삭제한 열의 결측 indicator 유지, 범주형 열 제외, 숫자 범주를
수치형으로 변환하는 7개 방법을 같은 XGBoost 하이퍼파라미터로 비교한다.
원형은 원래 수치값과 결측을 유지하고 문자열에 학습 구간에서 만든 정수 코드를
부여한다. 특수값을 추가로 결측으로 추정하거나 월을 예측 특성에 넣지 않는다.
One-hot은 실제 0과 결측을 구분하는 dense float32 입력을 사용한다.

F7은 202306~202311 학습 / 202312 검증, F8은 202306~202312 학습 / 202401 검증,
F9는 202306~202401 학습 / 202402 검증이다. 각 fold의 학습 행에서만 범주 목록과
결측 삭제 대상을 결정한다. 세 검증 월의 예측을 합쳐 임계값을 정하며,
최적 반복 횟수의 중앙값을 최종 학습에 사용한다. 검증 점수는 이러한 선택에
사용된 점수이므로 독립 테스트 점수로 해석하지 않는다.

검증 상위 2개 방법만 각각 12 trial(초기 4 trial은 공통 기준 1개와 무작위 3개)로
TPE 탐색한다. 모든 선택을 마친 뒤 각 방법을 202402까지 재학습해 202403~202405를
한 번에 평가한다. 월별 가중치는 각각 0.1, 0.3, 0.6이며 점수는
`sum(row_weight * error) / 전체 테스트 행 수`다. 테스트로 early stopping하거나
임계값을 조정하지 않는다. 과거에 일부 테스트 월을 이미 살펴본 탐색적 비교이며,
아직 제공되지 않은 202406~202411의 실제 성능을 측정하는 실험은 아니다.

결과는 `outputs/preprocessing_experiment/`의 JSON·CSV와 PNG·PDF·독립 HTML 보고서로
저장한다. 공통 설정 7개와 추가 튜닝 결과를 구분해서 비교한다. 해당 디렉터리의
`checkpoints/`에는 재개용 예측, 최종 실험 모델, 전처리 객체와 SQLite study를 보관하며
Git에서 제외한다. 같은 설정과 출력 경로로 실행하면 완료한 fold와 trial을 재사용한다.
설정이나 코드가 달라지면 새 출력 경로를 지정한다.

학습 없이 시각화를 다시 생성하거나 전처리 검증을 실행할 수 있다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m XGBoost.visualize_preprocessing
.\.venv\Scripts\python.exe -m unittest XGBoost.test_preprocessing_experiment -v
```

## 전처리 3개 × 시간 포함 여부

```powershell
.\.venv\Scripts\python.exe -X utf8 -m XGBoost.calendar_experiment
```

설정은 `conf/calendar_experiment.yaml`이며 기존 7가지 실험의 공통 모델 설정과
분할을 상속한다. 원형 최소 전처리, 숫자 범주를 수치형으로 전환, 범주형 상한 5의
세 방법을 각각 시간 제외/포함으로 비교한다. 시간은 고정된 기준으로
`202306=1, ..., 202405=12, 202406=13, ..., 202411=18`로 변환하며 숫자형
`month_index` 한 열로 입력한다. 시간 입력에 범주형 cardinality 규칙을 적용하지 않는다.

상한 5 조건은 기존 type map의 정수값·간격 규칙을 유지하면서, 숫자로 해석되는
범주형 열 중 **학습 데이터의 고유 수치값이 5개를 넘는 열**을 수치형으로 바꾼다.
5개 이하는 기존 범주형을 유지한다. 문자열 범주, boolean, 기존 수치형은 유지한다.
결측은 고유값 수에서 제외하고 `1`과 `1.0`은 같은 수치값으로 센다. 각 fold 및
최종 학습 구간의 변경 변수는 `type_changes.csv`에 기록한다. 공통 type map은 수정하지 않는다.

F7~F9 검증, F10~F12 평가와 3·4·5월 가중치 0.1·0.3·0.6 및 전체 행 수 분모를
유지한다. 같은 기본 설정의 6조건을 먼저 비교하고 검증 상위 2조건에만 12-trial
TPE를 적용한다. 검증에서 임계값·반복 횟수·튜닝 후보를 결정한 뒤 평가한다.

데이터 hash, type map hash, Python/라이브러리 버전, seed, fold, 평가 구간,
공통 하이퍼파라미터가 일치하는 기존 `outputs/preprocessing_experiment/` 결과만
재사용한다. 시간 제외의 원형/숫자 범주 전환 결과와, 선택될 경우 동일 조건의
기존 TPE 결과를 재사용한다. 이전 결과가 없으면 6조건을 모두 학습한다.
새 설정과 재사용할 결과가 다르다면 `reuse.enabled=false`로 새로 학습한다.
기존 모델·예측 파일을 복사하지 않고 결과의 `checkpoint_path`로 출처를 남긴다.

결과는 `outputs/calendar_experiment/`에 저장한다. `overview.png/pdf`는 3개 방법의
시간 제외/포함 비교, 시간 추가에 따른 점수 차이, 월별 평가와 검증 성능을 보여준다.
`paired_comparison.csv`는 시간 포함−제외의 차이를 저장하며 음수는 개선이다.
`tuning_comparison.png/pdf`는 공통 설정과 TPE 결과를 비교한다. `report.html`은
이미지가 내장된 독립 보고서이며 표 정렬과 공통 설정/TPE 필터를 제공한다.
`metrics.json`과 `comparison.csv`에는 기존 결과 재사용 여부를 표시한다.
작은 결과물은 Git 대상이며 `checkpoints/`, Hydra 기록은 제외한다.

이 실험은 과거에 살펴본 2024년 3~5월에 대한 탐색적 비교다. 정답과 입력이
아직 없는 2024년 6~11월의 성능을 측정하지 않는다. 숫자 시간 인덱스는 트리 모델의
학습 최대 월 이후 추세를 선형으로 연장하는 기능이 아니다.

학습 없이 시각화를 다시 만들거나 전처리 검증을 실행한다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m XGBoost.visualize_calendar
.\.venv\Scripts\python.exe -m unittest XGBoost.test_calendar_experiment XGBoost.test_preprocessing_experiment -v
.\.venv\Scripts\python.exe -X utf8 -c "from XGBoost.test_calendar_experiment import verify_saved_results; verify_saved_results()"
```

마지막 명령은 저장된 OOF·평가 예측을 원본 행 번호/정답과 대조하고, 임계값과
오분류율·가중 점수를 다시 계산한다. 모델의 입력 특성과 학습 횟수, 코드/데이터
hash 및 cardinality 경계값도 확인한다. 이 검증에는 로컬 checkpoint가 필요하다.

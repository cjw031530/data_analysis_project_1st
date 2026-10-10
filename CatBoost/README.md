# CatBoost

KCB 월별 데이터를 사용한 CatBoost 이진분류 기준 모델입니다. 미래 정보 누수를 막기 위해 시간 순서 rolling OOF 방식으로 평가했습니다.

## 데이터 분할

| Fold | 학습 기간 | 검증 기간 |
|---|---|---|
| F7 | 2023-06 ~ 2023-11 | 2023-12 |
| F8 | 2023-06 ~ 2023-12 | 2024-01 |
| F9 | 2023-06 ~ 2024-01 | 2024-02 |

2024-03 ~ 2024-05 데이터는 최종 test이며 현재 결과에는 사용하지 않았습니다.

## 전처리

- 수치형 변수: 중앙값 대치, 결측 여부 indicator 추가, 표준화
- 문자형·범주형 변수: 최빈값 대치 후 one-hot encoding
- 각 fold의 전처리는 해당 fold 학습 데이터에만 적합
- 분류 임계값: `0.5`
- random seed: `2026`

## 모델 설정

```python
CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=2026,
    thread_count=8,
    verbose=False,
    allow_writing_files=False,
)
```

## 실행 방법

원본 데이터 `kcb_202306_202405_undersampled_1to4.csv`를 Repository 최상단에 둔 뒤 최상단에서 실행합니다.

```bash
python CatBoost/train.py
```

실행하면 `CatBoost/outputs/`에 다음 파일이 생성됩니다.

- `metrics.csv`: fold별 및 전체 OOF 평가 지표
- `predictions.csv`: 고객별 OOF 예측 확률과 예측값
- `results.json`: 설정과 평가 결과

현재는 최종 모델 선택 전이므로 `--run-final-test` 옵션을 사용하지 않습니다.

## OOF 결과

| Split | n | 오분류율 | Accuracy | AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| F7 | 13,513 | 0.157922 | 0.842078 | 0.804698 | 0.378854 |
| F8 | 13,707 | 0.162107 | 0.837893 | 0.811738 | 0.385274 |
| F9 | 12,460 | 0.170064 | 0.829936 | 0.814091 | 0.392707 |
| 전체 OOF | 39,680 | **0.163180** | **0.836820** | **0.810265** | **0.385422** |

검증 월이 뒤로 갈수록 오분류율이 상승했지만 AUC는 소폭 상승했습니다. HistGradientBoosting보다는 전체 OOF 성능이 낮았으나, 후속 앙상블 후보로 보존합니다.

상세 수치는 `results_oof_metrics.csv`에 저장되어 있습니다.

## 필요 패키지

```bash
pip install numpy pandas scikit-learn catboost threadpoolctl
```

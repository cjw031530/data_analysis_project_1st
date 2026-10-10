# HistGradientBoosting

KCB 월별 데이터를 사용한 HistGradientBoosting 이진분류 기준 모델입니다. 미래 정보 누수를 막기 위해 시간 순서 rolling OOF 방식으로 평가했습니다.

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
HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=2026,
)
```

## 실행 방법

원본 데이터 `kcb_202306_202405_undersampled_1to4.csv`를 Repository 최상단에 둔 뒤 최상단에서 실행합니다.

```bash
python HistGradientBoosting/train.py
```

실행하면 `HistGradientBoosting/outputs/`에 다음 파일이 생성됩니다.

- `metrics.csv`: fold별 및 전체 OOF 평가 지표
- `predictions.csv`: 고객별 OOF 예측 확률과 예측값
- `results.json`: 설정과 평가 결과

현재는 최종 모델 선택 전이므로 `--run-final-test` 옵션을 사용하지 않습니다.

## OOF 결과

| Split | n | 오분류율 | Accuracy | AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| F7 | 13,513 | 0.156590 | 0.843410 | 0.812034 | 0.373301 |
| F8 | 13,707 | 0.160502 | 0.839498 | 0.822241 | 0.377143 |
| F9 | 12,460 | 0.166051 | 0.833949 | 0.822919 | 0.384830 |
| 전체 OOF | 39,680 | **0.160912** | **0.839088** | **0.819209** | **0.378248** |

이번 5개 기준 모델 중 오분류율, AUC, log loss가 모두 가장 좋았습니다. 단일 모델 후보로 유지하며 xRFM과의 앙상블 후보로도 사용합니다.

상세 수치는 `results_oof_metrics.csv`에 저장되어 있습니다.

## 필요 패키지

```bash
pip install numpy pandas scikit-learn threadpoolctl
```

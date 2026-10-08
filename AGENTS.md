# AGENTS.md

이 Repository에서 작업할 때 다음 규칙을 따른다.

## 1. 기본 원칙

- 작업을 시작하기 전에 Repository 전체 구조와 관련 코드를 먼저 확인한다.
- 기존 기능과 프로젝트 구조를 불필요하게 변경하지 않는다.
- 기존 코드가 있다면 가능한 한 재사용하고 필요한 부분만 수정한다.
- 작업 후 실행 또는 테스트가 가능하다면 직접 검증한다.

## 2. 프로젝트 구조

Repository 최상단에는 프로젝트 전체에서 공통으로 사용하는 파일만 둔다.

```text
data_analysis_project_1st/
├── README.md
├── AGENTS.md
├── .gitignore
├── requirements.txt
├── kcb_202306_202405_undersampled_1to4.csv
├── XGBoost/
├── RandomForest/
├── LightGBM/
└── ...
```

분석 및 모델링 코드는 **모델 또는 작업 단위별 디렉터리로 구분하여 관리한다.**

새로운 모델을 추가할 때는 해당 모델 이름을 사용한 별도의 디렉터리를 만든다.

```text
XGBoost/
├── train.py
├── evaluate.py
└── ...

RandomForest/
├── train.py
├── evaluate.py
└── ...
```

단순한 모델은 불필요하게 많은 파일로 분리하지 않는다.

```text
XGBoost/
└── train.py
```

코드가 복잡해지면 해당 모델 디렉터리 안에서 필요한 파일을 추가한다.

```text
XGBoost/
├── train.py
├── evaluate.py
├── config.py
├── utils.py
└── ...
```

Repository 최상단에 `train_xgboost.py`, `xgb_final.py`, `model2.py` 등의 모델별 파일을 무분별하게 생성하지 않는다.

기존 모델에 대한 작업이라면 새로운 디렉터리를 만들지 말고 기존 모델 디렉터리를 사용한다.

## 3. 원본 데이터

기본 원본 데이터 파일은:

```text
kcb_202306_202405_undersampled_1to4.csv
```

이며 **Repository 최상단에 존재한다고 가정한다.**

모델별 디렉터리 안에 원본 데이터를 복사하지 않는다.

각 모델의 코드는 최상단의 원본 데이터를 읽도록 작성한다.

원본 데이터는 `.gitignore` 규칙에 따라 Git Repository에 commit하지 않는다.

## 4. 재현 가능한 코드 작성

모든 분석, 실험 및 모델 학습 코드는 다른 사용자가 동일한 실험을 다시 실행할 수 있도록 작성한다.

- random seed가 필요한 경우 명시적으로 설정한다.
- 데이터 전처리를 코드로 재현할 수 있도록 한다.
- train/validation/test 분할 방법을 재현할 수 있도록 한다.
- 모델 학습 및 평가 과정을 코드로 재현할 수 있도록 한다.
- 중요한 hyperparameter를 명시적으로 관리한다.
- 입력 및 출력 경로를 명확하게 관리한다.
- 수동 파일 수정에 의존하지 않는다.
- 새로운 dependency가 필요하면 `requirements.txt`에도 반영한다.

가능하면 사용자가 다음 과정만으로 실험을 다시 수행할 수 있도록 한다.

```text
Repository clone
    ↓
데이터 파일 배치
    ↓
requirements 설치
    ↓
학습 코드 실행
```

## 5. requirements.txt

Python dependency는 Repository 최상단의 `requirements.txt`에서 관리한다.

새로운 Python package가 필요한 코드를 추가한 경우 `requirements.txt`도 함께 수정한다.

이미 존재하는 dependency를 불필요하게 중복 추가하지 않는다.

## 6. 모델 및 Checkpoint 관리

모델 학습 과정에서 생성되는 모든 checkpoint를 Git에 commit하지 않는다.

- 중간 checkpoint → Git에서 제외
- 임시 모델 → Git에서 제외
- 자동 생성되는 대용량 학습 결과 → Git에서 제외
- 최종적으로 사용할 모델 → 필요한 경우에만 보존

checkpoint를 생성하는 코드를 작성할 경우 `.gitignore`를 확인하고, 필요한 경우 적절한 규칙을 추가한다.

최종 모델도 파일 크기가 지나치게 크다면 Git에 직접 commit하지 않는다.

## 7. 출력 파일 관리

모델에서 생성되는 결과는 가능한 한 **해당 모델의 작업 디렉터리 내부에서 관리한다.**

예:

```text
XGBoost/
├── train.py
├── outputs/
│   ├── metrics.json
│   └── ...
└── checkpoints/
    └── ...
```

다른 모델의 결과를 서로 섞지 않는다.

대용량 출력이나 임시 결과는 Git에서 제외한다.

최종적으로 필요한 작은 결과 파일이나 평가 결과는 프로젝트 목적에 따라 보존할 수 있다.

## 8. 생성 파일 관리

다음 자동 생성 파일은 특별한 이유가 없다면 Git에 commit하지 않는다.

- cache
- temporary files
- logs
- intermediate outputs
- intermediate checkpoints
- IDE generated files
- OS generated files

필요하면 `.gitignore`에 규칙을 추가한다.

단, 기존 `.gitignore`를 먼저 확인하고 프로젝트에 필요한 파일까지 제외하지 않도록 한다.

## 9. 기존 파일 보호

기존 데이터, 모델, 실험 결과 또는 사용자가 생성한 파일을 임의로 삭제하지 않는다.

기존 파일을 대량으로 삭제하거나 프로젝트 구조를 크게 변경해야 하는 경우 먼저 사용자에게 알린다.

명확한 이유가 없다면 기존 파일과 결과물을 유지한다.

## 10. 보안

다음 정보는 코드나 Git Repository에 저장하지 않는다.

- API Key
- Access Token
- Password
- Private Key
- Credential
- 기타 비밀정보

환경변수가 필요한 경우 `.env` 등을 사용하고 `.env`는 Git에 commit하지 않는다.

필요하다면 실제 비밀값 대신 `.env.example`을 제공한다.

## 11. 코드 변경 시

다음 순서로 작업한다.

1. Repository 구조를 확인한다.
2. 관련 모델 또는 작업 디렉터리를 확인한다.
3. 관련 코드를 읽고 기존 동작을 이해한다.
4. 필요한 부분만 수정한다.
5. 실행 또는 테스트한다.
6. 오류가 있다면 수정한다.
7. 변경된 파일을 확인한다.

## 12. 새로운 모델 추가 시

1. 모델 이름을 기준으로 새로운 작업 디렉터리를 만든다.
2. 해당 디렉터리에 학습 코드를 작성한다.
3. 필요한 경우 평가 및 설정 파일을 추가한다.
4. 최상단 원본 데이터를 사용한다.
5. 재현 가능한 random seed와 hyperparameter를 설정한다.
6. 필요한 dependency를 `requirements.txt`에 반영한다.
7. checkpoint와 대용량 출력에 대한 `.gitignore`를 확인한다.
8. 가능한 경우 실제 실행하여 정상 동작을 검증한다.

예:

```text
data_analysis_project_1st/
├── kcb_202306_202405_undersampled_1to4.csv
├── requirements.txt
└── XGBoost/
    ├── train.py
    ├── evaluate.py
    └── ...
```

## 13. 작업 완료 전 확인

- 코드가 정상적으로 실행되는가?
- 가능한 범위에서 직접 테스트했는가?
- 결과를 다시 재현할 수 있는가?
- 필요한 random seed가 설정되어 있는가?
- hyperparameter가 확인 가능한가?
- 필요한 dependency가 `requirements.txt`에 있는가?
- 코드가 적절한 모델/작업 디렉터리에 있는가?
- 원본 데이터가 중복 복사되지 않았는가?
- 대용량 데이터가 Git 대상에 포함되지 않았는가?
- 불필요한 checkpoint가 포함되지 않았는가?
- cache, log, 임시 파일이 포함되지 않았는가?
- API Key나 Token 등의 비밀정보가 포함되지 않았는가?
- 기존 파일을 불필요하게 삭제하지 않았는가?

## 14. 작업 결과 보고

작업이 끝나면 사용자에게 간단하게 다음을 설명한다.

1. 무엇을 변경했는지
2. 어떤 파일을 생성하거나 수정했는지
3. 실행 또는 테스트 결과
4. 주요 모델 성능 또는 결과
5. 추가로 필요한 작업이 있는지

불필요하게 긴 설명은 하지 않는다.

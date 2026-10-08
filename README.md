# Data Analysis Project

Repository:

```text
https://github.com/cjw031530/data_analysis_project_1st.git
```

처음 GitHub와 Codex를 사용하는 사람을 위한 간단한 사용법입니다.

기본적인 작업 순서는 다음과 같습니다.

```text
GitHub에서 코드 받기
        ↓
Codex로 코드 작업
        ↓
변경사항 확인
        ↓
GitHub에 올리기
```

---

# 1. 처음 사용할 때: GitHub 로그인

VS Code에서 **Terminal → New Terminal**을 눌러 터미널을 엽니다.

## GitHub CLI 설치

Windows에서는 다음 명령어로 GitHub CLI를 설치합니다.

```powershell
winget install --id GitHub.cli
```

설치가 끝나면 **VS Code를 종료하고 다시 실행**합니다.

설치 확인:

```powershell
gh --version
```

## GitHub 로그인

```powershell
gh auth login
```

다음과 같이 선택합니다.

```text
GitHub.com
HTTPS
Login with a web browser
```

브라우저가 열리면 GitHub 계정으로 로그인합니다.

로그인 확인:

```powershell
gh auth status
```

---

# 2. 처음 프로젝트를 받을 때: git clone

`clone`은 **GitHub에 있는 프로젝트를 내 컴퓨터로 복사하는 명령어**입니다.

```powershell
git clone https://github.com/cjw031530/data_analysis_project_1st.git
```

프로젝트 폴더로 이동합니다.

```powershell
cd data_analysis_project_1st
```

VS Code로 프로젝트를 엽니다.

```powershell
code .
```

`code .`가 동작하지 않으면 VS Code에서 **File → Open Folder**를 선택하여 프로젝트 폴더를 열면 됩니다.

> `git clone`은 프로젝트를 처음 받을 때 한 번만 하면 됩니다.

---

# 3. Python 패키지 설치

프로젝트에 필요한 Python 패키지는 최상단의 `requirements.txt`에 정리되어 있습니다.

```powershell
pip install -r requirements.txt
```

새로운 Python 패키지를 사용하는 코드를 추가했다면 `requirements.txt`도 함께 업데이트해야 합니다.

---

# 4. 작업 시작 전: git pull

`pull`은 **GitHub에 올라와 있는 최신 코드를 내 컴퓨터로 가져오는 명령어**입니다.

```powershell
git pull
```

여러 명이 함께 작업할 경우 다른 사람이 수정한 내용을 먼저 받아올 수 있습니다.

---

# 5. 코드 수정 후: add → commit → push

### ① 변경된 파일 확인

```powershell
git status
```

### ② 변경된 파일 추가

```powershell
git add .
```

### ③ 변경사항 기록

```powershell
git commit -m "수정 내용"
```

예:

```powershell
git commit -m "XGBoost 학습 코드 추가"
```

### ④ GitHub에 업로드

```powershell
git push
```

---

# 6. 평소에는 이것만 기억하면 됩니다

```powershell
git pull

# Codex / VS Code에서 코드 작업

git status
git add .
git commit -m "수정 내용"
git push
```

---

# 7. VS Code에서 Codex 설치 및 열기

VS Code 왼쪽의 **Extensions** 메뉴를 엽니다.

단축키:

```text
Ctrl + Shift + X
```

검색창에서 `Codex`를 검색하고 **OpenAI에서 제공하는 Codex Extension**을 설치합니다.

설치가 끝나면 **VS Code를 한 번 종료하고 다시 실행**합니다.

## Codex 열기

VS Code를 다시 실행한 후 Explorer(파일 목록) 영역의:

```text
Open Codex Sidebar
```

를 클릭합니다.

그러면 오른쪽 사이드바에 **CHAT 옆 CODEX 탭**이 나타납니다.

이후 **CODEX 탭**을 눌러 Codex를 사용하면 됩니다.

---

# 8. Codex로 코드 작업하기

Repository 최상단의 `AGENTS.md`에는 Codex가 작업할 때 따라야 할 공통 규칙이 작성되어 있습니다.

예:

```text
AGENTS.md의 지침에 따라
XGBoost 모델을 학습하고 평가하는 코드를 만들어줘.
```

또는:

```text
AGENTS.md의 지침에 따라
새로운 모델의 작업 디렉터리를 만들고
학습 코드를 작성해서 실행 결과까지 확인해줘.
```

---

# 9. 프로젝트 구조

기본 구조는 다음과 같습니다.

```text
data_analysis_project_1st/
│
├── README.md
├── AGENTS.md
├── .gitignore
├── requirements.txt
├── kcb_202306_202405_undersampled_1to4.csv
│
├── XGBoost/
│   ├── train.py
│   └── ...
│
├── RandomForest/
│   ├── train.py
│   └── ...
│
└── ...
```

원본 데이터 `kcb_202306_202405_undersampled_1to4.csv`는 **Repository 최상단에 배치**합니다.

모델이나 분석 방법이 추가되는 경우에는 각각 별도의 작업 디렉터리를 만들어 관리합니다.

---

# 10. 데이터 파일과 .gitignore

대용량 데이터는 GitHub에 올리지 않습니다.

`kcb_202306_202405_undersampled_1to4.csv`는 최상단 `.gitignore`에 등록되어 있으므로 GitHub에는 업로드되지 않습니다.

Repository를 처음 clone한 사용자는 데이터 파일을 별도로 받아 **Repository 최상단에 직접 배치**해야 합니다.

작업 후에는:

```powershell
git status
```

를 실행하여 대용량 데이터, checkpoint, 임시 파일 등이 실수로 포함되지 않았는지 확인합니다.

---

# 11. 모델 및 Checkpoint

모델 학습 과정에서 생성되는 **모든 checkpoint를 GitHub에 올리지 않습니다.**

- 중간 checkpoint → GitHub에 올리지 않음
- 임시 모델 → GitHub에 올리지 않음
- 최종 모델 → 필요한 경우에만 보존

중간 checkpoint와 임시 결과는 `.gitignore`를 통해 제외합니다.

최종 모델도 파일 크기가 지나치게 크다면 Git Repository에 직접 올리지 않습니다.

---

# 12. Codex 작업 후 확인

Codex가 작업을 완료하면:

```powershell
git status
```

를 실행합니다.

다음을 확인합니다.

- 필요한 코드만 변경되었는가?
- 코드가 정상적으로 실행되는가?
- 데이터 파일이 포함되지 않았는가?
- 불필요한 checkpoint가 포함되지 않았는가?
- API Key, Password, Token 등이 포함되지 않았는가?

문제가 없다면:

```powershell
git add .
git commit -m "수정 내용"
git push
```

---

# 13. 명령어 한눈에 보기

| 명령어 | 의미 |
|---|---|
| `winget install --id GitHub.cli` | GitHub CLI 설치 |
| `gh auth login` | GitHub 로그인 |
| `gh auth status` | 로그인 확인 |
| `git clone 주소` | 프로젝트를 처음 다운로드 |
| `pip install -r requirements.txt` | 필요한 Python 패키지 설치 |
| `git pull` | 최신 코드 가져오기 |
| `git status` | 변경된 파일 확인 |
| `git add .` | 변경된 파일 추가 |
| `git commit -m "내용"` | 변경사항 기록 |
| `git push` | GitHub에 업로드 |

> **중요:** 작업 시작 전에는 `git pull`, 작업 완료 후에는 `git status`를 확인합니다. 데이터 파일과 불필요한 대용량 파일은 GitHub에 올리지 않습니다.

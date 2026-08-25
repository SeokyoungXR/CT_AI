# KOCCA CTxAI: 3D Semantic Scene Graph Generation 튜토리얼 (DeWorldSG, ECCV 2026)

목표: 저장된 DeWorldSG temporal trace를 이용해 ReplicaSSG 데이터셋 내 `office_1`과 `apartment_1` 영상을 기반으로 3D Scene Graph를 생성합니다.



## 1. 프로젝트와 데이터 받기

먼저 이 GitHub 프로젝트를 clone하거나 ZIP으로 내려받습니다. 이후 모든 명령은 `README.md`와 `environment.yml`이 있는 프로젝트 최상위 폴더에서 실행합니다.

그다음 아래 Google Drive에서 데이터를 내려받습니다.

**[ct_ai_office1_apartment1_assets.zip 다운로드](https://drive.google.com/file/d/1uBHToWbH5HfZWdV4kgqd4Cl7G7FZJXNq/view?usp=share_link)**


브라우저 대신 터미널로 받을 수도 있습니다.

```bash
curl -L --fail --retry 3 --progress-bar \
  'https://drive.usercontent.google.com/download?id=1uBHToWbH5HfZWdV4kgqd4Cl7G7FZJXNq&export=download&confirm=t' \
  -o ct_ai_office1_apartment1_assets.zip
```


ZIP을 프로젝트 최상위 폴더에 놓고 압축을 풉니다. `-o`는 다시 실행할 때 기존 asset을 같은 파일로 덮어씁니다.

```bash
unzip -o ct_ai_office1_apartment1_assets.zip -d .
```


```text
CT_AI/
├── assets/traces/
│   ├── office_1/
│   └── apartment_1/
├── data/
│   ├── office_1/
│   └── apartment_1/
├── scripts/
├── environment.yml
└── README.md
```




## 2. conda 환경 만들기

Anaconda가 설치되어 있어야 합니다.

```bash
conda env create -f environment.yml
conda activate ct-ai
```

그 후 다음처럼 업데이트합니다.

```bash
conda env update -f environment.yml
```


## 3. 한 프레임으로 설치 확인하기

먼저 미리보기 한 장을 만들어 설치 상태를 확인합니다.

```bash
python scripts/replay_scene.py --scene office_1 --preview-only
python scripts/replay_scene.py --scene apartment_1 --preview-only
```

결과는 다음 위치에 저장됩니다.

```text
outputs/office_1_preview.png
outputs/apartment_1_preview.png
```

## 4. 3D 장면 그래프 렌더링

### 넓은 구간 샘플 영상 (`--quick`)

처음에는 `--quick`을 권장합니다. 원본 trajectory에서 12프레임마다 선택한 240프레임(`0, 12, ..., 2868`)을 8 FPS로 재생해 30초짜리 영상을 만듭니다. 아래 기본 영상도 240프레임·30초이지만, `--quick`은 더 넓은 시간 구간을 샘플하고 기본 영상은 처음 240프레임을 연속으로 보여줍니다.

```bash
python scripts/replay_scene.py --scene office_1 --quick --output outputs/office_1_quick.mp4
python scripts/replay_scene.py --scene apartment_1 --quick --output outputs/apartment_1_quick.mp4
```

빠른 영상은 `outputs/office_1_quick.mp4`와 `outputs/apartment_1_quick.mp4`에 저장되므로 이후 기본 영상을 만들어도 덮어쓰지 않습니다.

### 기본 영상

옵션 없이 실행하면 연속된 첫 240프레임을 8 FPS로 재생해 30초짜리 영상을 만듭니다.

```bash
python scripts/replay_scene.py --scene office_1
python scripts/replay_scene.py --scene apartment_1
```

결과:

```text
outputs/office_1.mp4
outputs/apartment_1.mp4
```

원하는 수만큼만 렌더링할 수도 있습니다.

```bash
python scripts/replay_scene.py --scene office_1 --max-frames 50
```


| 장면 | 기본 프레임 | 전체 프레임 | 기본 영상 | 전체 영상 |
|---|---:|---:|---:|---:|
| `office_1` | 240 | 3,598 | 30초 | 약 7분 29.8초 |
| `apartment_1` | 240 | 3,600 | 30초 | 7분 30초 |

영상 길이와 실제 렌더링 시간은 다릅니다. 원본 Gaussian 렌더링은 오래 걸릴 수 있으므로 `--preview-only` → `--quick` → 기본 240프레임 순서로 확인하는 것을 권장합니다. ZIP, conda 환경, 렌더 cache를 위해 최소 6GB의 디스크 여유 공간을 권장합니다. 더 빠르거나 느리게 재생하려면 `--fps` 값을 바꿀 수 있습니다.


## 중단 후 이어서 실행하기

렌더링 중 생성되는 패널은 `outputs/.<scene>_frames/` cache에 저장됩니다. `Ctrl+C`로 중단해도 같은 명령을 다시 실행하면 완료된 프레임은 건너뜁니다. MP4 생성이 성공하면 cache는 자동으로 삭제됩니다.

```bash
# cache까지 보존
python scripts/replay_scene.py --scene office_1 --keep-frames

# 이 프로그램이 만든 cache를 지우고 처음부터 다시 시작
python scripts/replay_scene.py --scene office_1 --restart
```

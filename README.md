# DeWorldSG × ReplicaSSG 3D Scene Graph 튜토리얼

저장된 DeWorldSG temporal trace를 이용해 ReplicaSSG의 `office_1`과 `apartment_1`을 원본 스타일의 3D Scene Graph 영상으로 재생합니다.

- 상단: RGB + 프레임별 2D object/relation
- 하단: vertex-color mesh + 프레임별 3D Gaussian object/relation
- 출력: 960×1080, 15 FPS, H.264 MP4
- macOS와 Linux 지원
- CUDA와 PyTorch 불필요

> 이 튜토리얼은 모델을 다시 추론하지 않습니다. 원본 추론에서 저장한 `obj.pkl`, `rel.pkl`, `obj_2d.pkl`, `rel_2d.pkl`을 재생하므로 별도의 model weight는 필요하지 않습니다. 배포 파일은 `.pkl.gz` 형식이며 따로 압축을 풀지 않습니다.

## 1. 프로젝트와 데이터 받기

먼저 이 GitHub 프로젝트를 clone하거나 ZIP으로 내려받습니다. 이후 모든 명령은 `README.md`와 `environment.yml`이 있는 프로젝트 최상위 폴더에서 실행합니다. GitHub ZIP을 사용했다면 폴더 이름이 `CT_AI-main`일 수 있습니다.

그다음 아래 Google Drive에서 학생용 데이터 ZIP을 내려받습니다.

**[ct_ai_office1_apartment1_assets.zip 다운로드](https://drive.google.com/file/d/1uBHToWbH5HfZWdV4kgqd4Cl7G7FZJXNq/view?usp=share_link)**

| 항목 | 값 |
|---|---|
| 파일 크기 | `553,993,805 bytes` (약 528MiB) |
| SHA-256 | `617bab5a4d16e20c4a3480bd8c8fde6550262898e77c915f3b290d545f17a5cb` |

브라우저 대신 터미널로 받을 수도 있습니다.

```bash
curl -L --fail --retry 3 --progress-bar \
  'https://drive.usercontent.google.com/download?id=1uBHToWbH5HfZWdV4kgqd4Cl7G7FZJXNq&export=download&confirm=t' \
  -o ct_ai_office1_apartment1_assets.zip
```

압축을 풀기 전에 선택적으로 다운로드 무결성을 확인할 수 있습니다.

```bash
# macOS
shasum -a 256 ct_ai_office1_apartment1_assets.zip

# Linux
sha256sum ct_ai_office1_apartment1_assets.zip
```

ZIP을 프로젝트 최상위 폴더에 놓고 압축을 풉니다. `-o`는 다시 실행할 때 기존 asset을 같은 파일로 덮어씁니다.

```bash
unzip -o ct_ai_office1_apartment1_assets.zip -d .
```

macOS Finder로 압축을 풀어도 됩니다. 완료 후 반드시 다음 구조가 되어야 합니다.

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

다음 명령 네 개가 모두 파일을 출력하면 올바르게 설치된 것입니다.

```bash
ls data/office_1/sequence/_info.txt
ls data/apartment_1/sequence/_info.txt
ls assets/traces/office_1/manifest.json
ls assets/traces/apartment_1/manifest.json
```

`CT_AI/ct_ai_office1_apartment1_assets/data/...`처럼 폴더가 한 단계 더 생겼다면 그 안의 `data`와 `assets`를 `CT_AI` 바로 아래로 옮기세요.

## 2. conda 환경 만들기

Miniforge, Miniconda 또는 Anaconda가 설치되어 있어야 합니다. 환경 생성은 처음 한 번만 실행합니다.

```bash
conda env create -f environment.yml
conda activate ct-ai
```

이미 `ct-ai` 환경이 있다면 다음처럼 업데이트합니다.

```bash
conda activate ct-ai
conda env update -f environment.yml
```

PyVista/VTK는 원본 3D Gaussian을 CUDA 없이 VTK/OpenGL로 렌더링하는 데 사용됩니다. macOS에서는 GUI로 로그인한 상태의 Terminal에서 실행하세요.

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

## 4. 영상 만들기

### 빠른 영상

처음에는 `--quick`을 권장합니다. 원본 trajectory에서 12프레임마다 선택한 최대 300프레임으로 약 20초짜리 영상을 만듭니다.

```bash
python scripts/replay_scene.py --scene office_1 --quick --output outputs/office_1_quick.mp4
python scripts/replay_scene.py --scene apartment_1 --quick --output outputs/apartment_1_quick.mp4
```

빠른 영상은 `outputs/office_1_quick.mp4`와 `outputs/apartment_1_quick.mp4`에 저장되므로 이후 기본 영상을 만들어도 덮어쓰지 않습니다.

### 기본 영상

옵션 없이 실행하면 첫 1,000프레임에서 자동 종료되며 약 1분 6.7초짜리 영상을 만듭니다.

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
python scripts/replay_scene.py --scene office_1 --max-frames 500
```

전체 원본 trajectory가 필요할 때만 `--all-frames`를 사용합니다.

```bash
python scripts/replay_scene.py --scene office_1 --all-frames
python scripts/replay_scene.py --scene apartment_1 --all-frames
```

| 장면 | 기본 프레임 | 전체 프레임 | 기본 영상 | 전체 영상 |
|---|---:|---:|---:|---:|
| `office_1` | 1,000 | 3,598 | 약 1분 6.7초 | 약 3분 59.9초 |
| `apartment_1` | 1,000 | 3,600 | 약 1분 6.7초 | 4분 |

영상 길이와 실제 렌더링 시간은 다릅니다. 원본 Gaussian 렌더링은 오래 걸릴 수 있으므로 `--preview-only` → `--quick` → 기본 1,000프레임 순서로 확인하는 것을 권장합니다. ZIP, conda 환경, 렌더 cache를 위해 최소 6GB의 디스크 여유 공간을 권장합니다.

## 중단 후 이어서 실행하기

렌더링 중 생성되는 패널은 `outputs/.<scene>_frames/` cache에 저장됩니다. `Ctrl+C`로 중단해도 같은 명령을 다시 실행하면 완료된 프레임은 건너뜁니다. MP4 생성이 성공하면 cache는 자동으로 삭제됩니다.

```bash
# cache까지 보존
python scripts/replay_scene.py --scene office_1 --keep-frames

# 이 프로그램이 만든 cache를 지우고 처음부터 다시 시작
python scripts/replay_scene.py --scene office_1 --restart
```

## 문제 해결

### 데이터가 없다는 오류

현재 위치와 압축 해제 위치를 확인합니다.

```bash
pwd
ls data/office_1
ls data/apartment_1
ls assets/traces/office_1
ls assets/traces/apartment_1
```

### Python 모듈 오류

```bash
conda activate ct-ai
conda env update -f environment.yml
```

### 전체 옵션 확인

```bash
python scripts/replay_scene.py --help
```

## 참고

- Google Drive ZIP에는 두 장면의 RGB, GT pose, mesh와 temporal PKL이 들어 있습니다.
- depth, semantic, texture, SLAM pose, detector weight는 사용하지 않습니다.
- `apartment_1`의 dense relation trace는 렌더러가 실제 사용하는 predicate와 evidence를 정확히 보존한 compact NumPy 형식으로 변환했습니다.
- macOS와 Linux의 VTK/OpenGL 및 글꼴 차이 때문에 픽셀이나 MP4 해시까지 같지는 않을 수 있지만, 장면 데이터·노드·관계·레이아웃은 동일합니다.
- PKL은 역직렬화 시 코드를 실행할 수 있으므로 이 프로젝트에서 제공한 신뢰할 수 있는 파일만 사용하세요. 각 trace는 `manifest.json`의 SHA-256으로 검증됩니다.

개발자용 테스트:

```bash
python -m unittest discover -v
```

시각화 코드는 Apache-2.0인 FROSS `Merging/Visualization`을 macOS/Linux NumPy-only 튜토리얼에 맞게 이식했습니다.

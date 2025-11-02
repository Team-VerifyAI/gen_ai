# Deepfake Detection Dataset Generation

딥페이크 탐지를 위한 고품질 얼굴 이미지 생성 도구 모음

**빠른 시작**: [QUICKSTART.md](QUICKSTART.md) 참고 (설치부터 실행까지 한눈에)

## 지원 모델

### 1. HunyuanImage-2.1 (추천)
- **품질**: 최고 (2048x2048 photorealistic)
- **속도**: ~226초/이미지
- **용량**: 173GB
- **GPU**: 24GB (FP8 quantization)
- **Repository**: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1

### 2. FLUX.1 [dev]
- **품질**: 높음 (1024x1024)
- **속도**: 빠름
- **라이선스**: 비상업적 용도
- **GPU**: 24GB

### 3. FLUX.1 [schnell]
- **품질**: 중상
- **속도**: 매우 빠름 (4 steps)
- **라이선스**: Apache-2.0 (상업적 사용 가능)
- **GPU**: 16GB

## 시스템 요구사항

- **GPU**: NVIDIA GPU with 24GB+ VRAM (HunyuanImage-2.1)
- **System RAM**: 32GB+
- **Storage**: 200GB+ (모델 173GB + 생성 이미지)
- **Python**: 3.10
- **CUDA**: 11.8 or 12.1
- **OS**: Linux (SLURM 환경 권장)

## 설치 가이드

### 1. 저장소 클론
```bash
# 작업 디렉토리로 이동 (충분한 공간이 있는 경로)
cd /data/your_username/repos  # 본인 경로로 수정

# 이 저장소 클론
git clone https://github.com/your-username/gen_image.git
cd gen_image
```

### 2. HunyuanImage-2.1 설치 (추천)

#### 2.1. HunyuanImage 저장소 클론
```bash
cd /data/your_username/repos
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
cd HunyuanImage-2.1
```

#### 2.2. Conda 환경 생성
```bash
# Conda 환경 생성
conda create -n hunyuan python=3.10 -y
conda activate hunyuan

# 의존성 설치
pip install -r requirements.txt

# Flash Attention 설치 (10-20분 소요, CUDA 컴파일 필요)
pip install flash-attn==2.7.3 --no-build-isolation
```

#### 2.3. 모델 다운로드 (173GB, 1-2시간 소요)

**HuggingFace 토큰 발급:**
1. https://huggingface.co/settings/tokens 접속
2. "New token" 클릭
3. 토큰 복사

**모델 다운로드:**
```bash
# 환경변수 설정
export HF_HOME=/data/your_username/.cache/huggingface
export HF_TOKEN=hf_your_token_here  # 발급받은 토큰으로 교체

# 모델 다운로드 (173GB)
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ./ckpts

# 다운로드 중단 시 재개
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ./ckpts --resume-download
```

**모델 구조 확인:**
```bash
ls -lh ckpts/
# 예상 출력:
# ckpts/
# ├── vae/
# ├── text_encoder/
# ├── text_encoder_2/
# ├── t2i/
# └── ...
```

#### 2.4. 환경변수 설정 (필수)

**임시 설정:**
```bash
export HF_HOME=/data/your_username/.cache/huggingface
export HF_TOKEN=hf_your_token_here
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

**영구 설정 (~/.bashrc에 추가):**
```bash
echo 'export HF_HOME=/data/your_username/.cache/huggingface' >> ~/.bashrc
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'" >> ~/.bashrc
source ~/.bashrc
```

#### 2.5. gen_image 스크립트 설정

**run_hunyuan.sh 경로 수정:**
```bash
cd /data/your_username/repos/gen_image

# run_hunyuan.sh 열기
nano run_hunyuan.sh

# 아래 경로들을 본인 환경에 맞게 수정:
# 1. 작업 디렉토리: cd /data/your_username/repos/gen_image
# 2. HF_HOME: export HF_HOME=/data/your_username/.cache/huggingface
# 3. TMPDIR: export TMPDIR=/data/your_username/.tmp
# 4. HF_TOKEN: export HF_TOKEN=hf_your_token_here
# 5. Conda 경로: source /data/your_username/anaconda3/etc/profile.d/conda.sh
```

**scripts/test_hunyuan.py, scripts/generate_hunyuan_dataset.py 경로 수정:**
```bash
# 두 파일의 상단 경로를 본인 경로로 수정
nano scripts/test_hunyuan.py
nano scripts/generate_hunyuan_dataset.py

# 수정할 부분:
# sys.path.insert(0, '/data/your_username/repos/HunyuanImage-2.1')
# sys.path.insert(0, '/data/your_username/repos/gen_image')
# os.chdir('/data/your_username/repos/HunyuanImage-2.1')
```

### 3. FLUX 설치 (선택)

#### 3.1. FLUX 저장소 클론
```bash
cd /data/your_username/repos
git clone https://github.com/black-forest-labs/flux
cd flux
```

#### 3.2. Conda 환경 생성
```bash
conda create -n flux python=3.10 -y
conda activate flux
pip install -e ".[all]"
```

### 4. 설치 확인

#### HunyuanImage-2.1 테스트
```bash
cd /data/your_username/repos/gen_image
conda activate hunyuan

# 단일 이미지 테스트
python scripts/test_hunyuan.py

# 성공 시 outputs/hunyuan_test/ 디렉토리에 이미지 생성됨
ls -lh outputs/hunyuan_test/
```

#### FLUX 테스트
```bash
conda activate flux

# FLUX dev 테스트
python scripts/generate_test.py

# FLUX schnell 테스트
python scripts/generate_schnell_512.py
```

## SLURM 환경 설정 (배치 작업)

### SLURM 파티션 확인
```bash
# 사용 가능한 파티션 확인
sinfo

# GPU 리소스 확인
sinfo -o "%20P %5D %14F %10m %11l %N %G"
```

### run_hunyuan.sh SLURM 설정 수정
```bash
nano run_hunyuan.sh

# SLURM 설정 확인 및 수정:
#SBATCH --partition=batch_ugrad  # 본인 환경의 GPU 파티션으로 변경
#SBATCH --gres=gpu:1              # GPU 타입 지정 필요시: gpu:a100:1
```

## 사용법

### 1. HunyuanImage-2.1 단일 테스트
```bash
# 테스트 (2048x2048)
python scripts/test_hunyuan.py

# 배치 생성 (SLURM)
sbatch run_hunyuan.sh
```

### 2. FLUX 단일 테스트
```bash
# FLUX dev (1024x1024, 고품질)
python scripts/generate_test.py

# FLUX schnell (512x512, 빠름)
python scripts/generate_schnell_512.py
```

## 프롬프트 생성

다양한 얼굴 이미지를 위한 자동 프롬프트 생성:

```python
from prompts_deepfake import generate_prompt_batch

# 다양한 프롬프트 생성
prompts = generate_prompt_batch(100, balanced=True)

# 특징:
# - 70+ 인종 (동아시아, 남아시아, 중동, 유럽, 아프리카, 남미 등)
# - 5개 연령대 (어린이, 청소년, 청년, 중년, 노년)
# - 3개 성별 (남성, 여성, 논바이너리)
# - 다양한 조명, 배경, 표정, 헤어스타일
```

## SLURM 배치 실행

### HunyuanImage-2.1 대량 생성
```bash
# run_hunyuan.sh 설정
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:59:00

# 실행
sbatch run_hunyuan.sh

# 진행상황 확인
squeue -u $USER
tail -f logs/hunyuan_*.out
watch -n 60 'ls outputs/hunyuan_dataset/*.png | wc -l'
```

### 생성 재개 (중단 시)
- 스크립트가 자동으로 기존 이미지 스킵
- 새로운 프롬프트만 생성 (중복 방지)
- 프롬프트는 `outputs/hunyuan_dataset/prompts.txt`에 누적 저장

## 주요 파라미터

### HunyuanImage-2.1
```python
WIDTH = 2048
HEIGHT = 2048
NUM_STEPS = 50
GUIDANCE = 3.5
SHIFT = 5
use_fp8 = True
use_refiner = False  # 200GB RAM 필요
use_reprompt = False  # 상세 프롬프트로 대체
```

### FLUX dev
```python
width = 1024
height = 1024
num_steps = 50
guidance = 3.5
```

### FLUX schnell
```python
width = 512
height = 512
num_steps = 4
guidance = 0.0  # schnell은 guidance-free
```

## 메모리 최적화

### GPU VRAM (24GB 제한)
- FP8 quantization 활성화
- CPU offloading 활성화
- Resolution 조정 (2048 → 1024)

### System RAM (32GB 제한)
- Refiner 비활성화 (200GB 필요)
- Reprompt 비활성화 (14GB 필요)
- 명시적 메모리 해제 (`del`, `gc.collect()`)

## 출력 구조

```
outputs/
└── hunyuan_dataset/
    ├── fake_00000_00.png
    ├── fake_00000_01.png
    ├── ...
    ├── metadata.jsonl      # 프롬프트, seed, 생성시간 등
    └── prompts.txt         # 전체 프롬프트 목록 (중복 방지)
```

## 트러블슈팅

### OOM (GPU)
```bash
# FP8 확인
use_fp8 = True

# Resolution 낮추기
WIDTH = 1024
HEIGHT = 1024
```

### OOM (System RAM)
```bash
# Refiner/Reprompt 비활성화
use_refiner = False
use_reprompt = False

# 메모리 해제 강화
import gc
del image
torch.cuda.empty_cache()
gc.collect()
```

### 모델 다운로드 실패
```bash
# Resume download
huggingface-cli download tencent/HunyuanImage-2.1 \
  --local-dir ./ckpts \
  --resume-download
```

### Flash Attention Error
```bash
pip uninstall flash-attn -y
pip install flash-attn==2.7.3 --no-build-isolation
```

## 파일 설명

### 핵심 스크립트
- `scripts/test_hunyuan.py` - HunyuanImage-2.1 단일 테스트
- `scripts/generate_hunyuan_dataset.py` - HunyuanImage-2.1 배치 생성
- `scripts/generate_test.py` - FLUX dev 테스트
- `scripts/generate_schnell_512.py` - FLUX schnell 테스트
- `prompts_deepfake.py` - 프롬프트 자동 생성기

### SLURM 스크립트
- `run_hunyuan.sh` - HunyuanImage-2.1 배치 실행

## 데이터셋 특징

- **단일 인물**: 한 사람만 포함
- **중앙 배치**: 얼굴이 프레임 중앙에 위치
- **고해상도**: 2048x2048 (HunyuanImage-2.1)
- **다양성**: 70+ 인종, 5 연령대, 3 성별
- **현실적**: Photorealistic rendering, 피부 텍스처, 머리카락 디테일
- **Professional**: 스튜디오 조명, 깔끔한 배경

## 참고 자료

- [HunyuanImage-2.1 GitHub](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)
- [FLUX GitHub](https://github.com/black-forest-labs/flux)
- [HuggingFace Models](https://huggingface.co/tencent)

## 라이선스

- HunyuanImage-2.1: Tencent License
- FLUX schnell: Apache-2.0
- FLUX dev: Non-commercial use only

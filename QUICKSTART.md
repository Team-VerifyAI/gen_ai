# Quick Start Guide

가장 빠르게 시작하는 방법 (HunyuanImage-2.1)

## 1. 사전 준비 (5분)

```bash
# 1. 저장소 클론
cd /data/your_username/repos
git clone https://github.com/your-username/gen_image.git
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git

# 2. Conda 환경 생성
conda create -n hunyuan python=3.10 -y
conda activate hunyuan

# 3. 의존성 설치 (HunyuanImage-2.1 디렉토리에서)
cd HunyuanImage-2.1
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation  # 10-20분 소요
```

## 2. 모델 다운로드 (1-2시간)

```bash
# HuggingFace 토큰 발급: https://huggingface.co/settings/tokens

# 환경변수 설정
export HF_HOME=/data/your_username/.cache/huggingface
export HF_TOKEN=hf_your_token_here

# 모델 다운로드 (173GB)
cd /data/your_username/repos/HunyuanImage-2.1
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ./ckpts
```

## 3. 경로 설정 (5분)

**파일 수정 목록:**
- `gen_image/run_hunyuan.sh`
- `gen_image/scripts/test_hunyuan.py`
- `gen_image/scripts/generate_hunyuan_dataset.py`

**수정 방법:**

### run_hunyuan.sh
```bash
cd /data/your_username/repos/gen_image
nano run_hunyuan.sh

# 수정할 부분 (4곳):
cd /data/your_username/repos/gen_image
export HF_HOME=/data/your_username/.cache/huggingface
export TMPDIR=/data/your_username/.tmp
export HF_TOKEN=hf_your_token_here
source /data/your_username/anaconda3/etc/profile.d/conda.sh
```

### scripts/test_hunyuan.py 및 generate_hunyuan_dataset.py
```bash
nano scripts/test_hunyuan.py
nano scripts/generate_hunyuan_dataset.py

# 각 파일 상단 수정 (3곳):
sys.path.insert(0, '/data/your_username/repos/HunyuanImage-2.1')
sys.path.insert(0, '/data/your_username/repos/gen_image')
os.chdir('/data/your_username/repos/HunyuanImage-2.1')
```

## 4. 테스트 실행 (5분)

```bash
cd /data/your_username/repos/gen_image
conda activate hunyuan
python scripts/test_hunyuan.py

# 성공 시 outputs/hunyuan_test/ 디렉토리에 이미지 생성됨
```

## 5. 배치 생성 (23시간 59분)

```bash
# SLURM 파티션 확인
sinfo

# run_hunyuan.sh에서 파티션 수정
nano run_hunyuan.sh
# #SBATCH --partition=your_gpu_partition

# 배치 작업 제출
sbatch run_hunyuan.sh

# 진행상황 확인
squeue -u $USER
tail -f logs/hunyuan_*.out
watch -n 60 'ls outputs/hunyuan_dataset/*.png | wc -l'
```

## 체크리스트

- [ ] Conda 환경 생성 완료
- [ ] Flash Attention 설치 완료
- [ ] HunyuanImage-2.1 모델 다운로드 완료 (173GB)
- [ ] HuggingFace 토큰 발급 및 설정
- [ ] 환경변수 설정 (~/.bashrc)
- [ ] run_hunyuan.sh 경로 수정 (5곳)
- [ ] scripts/test_hunyuan.py 경로 수정 (3곳)
- [ ] scripts/generate_hunyuan_dataset.py 경로 수정 (3곳)
- [ ] SLURM 파티션 확인 및 설정
- [ ] 테스트 실행 성공
- [ ] 배치 작업 제출

## 문제 해결

### Flash Attention 설치 실패
```bash
# CUDA 버전 확인
nvcc --version

# 재설치
pip uninstall flash-attn -y
pip install flash-attn==2.7.3 --no-build-isolation
```

### 모델 다운로드 중단
```bash
# Resume download
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ./ckpts --resume-download
```

### OOM Error
```bash
# scripts/generate_hunyuan_dataset.py에서 해상도 낮추기
WIDTH = 1024  # 2048에서 1024로 변경
HEIGHT = 1024
```

### "No such file or directory" 에러
```bash
# 경로 확인
ls /data/your_username/repos/HunyuanImage-2.1
ls /data/your_username/repos/gen_image

# 경로가 맞는지 확인 후 스크립트 수정
```

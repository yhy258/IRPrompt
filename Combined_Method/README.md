# All-in-One Image Restoration Pipeline

이 프로젝트는 "Degradation-Aware Feature Perturbation for All-in-One Image Restoration" 연구의 데이터 파이프라인을 참고하여 여러 열화에 대한 이미지 복원을 수행하는 통합 파이프라인입니다.

## 방법론

### 1. Autoencoder 구조 (Taming Transformers VQGAN 기반)
- **E_G**: 깨끗한 이미지를 위한 인코더
- **D**: 디코더
- **훈련**: MSE Loss를 사용하여 깨끗한 이미지들로 훈련

### 2. RCG (Return of Unconditional Generation) 적용
- Autoencoder의 bottleneck에서의 latent vector에 diffusion process 적용
- Latent vector 자체에 대한 diffusion 모델 훈련

### 3. Encoder Alignment (E_D 훈련)
- **E_D**: 열화된 이미지를 위한 인코더 (E_G의 파라미터를 복사하여 초기화)
- **목표**: z_D (열화된 이미지의 latent)와 z_G (깨끗한 이미지의 latent) 간의 유사도 최대화
- **훈련**: E_G와 D는 freeze, E_D만 업데이트

### 4. 추론 파이프라인
1. 열화된 이미지를 E_D에 넣어서 z_D 도출
2. z_D에 대해 RCG diffusion process 적용
3. Diffusion process가 적용된 z_D를 D에 넣어서 복원

## 데이터셋

다음 6개 데이터셋의 깨끗한 이미지들을 사용:
- **Denoising**: BSD400, WED
- **Dehazing**: SOTS (indoor/outdoor)
- **Deraining**: Rain100L
- **Low-light**: LOL-v1
- **Deblurring**: GoPro

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install torch torchvision pytorch-lightning diffusers omegaconf pillow
```

### 2. 데이터셋 준비
```
dataset/
├── BSD400/
├── WED/
├── SOTS/
│   ├── indoor/
│   │   ├── hazy/
│   │   └── gt/
│   └── outdoor/
│       ├── hazy/
│       └── gt/
├── Rain100L/
├── lol_dataset/
│   └── our485/
│       ├── low/
│       └── high/
└── gopro/
    └── train/
        └── [sequence]/
            ├── blur/
            └── sharp/
```

## 사용법

### 1. 전체 파이프라인 훈련 (권장)

```bash
python train_full_pipeline.py \
    --data_root ./dataset \
    --output_dir ./outputs \
    --test_image ./test_images/degraded.jpg
```

### 2. 단계별 훈련

#### Step 1: Autoencoder 훈련
```bash
python train_autoencoder.py \
    --config autoencoder_config.yaml \
    --data_root ./dataset
```

#### Step 2: RCG 모델 훈련
```bash
cd ../rcg
python main_rdm.py \
    --config config/rdm_config.yaml \
    --data_path ../dataset \
    --output_dir ../outputs/rcg
```

#### Step 3: Encoder Alignment 훈련
```bash
python train_encoder_alignment.py \
    --pretrained_autoencoder_path ./checkpoints/autoencoder.ckpt \
    --data_root ./dataset \
    --batch_size 16 \
    --learning_rate 1e-4
```

### 3. 추론

```bash
python inference_pipeline.py \
    --autoencoder_path ./checkpoints/autoencoder.ckpt \
    --encoder_degraded_path ./checkpoints/encoder_alignment.ckpt \
    --rcg_config_path ../rcg/config/rdm_config.yaml \
    --input_path ./test_images/degraded.jpg \
    --output_path ./results/restored.jpg
```

## Stable Diffusion VQVAE 사용

기존 Autoencoder 대신 Stable Diffusion의 pretrained VQVAE를 사용할 수 있습니다:

```bash
python train_full_pipeline.py \
    --data_root ./dataset \
    --output_dir ./outputs \
    --use_stable_diffusion \
    --test_image ./test_images/degraded.jpg
```

## 설정 파일

### Autoencoder 설정 (`autoencoder_config.yaml`)
```yaml
model:
  base_learning_rate: 4.5e-6
  target: taming.models.vqgan.Autoencoder
  params:
    ddconfig:
      double_z: False
      z_channels: 256
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [1,1,2,2,4]
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0
    loss_params:
      lpips_weight: 0.0

data:
  params:
    batch_size: 16
    num_workers: 8

lightning:
  trainer:
    accelerator: gpu
    max_epochs: 100
  logger:
    target: pytorch_lightning.loggers.TensorBoardLogger
    params:
      save_dir: "logs/"
      name: "autoencoder_training"
```

## 파일 구조

```
Combined_Method/
├── train_autoencoder.py          # Step 1: Autoencoder 훈련
├── train_encoder_alignment.py    # Step 3: Encoder Alignment 훈련
├── inference_pipeline.py         # Step 4: 추론 파이프라인
├── train_full_pipeline.py        # 전체 파이프라인 통합 스크립트
├── stable_diffusion_vqvae.py     # Stable Diffusion VQVAE 래퍼
├── data.py                       # 데이터셋 클래스
├── autoencoder_config.yaml       # Autoencoder 설정
├── taming/                       # Taming Transformers 모델
│   └── models/
│       └── vqgan.py
├── checkpoints/                  # 훈련된 모델 체크포인트
├── logs/                         # 훈련 로그
└── README.md                     # 이 파일
```

## 성능 최적화 팁

1. **GPU 메모리**: 배치 크기를 GPU 메모리에 맞게 조정
2. **데이터 로딩**: `num_workers`를 CPU 코어 수에 맞게 설정
3. **Mixed Precision**: 대용량 모델의 경우 `precision=16` 사용
4. **Gradient Accumulation**: 메모리 부족 시 `accumulate_grad_batches` 사용

## 문제 해결

### 일반적인 오류

1. **CUDA out of memory**: 배치 크기 줄이기
2. **데이터셋 경로 오류**: 데이터셋 구조 확인
3. **모델 로딩 오류**: 체크포인트 경로 확인

### 디버깅

```bash
# 상세한 로그 출력
python train_autoencoder.py --config autoencoder_config.yaml --data_root ./dataset --debug

# GPU 사용량 모니터링
nvidia-smi -l 1
```

## 라이선스

이 프로젝트는 연구 목적으로만 사용되어야 합니다.

## 참고 문헌

1. "Taming Transformers for High-Resolution Image Synthesis"
2. "Return of Unconditional Generation: A Self-supervised Representation Generation Method"
3. "Degradation-Aware Feature Perturbation for All-in-One Image Restoration"







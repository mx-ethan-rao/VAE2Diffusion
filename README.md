<div align="center">
<h1>Latent Diffusion Inversion Requires Understanding the Latent Space (CVPR 2026)</h1>

<a href="https://arxiv.org/abs/2511.20592"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>

**[VINE Lab, Vanderbilt University](https://vine-lab.notion.site/)**

[Mingxing (Ethan) Rao](https://mx-ethan-rao.github.io/), [Bowen Qu](https://www.linkedin.com/in/bowen-qu-b852b724a/), [Daniel Moyer](https://dcmoyer.github.io/)
</div>

<p align="center">
<img src=figures/MainFig.png />
</p>

**Abstract:**The recovery of training data from generative models ("model inversion") has been extensively studied for diffusion models in the data domain as a memorization/overfitting phenomenon. Latent diffusion models (LDMs), which operate on the latent codes from encoder/decoder pairs, have been robust to prior inversion methods. In this work we describe two key findings: (1) the diffusion model exhibits non-uniform memorization across latent codes, tending to overfit samples located in high-distortion regions of the decoder pullback metric; (2) even within a single latent code, memorization contributions are unequal across representation dimensions. Our proposed method to ranks latent dimensions by their contribution to the decoder pullback metric, which in turn identifies dimensions that contribute to memorization. For score-based membership inference, a sub-task of model inversion, we find that removing less-memorizing dimensions improves performance on all tested methods and datasets, with average AUROC gains of 1-4% and substantial increases in TPR@1%FPR (1-32%) across diverse datasets including CIFAR-10, CelebA, ImageNet-1K, Pokémon, MS-COCO, and Flickr. Our results highlight the overlooked influence of the auto-encoder geometry on LDM memorization and provide a new perspective for analyzing privacy risks in diffusion-based generative models.



  
## Requirements
Please refer to the environment setting for [diffuser](https://github.com/huggingface/diffusers).

## Experiments
Please download all dataset splits and checkpoints [here](). Notably, the default attacker are [SimA](https://arxiv.org/abs/2509.25003)

### <a id="ldm-light"></a>MNIST, CIFAR-10, CelebA
#### Working dir
```
cd ldm_light
```
#### Train a LDM
```
# mnist and cifar-10
python main.py --dataset cifar10 --dataset_root /path/to/datasets --split_file /path/to/splits/CIFAR10_train_ratio0.5.npz --out_dir /path/to/output --kl_beta 1e-2 --skip_vqvae --skip_ldm_vq

# celeba
python main.py --dataset celeba --dataset_root /path/to/celeba --split_file /path/to/splits/CELEBA_train_ratio0.5.npz --out_dir /path/to/output --img_size 64 --in_channels 3 --latent_channels 4 --ae_base 128 --kl_beta 1e-3 --unet_model_ch 224 --unet_channel_mult 1 2 3 4 --batch_size 256 --num_workers 8 --skip_vqvae --skip_ldm_vq --epochs_ldm 512
```

#### Computing the pullback metric for each data
```
# mnist and cifar-10
python cal_pullback.py --dataset cifar10 --split-file /path/to/splits/CIFAR10_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --dataset-root /path/to/data --out /path/to/output/logvols_cifar10_beta_1e_2.npz --device cuda --batch-size 32 --k 50 --n-iter 30

# celeba
python cal_pullback.py --dataset celeba --split-file /path/to/splits/CELEBA_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --dataset-root /path/to/celeba_data --out /path/to/output/logvols_celeba_beta_1e_3.npz --device cuda --batch-size 32 --k 50 --n-iter 30 --img-size 64
```

#### Computing the per-dimensional contribution to distortion
```
# mnist and cifar-10
python cal_per_dim_contri.py --dataset cifar10 --split-file /path/to/splits/CIFAR10_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --dataset-root /path/to/data --out /path/to/output/per_dim_cifar10_beta_1e_2.npz --device cuda --n-mc 16

# celeba
python cal_per_dim_contri.py --dataset celeba --split-file /path/to/splits/CELEBA_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --dataset-root /path/to/celeba_data --out /path/to/output/per_dim_celeba_beta_1e_3.npz --device cuda --n-mc 16 --img-size 64
```

#### MIA for different distortion quartiles
```
# mnist and cifar-10
python attack.py --split-file /path/to/splits/CIFAR10_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --unet-ckpt /path/to/ldm_vae/unet_last.pt --logvol /path/to/results/logvols_cifar10_beta_1e_2.npz --device cuda --grouping median --batch-size 64

# celeba
python attack_per_dim_stream.py --dataset celeba --dataset-root /path/to/celeba_data --split-file /path/to/splits/CELEBA_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --unet-ckpt /path/to/ldm_vae/unet_last.pt --logvol /path/to/results/logvols_celeba_beta_1e_3.npz --perdim /path/to/results/per_dim_celeba_beta_1e_3.npz --img-size 64 --unet_model_ch 224 --unet_channel_mult 1 2 3 4 --device cuda --grouping quartiles --drop-percent 0.0
```

#### Removing 40% less-memorizing dimensions (the ratio is adjustable)
```
# mnist and cifar-10
python attack_per_dim.py --dataset cifar10 --dataset-root /path/to/datasets --split-file /path/to/splits/CIFAR10_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --unet-ckpt /path/to/ldm_vae/unet_last.pt --logvol /path/to/results/logvols_cifar10_beta_1e_2.npz --perdim /path/to/results/per_dim_cifar10_beta_1e_2.npz --device cuda --grouping random --random_groups 1 --drop-percent 40.0

# celeba
python attack_per_dim_stream.py --dataset celeba --dataset-root /path/to/celeba_data --split-file /path/to/splits/CELEBA_train_ratio0.5.npz --vae-ckpt /path/to/vae/vae_last.pt --unet-ckpt /path/to/ldm_vae/unet_last.pt --logvol /path/to/results/logvols_celeba_beta_1e_3.npz --perdim /path/to/results/per_dim_celeba_beta_1e_3.npz --img-size 64 --unet_model_ch 224 --unet_channel_mult 1 2 3 4 --device cuda --grouping random --random_groups 1 --drop-percent 40.0
```


### <a id="ldm4imagenet"></a>TinyImageNet (100K images)
Experimental data is provided above. For LDM on ImageNet, we resue the VAE encoder from publicly released autoencoder model
[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)

#### Working dir
```
cd ldm4imagenet
```
#### Train a LDM
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN \
accelerate launch --multi_gpu main.py --data_root /path/to/imagenet100k --out_dir /path/to/output_dir --amp --per_device_batch 21 --unet_epochs 600
```

#### Computing the pullback metric for each data
```
python cal_pullback.py --data_root /path/to/imagenet100k --out_dir /path/to/output_dir --out_npz /path/to/output_dir/imnetv1_10k_pullback.npz --method fd
```

#### Computing the per-dimensional contribution to distortion
```
python cal_per_dim_contri.py --data_root /path/to/imagenet100k --out_dir /path/to/output_dir --pullback_npz /path/to/output_dir/imnetv1_10k_pullback.npz --out_npz /path/to/output_dir/imnetv1_10k_per_dim.npz
```

#### MIA for different distortion quartiles
```
python attack_by_group_advance.py --data_root /path/to/imagenet100k --out_dir /path/to/output_dir --pullback_npz /path/to/output_dir/imnetv1_10k_pullback.npz
```

#### Removing 40% less-memorizing dimensions (the ratio is adjustable)
```
python attack_per_dim.py --data_root /path/to/imagenet100k/data --out_dir /path/to/output_dir --pullback_npz /path/to/output_dir/imnetv1_10k_pullback.npz --perdim_npz /path/to/output_dir/imnetv1_10k_per_dim.npz --grouping random --random_groups 1 --drop_percent 40.0
```

### <a id="diffusers"></a>Pokemon, MS-COCO, and Flickr
Experimental data is provided above. The code base is built on top of [diffuser](https://github.com/huggingface/diffusers). We fine-tuned a [Stable Diffusion V1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) on those datasets.

#### Working dir
```
cd diffusers
```
#### Train a LDM
```
accelerate launch examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/path/to/POKEMON/pokemon_blip_splits" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/path/to/POKEMON/logs/sd-pokemon-model" 
```

#### Computing the pullback metric for each data
```
python -m src.mia.cal_pullback.py --dataset pokemon --ckpt-path /path/to/POKEMON/logs/sd-pokemon-model --out-npz /path/to/output_dir/sd_pokemon_pullback.npz
```

#### MIA for different distortion quartiles
```
python -m src.mia.attack_by_group.py --dataset pokemon --ckpt-path /path/to/POKEMON/logs/sd-pokemon-model --logvol-npz /path/to/output_dir/sd_pokemon_pullback.npz
```

#### Computing the per-dimensional contribution to distortion & Removing 40% less-memorizing dimensions
```
python -m src.mia.attack_per_dim --attacker SimA --dataset pokemon --ckpt-path /path/to/checkpoint/sd-pokemon-model --perdim_npz /path/to/results/per_dim_pokemon_diffuser.npz --drop_percent 40.0
```



## BibTeX

```
@article{rao2025latent,
  title={Latent Diffusion Inversion Requires Understanding the Latent Space},
  author={Rao, Mingxing and Qu, Bowen and Moyer, Daniel},
  journal={arXiv preprint arXiv:2511.20592},
  year={2025}
}
```



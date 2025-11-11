python cal_log_volume.py \
  --dataset mnist \
  --split-file /home/ethanrao/MIA_LDM/data/mnist_mia_split.npz \
  --vae-ckpt /data/mingxing/tmp/MNIST/KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/data \
  --out /data/mingxing/tmp/MNIST/KL_sweep/1e_2/logvols_mnist_beta_1e_2.npz \
  --device cuda \
  --batch-size 32 \
  --k 50 \
  --n-iter 30

python cal_log_volume.py \
  --split-file  /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/data \
  --out /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/logvols_cifar10_beta_1e_2.npz \
  --device cuda \
  --batch-size 32 \
  --k 50 \
  --n-iter 30


python attack.py \
  --split-file /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /home/ethanrao/MIA_LDM/data/logvols_cifar10_beta_1e_2.npz \
  --device cuda \
  --grouping median \
  --batch-size 64


python cal_per_dim_contri.py \
  --dataset cifar10 \
  --split-file /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/data \
  --out /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/per_dim_cifar10_beta_1e_2.npz \
  --device cuda --n-mc 16

python attack_per_dim.py \
  --dataset cifar10 \
  --dataset-root /home/ethanrao/MIA_LDM/VAE2Diffusion/data/ \
  --split-file /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/logvols_cifar10_beta_1e_2.npz \
  --perdim /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/per_dim_cifar10_beta_1e_2.npz \
  --device cuda \
  --grouping random

###################celeba########################
python cal_log_volume.py \
  --dataset celeba \
  --split-file  /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/CELEBA_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/VAE2Diffusion/data/celeba \
  --out /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/logvols_celeba_beta_1e_3.npz \
  --device cuda \
  --batch-size 32 \
  --k 50 \
  --n-iter 30 \
  --img-size 64

python cal_per_dim_contri.py \
  --dataset celeba \
  --split-file /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/CELEBA_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/VAE2Diffusion/data/celeba \
  --out /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/per_dim_celeba_beta_1e_3.npz \
  --device cuda --n-mc 16 --img-size 64

python attack_per_dim_stream.py \
  --dataset celeba \
  --dataset-root /home/ethanrao/MIA_LDM/VAE2Diffusion/data/celeba \
  --split-file /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/CELEBA_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/vae/vae_last.pt \
  --unet-ckpt /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/ldm_vae/unet_last.pt \
  --logvol /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/logvols_celeba_beta_1e_3.npz \
  --perdim /data/mingxing/tmp/CELEBA/KL_sweep/1e_3/per_dim_celeba_beta_1e_3.npz \
  --img-size 64 \
  --unet_model_ch 224 --unet_channel_mult 1 2 3 4   \
  --device cuda \
  --grouping random \
  --random_groups 1


###################mnist########################

python cal_per_dim_contri.py \
  --dataset mnist \
  --split-file /data/mingxing/tmp/MNIST/KL_sweep/1e_2/mnist_mia_split.npz \
  --vae-ckpt /data/mingxing/tmp/MNIST/KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/VAE2Diffusion/data \
  --out /data/mingxing/tmp/MNIST/KL_sweep/1e_2/per_dim_mnist_beta_1e_2.npz \
  --device cuda --n-mc 16

python attack_per_dim.py \
  --dataset mnist \
  --dataset-root /home/ethanrao/MIA_LDM/VAE2Diffusion/data \
  --split-file /data/mingxing/tmp/MNIST/KL_sweep/1e_2/mnist_mia_split.npz \
  --vae-ckpt /data/mingxing/tmp/MNIST/KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /data/mingxing/tmp/MNIST/KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /data/mingxing/tmp/MNIST/KL_sweep/1e_2/logvols_mnist_beta_1e_2.npz \
  --perdim /data/mingxing/tmp/MNIST/KL_sweep/1e_2/per_dim_mnist_beta_1e_2.npz \
  --device cuda \
  --grouping random \
  --random_groups 1 \
  --batch-size 64

python attack.py \
  --dataset mnist \
  --split-file /home/ethanrao/MIA_LDM/data/mnist_mia_split.npz \
  --vae-ckpt /data/mingxing/tmp/MNIST/KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /data/mingxing/tmp/MNIST/KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /home/ethanrao/MIA_LDM/data_backup/logvols_mnist_beta_1e_2.npz \
  --device cuda \
  --grouping quartiles \
  --batch-size 64


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
  --split-file  /home/ethanrao/MIA_LDM/data/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/data \
  --out /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/logvols_cifar10_beta_1e_2.npz \
  --device cuda \
  --batch-size 32 \
  --k 50 \
  --n-iter 30


python attack.py \
  --split-file /home/ethanrao/MIA_LDM/data/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /data/mingxing/tmp/CIFAR10/KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /home/ethanrao/MIA_LDM/data/logvols_cifar10_beta_1e_2.npz \
  --device cuda \
  --grouping median \
  --batch-size 64


python cal_per_dim_contri.py \
  --dataset cifar10 \
  --split-file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /banana/ethan/MIA_LDM_data/KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/data \
  --out /home/ethanrao/MIA_LDM/data/per_dim_cifar10_beta_1e_2.npz \
  --device cuda

python attack_per_dim.py \
  --dataset cifar10 \
  --split-file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --vae-ckpt /banana/ethan/MIA_LDM_data/KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /banana/ethan/MIA_LDM_data/KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /home/ethanrao/MIA_LDM/data/logvols_cifar10_beta_1e_2.npz \
  --perdim /home/ethanrao/MIA_LDM/data/per_dim_cifar10_beta_1e_2.npz \
  --device cuda \
  --grouping quartiles \
  --batch-size 64




###################mnist########################

python cal_per_dim_contri.py \
  --dataset mnist \
  --split-file /home/ethanrao/MIA_LDM/data/mnist_mia_split.npz \
  --vae-ckpt /banana/ethan/MIA_LDM_data/MNIST_KL_sweep/1e_2/vae/vae_last.pt \
  --dataset-root /home/ethanrao/MIA_LDM/data \
  --out /home/ethanrao/MIA_LDM/ldm_light/log_volume/per_dim_mnist_beta_1e_2.npz \
  --device cuda

python attack_per_dim.py \
  --dataset mnist \
  --split-file /home/ethanrao/MIA_LDM/data/mnist_mia_split.npz \
  --vae-ckpt /banana/ethan/MIA_LDM_data/MNIST_KL_sweep/1e_2/vae/vae_last.pt \
  --unet-ckpt /banana/ethan/MIA_LDM_data/MNIST_KL_sweep/1e_2/ldm_vae/unet_last.pt \
  --logvol /home/ethanrao/MIA_LDM/data/logvols_mnist_beta_1e_2.npz \
  --perdim /home/ethanrao/MIA_LDM/data/per_dim_mnist_beta_1e_2.npz \
  --device cuda \
  --grouping quartiles \
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


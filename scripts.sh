python attack.py --dataset cifar10 \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir /banana/ethan/MIA_LDM_data/KL_sweep/1e_3 \
  --probe_min_t 0 --probe_max_t 300 --probe_step 10


CUDA_VISIBLE_DEVICES=7 python main.py \
  --dataset cifar10 \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir /banana/ethan/MIA_LDM_data/KL_sweep/1e_2 \
  --kl_beta 1e-2 \
  --skip_vqvae \
  --skip_ldm_vq
  


CUDA_VISIBLE_DEVICES=7 python main.py \
  --dataset cifar10 \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir /banana/ethan/MIA_LDM_data/VQ_sweep/1.5 \
  --vq_lambda 1.5 \
  --skip_vae \
  --skip_ldm_vae

python cal_recon_loss.py --dataset cifar10 \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir /banana/ethan/MIA_LDM_data/KL_sweep/1e_3


python main_gan.py \
  --dataset cifar10 \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir /banana/ethan/MIA_LDM_data/GAN_trick_vae/1e_3 \
  --kl_beta 1e-3 \
  --skip_vqvae \
  --skip_ldm_vq \
  --gan_enable \
  --disc_start 10000 \
  --disc_weight 0.8 \
  --disc_type hinge

python main_gan_percep.py \
  --dataset cifar10 \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir /banana/ethan/MIA_LDM_data/GAN_trick__percep_vae/0 \
  --kl_beta 0 \
  --skip_vqvae \
  --skip_ldm_vq \
  --gan_enable \
  --disc_start 2000 \
  --disc_weight 0.8 \
  --disc_type hinge \
  --perceptual_weight 1.0




python main.py \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --out_dir outputs_ldm_toy_compvis \
  --vae_ckpt outputs_ldm_toy_compvis/vae/vae_last.pt \
  --unet_vae_ckpt outputs_ldm_toy_compvis/ldm_vae/unet_last.pt \
  --skip_vqvae --skip_ldm_vq


python sampler.py --out_dir outputs_ldm_toy_compvis --device cuda


python ldm_fid_from_attack.py \
  --dataset cifar10 --dataset_root /home/ethanrao/MIA_LDM/data \
  --split_file /banana/ethan/MIA_data/CIFAR10/CIFAR10_train_ratio0.5.npz \
  --img_size 32 --in_channels 3 --latent_channels 4 \
  --vae_ckpt /banana/ethan/MIA_LDM_data/GAN_trick_vae/1/vae/vae_last.pt \
  --unet_vae_ckpt /banana/ethan/MIA_LDM_data/GAN_trick_vae/1/ldm_vae/unet_last.pt \
  --sampler ddpm \
  --n_gen 1000 --batch_gen 64 --batch_fid 128






  #####################################mnist###################################################
python main.py \
  --split_file /home/ethanrao/MIA_LDM/data/mnist_mia_split.npz \
  --dataset mnist \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --kl_beta 1e-2 \
  --out_dir /banana/ethan/MIA_LDM_data/MNIST_KL_sweep/1e_2 \
  --skip_vqvae --skip_ldm_vq


python attack.py --dataset mnist \
  --ae_base 64 \
  --unet_model_ch 64 \
  --unet_channel_mult 1 2 2 \
  --img_size 32 \
  --split_file /home/ethanrao/MIA_LDM/data/mnist_mia_split.npz \
  --dataset_root /home/ethanrao/MIA_LDM/data \
  --out_dir /banana/ethan/MIA_LDM_data/MNIST_KL_sweep/1e_3 \
  --probe_min_t 0 --probe_max_t 300 --probe_step 10
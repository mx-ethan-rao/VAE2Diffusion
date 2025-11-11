##################################mscoco##############

CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack_per_dim \
    --attacker SimA     \
    --dataset coco   \
    --ckpt-path /data/mingxing/tmp/MSCOCO/logs/sd-mscoco-model \
    --perdim_npz  /data/mingxing/tmp/MSCOCO/logs/per_dim_coco2017val_diffuser.npz \
    --drop_percent 0.0

CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack_per_dim \
    --attacker SimA     \
    --dataset coco   \
    --ckpt-path /data/mingxing/tmp/MSCOCO/logs/sd-mscoco-model \
    --perdim_npz  /data/mingxing/tmp/MSCOCO/logs/per_dim_coco2017val_diffuser.npz


##################################pokemon################

CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack_per_dim \
    --attacker SimA     \
    --dataset pokemon   \
    --ckpt-path /data/mingxing/tmp/POKEMON/logs/sd-pokemon-model \
    --perdim_npz  /data/mingxing/tmp/POKEMON/logs/per_dim_pokemon_diffuser.npz \
    --drop_percent 0.0



#########################Flickr30k#########################
CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack_per_dim \
    --attacker SimA     \
    --dataset flickr   \
    --ckpt-path /data/mingxing/tmp/FLICKR30K/logs/sd-flickr-model \
    --perdim_npz  /data/mingxing/tmp/FLICKR30K/logs/sd-flickr-model/per_dim_flickr_diffuser.npz \
    --drop_percent 0.0

#########################laion-aesthetic_laion-multitrans#########################


CUDA_VISIBLE_DEVICES=7 python -m src.mia.attack_per_dim \
    --attacker SimA     \
    --dataset laion-aesthetic_laion-multitrans   \
    --ckpt-path runwayml/stable-diffusion-v1-5 \
    --perdim_npz /data/mingxing/tmp/LAION5k/per_dim_laion_diffuser.npz \
    --drop_percent 0.0

CUDA_VISIBLE_DEVICES=6 python -m src.mia.attack_per_dim \
    --attacker SimA     \
    --dataset laion-aesthetic_laion-multitrans   \
    --ckpt-path runwayml/stable-diffusion-v1-5 \
    --perdim_npz /data/mingxing/tmp/LAION5k/per_dim_laion_diffuser.npz
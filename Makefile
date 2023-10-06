resnet18:
	python3 main_dino.py --timm_arch resnet18 \
	--data_path ../dataset/ --output_dir ./output/resnet18 \
	--batch_size_per_gpu 64 --num_workers 15 --saveckp_freq 10

convnextv2_small:
	python3 main_dino.py --timm_arch convnextv2_small \
	--data_path ../dataset/ --output_dir ./output/convnextv2_small \
	--batch_size_per_gpu 64 --num_workers 15 --saveckp_freq 10

convnextv2_pico:
	python3 main_dino.py --timm_arch convnextv2_pico \
	--data_path ../dataset/ --output_dir ./output/convnextv2_pico \
	--batch_size_per_gpu 64 --num_workers 15 --saveckp_freq 10

vit_tiny:
	python3 main_dino.py --timm_arch vit_tiny \
	--data_path ../dataset/ --output_dir ./output/vit_tiny \
	--batch_size_per_gpu 64 --num_workers 15 --saveckp_freq 10

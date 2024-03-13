CUDA_VISIBLE_DEVICES=1 python test.py --cfg cfgs/test_CNNSpot0.1_224.yaml

CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test_CNNSpot0.5_224.yaml

CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test_CNNSpot0.1_noresize.yaml

CUDA_VISIBLE_DEVICES=2 python test.py --cfg cfgs/test_CNNSpot0.5_noresize.yaml

CUDA_VISIBLE_DEVICES=3 python test.py --cfg cfgs/test_msclip_224.yaml


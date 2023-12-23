from .base_options import BaseOptions



from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
# dataroot = '~/workspace/datasets/aiart/output2/sd'
# # dataroot = '~/workspace/datasets/aiart/output2/dalle'
# # dataroot = '~/workspace/datasets/aiart/output2/imagen'
# # dataroot = '~/workspace/datasets/aiart/output2/mj'
# # dataroot = '~/workspace/datasets/aiart/output2/parti'

dataroot = '/home/teddy/workspace/datasets/aiart/test'

# list of synthesis algorithms
vals = ['sd','dalle','mj','parti','imagen']

# indicates if corresponding testset has multiple classes
multiclass = [0,0,0,0,0]

# model
# model_path = '/home/wangyabin/workspace/CDDDet/checkpoints/vit_sd_blur_jpg_prob0.5/model_epoch_best.pth'
# model_path = '/home/wangyabin/workspace/CDDDet/checkpoints/vit_sd_blur_jpg_prob0.5/model_epoch_best.pth'
# model_path = '/home/wangyabin/workspace/CDDDet/checkpoints/resnet_sd_blur_jpg_prob0.1/model_epoch_best.pth'
# model_path = '/home/wangyabin/workspace/CDDDet/checkpoints/resnet_sd_blur_jpg_prob0.5/model_epoch_best.pth'
# model_path = '/home/wangyabin/workspace/CDDDet/weights/blur_jpg_prob0.1.pth'
# model_path = '/home/wangyabin/workspace/CDDDet/weights/blur_jpg_prob0.5.pth'
model_path = '/home/teddy/workspace/CDDDet/checkpoints/vit_sdft_set3/model_epoch_latest.pth'




class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser

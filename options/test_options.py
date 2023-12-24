from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path', type=str, required=True, help='Path to the model file to be loaded.')
        parser.add_argument('--output_dir', type=str, required=True, help='Directory where the outputs will be saved.')
        parser.add_argument('--dataroot', type=str, required=True, help='Root directory of the dataset.')
        parser.add_argument('--sub_dirs', type=str, default='',
                            help='Sub-directory within the data root directory, if applicable.')
        return parser

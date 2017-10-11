
__all__ = ['config']

from .utils.config import get_config
# from .utils.logger import get_log

# log = get_log('lvmsurveysim', log_file_path='~/.lvm/lvmsurveysim.log')
config = get_config('~/.lvm/lvmsurveysim.yaml')


__version__ = '0.1.0dev'


__all__ = ['config']

from .utils.config import get_config
# from .utils.logger import get_log

# log = get_log('lvmsim', log_file_path='~/.lvm/lvmsim.log')
config = get_config('~/.lvm/lvmsim.yaml')

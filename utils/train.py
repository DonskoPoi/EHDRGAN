import os.path

from basicsr.train import train_pipeline

import archs
import data
import models


if __name__ == '__main__':
    '''
    usage:
    python train.py -opt <yml路径> --luncher <pytorch> <--auto_resume> <--debug> <--local_rank>
    '''
    root_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))  # root_path定向到这个项目的根目录
    # print(root_path)
    train_pipeline(root_path)  # 调用basicsr的train模块开始训练

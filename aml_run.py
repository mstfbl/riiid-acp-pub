# This file acts as the main Python script to start the training logic,
# by first calling 01_pre.py and then calling 02_train.py

# Expected args: 

import argparse
import os
import datetime
import subprocess
import sys
import time

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    parser.add_argument('--questions_data_path', default='/path/to/riiid/questions/', type=str,
        help='Path to questions.csv')
    parser.add_argument('--lectures_data_path', default='/path/to/riiid/lectures/', type=str,
        help='Path to lectures.csv')
    parser.add_argument('--train_data_path', default='/path/to/riiid/train/', type=str,
        help='Path to train.csv')
    parser.add_argument('--train_launch_command', default='python -m torch.distributed.launch --master_port 1235 --nproc_per_node=8 02_train.py --epochs 15 --bs 64 --fp16 to_fp16 --trf_heads 4 --mixup False --chunk_size 500 --trf_dim 512 --loss ce --n_chunks 1 --fit fit_flat_cos --fit_kwargs pct_start=0.5 div_final=100 --tfixup True --pad r --valid_pct 0.025 --trf_act gelu --opt ranger_lamb --lr 3e-3 --torch_ort False --sccl False --model pt1110_sccl_epochs15_gpus8_bs64', type=str,
        help='02_train.py launch command')
    return parser

def main(args):
    subprocess.check_call(["pwd"])
    subprocess.check_call(["ls", "-la"])
    subprocess.check_call(["nvidia-smi"])

    start_time_01_pre = time.time()
    subprocess.check_call("ipython 01_pre.py {0} {1} {2}".format(args.questions_data_path, args.lectures_data_path, args.train_data_path).split())
    total_time_01_pre = str(datetime.timedelta(seconds=int(time.time() - start_time_01_pre)))
    print(f"[{datetime.datetime.now()}]:")
    print('01_pre.py run time {}'.format(total_time_01_pre))

    start_time_02_train = time.time()
    subprocess.check_call(args.train_launch_command.split())
    total_time_02_train = str(datetime.timedelta(seconds=int(time.time() - start_time_02_train)))
    print(f"[{datetime.datetime.now()}]:")
    print('Train run time {}'.format(total_time_02_train))

if __name__ == '__main__':
    print(f'[{datetime.datetime.now()}]: start RIIID...')
    parser = argparse.ArgumentParser('RIIID', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

import os,sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    ############# exp setting
    parser.add_argument('--name',         type = str, default=None)
    parser.add_argument('--rank',         type = int, default = 0)
    parser.add_argument('--n_proc',       type = int, default = 1)
    parser.add_argument('--gpu',          type = int, default=[0],nargs='*')
    parser.add_argument('--seed',         type=int,   default=10)
    parser.add_argument('--agent',        type=str,   default='coarse')
    parser.add_argument('--step_report',  default=False, action='store_true')
    parser.add_argument('--cont',         default=False, action='store_true')

    ############# ai2thor
    parser.add_argument('--x_display',type=str,default='0')
    
    ############# alfred
    parser.add_argument('--reward_config',type= str,default='./utils/rewards.json')
    parser.add_argument('--split', type = str, default = 'valid_seen', 
                        choices=['tests_seen', 'tests_unseen', 'train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--task_type',  type=str,default=None,nargs='*',choices=[
        'look_at_obj_in_light',
        'pick_and_place_simple',
        'pick_two_obj_and_place',
        'pick_and_place_with_movable_recep',
        'pick_cool_then_place_in_recep',
        'pick_heat_then_place_in_recep',
        'pick_clean_then_place_in_recep'
    ])
    parser.add_argument('--task_index', type=int,default=None,nargs='*')
    parser.add_argument('--task_name',  type=str,default=None)
    parser.add_argument('--task_from',  type=int,default=None)
    parser.add_argument('--task_to',    type=int,default=None)
    parser.add_argument('--task_scene', type=int,default=None,nargs='*')
    parser.add_argument('--remove_repeat', default=False,action='store_true')
    parser.add_argument('--max_steps', default=1000,type=int)
    parser.add_argument('--max_fails', default=10,type=int)
    parser.add_argument('--smooth_nav', default=False,action='store_true')
    
    
    ############# main agent
    parser.add_argument('--vis',           default=False,action='store_true')
    parser.add_argument('--use_gt_text',   default=False,action='store_true')
    parser.add_argument('--use_gt_depth',   default=False,action='store_true')
    parser.add_argument('--use_gt_seg',   default=False,action='store_true')
    parser.add_argument('--use_gt_aff',   default=False,action='store_true')
    parser.add_argument('--text_append',   default=False,action='store_true')
    parser.add_argument('--map_size',      default=100,  type = int)
    parser.add_argument('--map_unit',      default=250,  type = int)
    parser.add_argument('--obj_thresh',    default=0.5,  type = float)
    parser.add_argument('--dim_hidden',    default=256,  type = float)
    parser.add_argument('--optim_lr',      default=1e-2, type = float)
    parser.add_argument('--optim_steps',   default=10,   type = int)
    parser.add_argument('--optim_min_pts_obj', default=100,   type = int)
    parser.add_argument('--optim_min_pts_aff', default=500,   type = int)
    

    ############# replay
    parser.add_argument('--save_rgb', default=False,    action='store_true')
    parser.add_argument('--save_depth', default=False,  action='store_true')
    parser.add_argument('--save_seg', default=False,    action='store_true')
    parser.add_argument('--save_aff', default=False,    action='store_true')
    

    ############# train opt
    parser.add_argument('--amp',         default=False,action='store_true')
    parser.add_argument('--bz',          default=10,type=int)
    parser.add_argument('--epoch',       default=100,type=int)
    parser.add_argument('--num_workers', default=32,type=int)
    parser.add_argument('--resume',      default=None,type=str)
    parser.add_argument('--optimizer',   default='AdamW',type=str)
    parser.add_argument('--base_lr',     default=5e-4, type=float)
    parser.add_argument('--momentum',    default=0.9, type=float)
    parser.add_argument('--weight_decay',default=0.1, type=float)
    parser.add_argument('--beta1',       default=0.9, type=float)
    parser.add_argument('--beta2',       default=0.999, type=float)
    parser.add_argument('--lr_scheduler',default='warmup',type=str)
    parser.add_argument('--warmup_step', default=1000,type=int)
    parser.add_argument('--save_freq',   default=1,type= int)
    parser.add_argument('--report_freq', default=1,type= int)
    parser.add_argument('--test_freq',   default=1,type= int)
    parser.add_argument('--cosine',     default=False,action='store_true')


    ############# sam train
    parser.add_argument('--image_size',  default=300,type=int)
    parser.add_argument('--mask_size',   default=300,type=int)
    parser.add_argument('--center_only', default=False,action='store_true')
    parser.add_argument('--neg_rate',    default=0.4,type=float)
    parser.add_argument('--neg_min',     default=1,type=int)
    parser.add_argument('--neg_max',     default=10,type=int)
    parser.add_argument('--loss_dice',   default=1., type=float)
    parser.add_argument('--loss_focal',  default=20., type=float)
    parser.add_argument('--loss_class',  default=1., type=float)
    parser.add_argument('--loss_iou',    default=1., type=float)
    parser.add_argument('--n_per_side',  default=8, type=int)
    
    
    
    ############# depth unet train
    parser.add_argument('--depth_bins',  default=50,   type=int)
    parser.add_argument('--max_depth',   default=5000, type=float)
    parser.add_argument('--horizon',     default=None,nargs='*', type=int)
    

    
    
    
    args  = parser.parse_args()

    return args
from utils import arguments, utils
import os
import torch.multiprocessing as mp
import importlib
from env import thor_env
import time
from utils import image_util
import cv2
import pickle as pkl
import shutil
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import numpy as np
from gen import constants
# constants.RENDER_CLASS_IMAGE = False
# constants.RENDER_OBJECT_IMAGE = False
# thor_env.DEFAULT_RENDER_SETTINGS['renderObjectImage'] = False
# thor_env.DEFAULT_RENDER_SETTINGS['renderClassImage'] = False
FORCE_HORIZON = 0

def save_rgb(root,step,env):
    cv2.imwrite(os.path.join(root,'step_%05d.png' % step), cv2.cvtColor(env.last_event.frame,cv2.COLOR_BGR2RGB))

def save_depth(root, step, env):
    depth_image = env.last_event.depth_frame
    assert depth_image.max() <= 5000
    depth_image = depth_image * (255 / 5000)
    depth_image = depth_image.astype(np.uint8)
    cv2.imwrite(os.path.join(root,'step_%05d.png' % step), depth_image)

def set_horizon(env, horizon):
    now_horizon = env.last_event.metadata['agent']['cameraHorizon']
    if abs(horizon - now_horizon) > 1:
        env.look_angle(horizon - now_horizon)
    pass

def work(args, tasks_queue, lock):
    n_tasks = tasks_queue.qsize()
    
    env = thor_env.ThorEnv(x_display = args.x_display)
    while True:
        lock.acquire()
        n_tasks_remain = tasks_queue.qsize()
        if n_tasks_remain == 0:
            lock.release()
            break
        n_tasks_remain -= 1
        task_info = tasks_queue.get()
        lock.release()

        root = os.path.join('./data/replay_depth0', args.split, task_info['task'])
        rgb_root = os.path.join(root,'rgb')
        depth_root = os.path.join(root,'depth')

        task_start_time = time.time()
        # benchmark
        env.reset(task_info['scene']['floor_plan'])
        env.restore_scene(
            object_poses    = task_info['scene']['object_poses'], 
            object_toggles  = task_info['scene']['object_toggles'], 
            dirty_and_empty = task_info['scene']['dirty_and_empty']
            )
        env.step(dict(task_info['scene']['init_action']))
        # env.set_task(task_info, args, reward_type='dense')
        expert_actions = [a['discrete_action'] for a in task_info['plan']['low_actions'] ]
        task_index = task_info['task_index']   
        n_steps = len(expert_actions)

        if args.cont:
            rerun = False
        else:
            rerun = True

        
        cnt = 0
        if args.save_rgb:
            if (not args.cont) or (len(os.listdir(rgb_root)) != len(expert_actions) + 1):
                rerun = True
                if os.path.exists(rgb_root):
                    shutil.rmtree(rgb_root)
                os.makedirs(rgb_root)
                
        if args.save_depth:        
            if (not args.cont) or (len(os.listdir(depth_root)) != len(expert_actions) + 1):
                rerun = True
                if os.path.exists(depth_root):
                    shutil.rmtree(depth_root)
                os.makedirs(depth_root)
                
        if not rerun:
            print(f'{utils.timestr()} Rank {args.rank} {task_index} RemainTodo={n_tasks_remain}. Skip!')
            continue
        
        orig_traj_horizon = 30
        for step, action in enumerate(expert_actions):
            if 'mask' in action['args'] :
                mask  = action['args']['mask'] 
                mask = image_util.decompress_mask(mask)
            else:
                mask = None
            
            if 'LookUp' in action['action']:
                orig_traj_horizon -= 15
            elif 'LookDown' in action['action']:
                orig_traj_horizon += 15
            elif mask is None:
                set_horizon(env, FORCE_HORIZON)
            else:
                set_horizon(env, orig_traj_horizon)
            
            env.va_interact(
                action        = action['action'],
                interact_mask = mask, 
                smooth_nav    = args.smooth_nav,
                debug         = False
                )
            
            now_horizon = env.last_event.metadata['agent']['cameraHorizon']
            if abs(FORCE_HORIZON - now_horizon) < 1:
                if args.save_rgb:
                    save_rgb(rgb_root, cnt, env)
                if args.save_depth:
                    save_depth(depth_root, cnt, env)
                cnt += 1
        task_end_time = time.time()
        task_run_time = task_end_time - task_start_time    


        print(f'{utils.timestr()} Rank {args.rank} {task_index} RemainTodo={n_tasks_remain}. Done. RunTime {task_run_time:.2f}. Step {n_steps}. FPS {n_steps/task_run_time :.2f}.')


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    args.remove_repeat = True
    # args.save_rgb = True
    # args.save_depth = True
    print(f'******************** Replay Trajs ******************')
    print('args.cont       = ',args.cont)
    print('args.save_rgb   = ',args.save_rgb)
    print('args.save_depth = ',args.save_depth)

    mp.set_start_method('spawn')
    manager = mp.Manager()
    lock = manager.Lock()

    tasks = utils.load_alfred_tasks(args)
    tasks_queue = manager.Queue()
    for t in tasks:
        tasks_queue.put(t)
    
    print(f'******************** Launch {len(tasks)} Tasks *** {args.n_proc} Processes *** Device {args.gpu} ******************')
    if args.n_proc > 1:
        threads = []
        for rank in range(args.n_proc):
            args.rank = rank
            thread = mp.Process(
                target=work, 
                args= (args, tasks_queue, lock))
            thread.start()
            time.sleep(1)
            threads.append(thread)

        for thread in threads:
            thread.join()
    else:
        work(args, tasks_queue, lock)

if __name__ == '__main__':
    main()
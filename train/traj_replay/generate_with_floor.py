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

def save_rgb(root,step,env):
    cv2.imwrite(os.path.join(root,'step_%05d.png' % step), cv2.cvtColor(env.last_event.frame,cv2.COLOR_BGR2RGB))

def save_depth(root, step, env):
    depth_image = env.last_event.depth_frame
    assert depth_image.max() <= 5000
    depth_image = depth_image * (255 / 5000)
    depth_image = depth_image.astype(np.uint8)
    cv2.imwrite(os.path.join(root,'step_%05d.png' % step), depth_image)


def save_seg(root,step,env):
    id2type = {}
    for obj in env.last_event.metadata['objects']:
        id2type[obj['objectId']] = obj['objectType']
         
    out = {}
    out['mask']  = np.zeros((300,300),dtype=np.uint8)
    out['object'] = []
    cnt = 0
    for color, objectId in env.last_event.color_to_object_id.items():
        if (objectId in id2type and id2type[objectId] in ALFRED_INTEREST_OBJECTS) or 'Floor' in objectId:
            objectType = id2type.get(objectId, 'Floor')
            mask = np.all(env.last_event.instance_segmentation_frame == color, axis = -1)
            if np.any(mask):
                cnt += 1
                out['mask'][mask.astype(bool)] = cnt
                out['object'].append({
                    'objectId'   : objectId,
                    'objectType' : objectType,
                    'label'      : cnt
                })          
    
    
    
    # out = {}
    # out['mask']  = np.zeros((300,300),dtype=np.uint8)
    # out['object'] = []
    # cnt = 0
    # for objectId,mask in env.last_event.instance_masks.items():
    #     if (objectId in id2type and id2type[objectId] in ALFRED_INTEREST_OBJECTS) or 'Floor' in objectId:
    #         objectType = id2type.get(objectId, 'Floor')
    #         cnt += 1
    #         out['mask'][mask.astype(bool)] = cnt
    #         out['object'].append({
    #             'objectId'   : objectId,
    #             'objectType' : objectType,
    #             'label'      : cnt
    #         })
    
    with open(os.path.join(root,'step_%05d.pkl' % step), 'wb') as f:
        pkl.dump(out,f)

def work(args, tasks_queue, lock):
    n_tasks = tasks_queue.qsize()
    
    env = thor_env.ThorEnv(x_display = args.x_display)
    while True:
        lock.acquire()
        n_tasks_remain = tasks_queue.qsize()
        if n_tasks_remain == 0:
            lock.release()
            break
        task_info = tasks_queue.get()
        lock.release()


        root = os.path.join('./data/replay_floor', args.split, task_info['task'])
        rgb_root = os.path.join(root,'rgb')
        depth_root = os.path.join(root,'depth')
        seg_root = os.path.join(root,'seg')

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
        if args.save_rgb:
            if (not args.cont) or (len(os.listdir(rgb_root)) != len(expert_actions) + 1):
                rerun = True
                if os.path.exists(rgb_root):
                    shutil.rmtree(rgb_root)
                os.makedirs(rgb_root)
                save_rgb(rgb_root,0,env)

        if args.save_depth:        
            if (not args.cont) or (len(os.listdir(depth_root)) != len(expert_actions) + 1):
                rerun = True
                if os.path.exists(depth_root):
                    shutil.rmtree(depth_root)
                os.makedirs(depth_root)
                save_depth(depth_root,0,env)
        
        if args.save_seg:        
            if (not args.cont) or (len(os.listdir(seg_root)) != len(expert_actions) + 1):
                rerun = True
                if os.path.exists(seg_root):
                    shutil.rmtree(seg_root)
                os.makedirs(seg_root)
                save_seg(seg_root,0,env)
        
        if not rerun:
            print(f'{utils.timestr()} Rank {args.rank} {task_index} ({n_tasks - n_tasks_remain + 1}/{n_tasks}). Skip!')
            continue
        for step, action in enumerate(expert_actions):
            if 'mask' in action['args'] :
                mask  = action['args']['mask'] 
                mask = image_util.decompress_mask(mask)
            else:
                mask = None
            env.va_interact(
                action        = action['action'],
                interact_mask = mask, 
                smooth_nav    = args.smooth_nav,
                debug         = False
                )
            
            if args.save_rgb:
                save_rgb(rgb_root,step + 1,env)
            
            if args.save_depth:
                save_depth(depth_root,step + 1,env)
            
            if args.save_seg:
                save_seg(seg_root,step + 1,env)

            if args.step_report:
                print('step %d action %s' % (step,action['action']))

        task_end_time = time.time()
        task_run_time = task_end_time - task_start_time    


        print(f'{utils.timestr()} Rank {args.rank} {task_index} ({n_tasks - n_tasks_remain + 1}/{n_tasks}). Done. RunTime {task_run_time:.2f}. Step {n_steps}. FPS {n_steps/task_run_time :.2f}.')


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    args.remove_repeat = True
    print(f'******************** Replay Trajs ******************')
    print('args.cont       = ',args.cont)
    print('args.save_rgb   = ',args.save_rgb)
    print('args.save_depth = ',args.save_depth)
    print('args.save_seg   = ',args.save_seg)

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
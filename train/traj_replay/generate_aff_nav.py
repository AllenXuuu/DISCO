from utils import arguments, utils
import os
import torch.multiprocessing as mp
import importlib
from env import thor_env
import time
from utils import image_util
import cv2,torch
import pickle as pkl
import shutil
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import numpy as np
from gen import constants

def save_rgb(root,step,env):
    cv2.imwrite(os.path.join(root,'step_%05d.png' % step), cv2.cvtColor(env.last_event.frame,cv2.COLOR_BGR2RGB))

def process(x,z,r,h):
    x = np.round(x * 4) / 4
    z = np.round(z * 4) / 4
    r = np.round(r / 90).astype(int) * 90
    while r >= 360:
        r -= 360
    while r < 0:
        r += 360
    h = np.round(h / 15).astype(int) * 15
    return x,z,r,h

def np_op(depth,x,z,r,h, nav_grids):
    height,width = depth.shape
    fov = 60
    
    ptz = depth
    ptx, pty = np.meshgrid(np.arange(width), np.arange(height-1,-1,-1))
    ptx = (ptx - width/2 + 0.5) / (width / 2 - 0.5)
    pty = (pty - height/2 + 0.5) / (height / 2 - 0.5)
    ptx = ptz * np.tan(np.deg2rad(fov / 2)) * ptx
    pty = ptz * np.tan(np.deg2rad(fov / 2)) * pty
    coords = np.stack([ptx,pty,ptz],-1)
    
    sin_h = np.sin(np.deg2rad(h))
    cos_h = np.cos(np.deg2rad(h))
    coords = coords @ np.array([
        [1,     0,      0],
        [0,     cos_h,  sin_h],
        [0,     -sin_h, cos_h],    
    ])
    sin_r = np.sin(np.deg2rad(r))
    cos_r = np.cos(np.deg2rad(r))
    coords = coords @ np.array([
        [cos_r, 0,      -sin_r],
        [0,     1,      0],
        [sin_r, 0,      cos_r],    
    ])
    coords = coords / 1000
    
    coords[... , 0] += x
    coords[... , 2] += z
    
    coords = coords[...,[0,2]]
    coords = np.round(coords * 4) / 4
    coords = np.tile(np.expand_dims(coords, 2), (1,1,nav_grids.shape[0],1))
    nav_mask = (coords == nav_grids).all(-1).any(-1)
    
    return nav_mask




def pt_op(depth,x,z,r,h,device,nav_grids):
    height,width = depth.shape
    fov = 60
    
    # torch_type = torch.float32
    torch_type = torch.float64
    
    ptz = torch.from_numpy(depth).type(torch_type).to(device)
    ptx, pty = torch.meshgrid(torch.arange(width,device=device), torch.arange(height-1,-1,-1,device=device))
    ptx, pty = ptx.T, pty.T
    ptx = (ptx - width/2 + 0.5) / (width / 2 - 0.5)
    pty = (pty - height/2 + 0.5) / (height / 2 - 0.5)
    ptx = ptz * np.tan(np.deg2rad(fov / 2)) * ptx
    pty = ptz * np.tan(np.deg2rad(fov / 2)) * pty
    coords = torch.stack([ptx,pty,ptz],-1)
    
    sin_h = np.sin(np.deg2rad(h))
    cos_h = np.cos(np.deg2rad(h))
    coords = coords @ torch.tensor([
        [1,     0,      0],
        [0,     cos_h,  sin_h],
        [0,     -sin_h, cos_h],    
    ], device=device, dtype=torch_type)
    sin_r = np.sin(np.deg2rad(r))
    cos_r = np.cos(np.deg2rad(r))
    coords = coords @ torch.tensor([
        [cos_r, 0,      -sin_r],
        [0,     1,      0],
        [sin_r, 0,      cos_r],   
    ], device=device, dtype=torch_type)
    coords = coords / 1000
    
    coords[... , 0] += x
    coords[... , 2] += z
    
    coords = coords[...,[0,2]]
    coords = torch.round(coords * 4) / 4
    nav_mask = 1
    
    coords = coords.unsqueeze(2).tile(1,1,nav_grids.shape[0],1)
    nav_mask = (coords == nav_grids).all(-1).any(-1)
    nav_mask = nav_mask.cpu().numpy()
    return nav_mask


def work(args, tasks_queue, lock):
    use_torch_cuda = len(args.gpu) > 0
    device = torch.device(f'cuda:{args.gpu[0]}' if use_torch_cuda else 'cpu')
    while True:
        lock.acquire()
        n_tasks_remain = tasks_queue.qsize()
        if n_tasks_remain == 0:
            lock.release()
            break
        task_info = tasks_queue.get()
        n_tasks_remain -= 1
        lock.release()
        
        task_start_time = time.time()
        task_index = task_info['task_index']   
        
        root = os.path.join('./data/replay', args.split, task_info['task'])
        scene_id = task_info['scene']['floor_plan']
        nav_grids = pkl.load(open(f'./data/scene_parse/nav/{scene_id}.pkl','rb'))
        nav_grids = np.array(nav_grids)
        if use_torch_cuda:
            nav_grids = torch.from_numpy(nav_grids).to(device)
        n_nav = nav_grids.shape[0]
        
        expert_actions = [a['discrete_action']['action'] for a in task_info['plan']['low_actions'] ]
        init_action = task_info['scene']['init_action']
        n_steps = len(expert_actions) + 1
        
        save_folder = os.path.join(root,'navigation')
        os.makedirs(save_folder,exist_ok=True)
        
        x,z,r,h = init_action['x'],init_action['z'],init_action['rotation'],init_action['horizon']
        x,z,r,h = process(x,z,r,h)
        # print(expert_actions)
        for step in range(n_steps):
            depth_path = os.path.join(root,'depth',f'step_{step:05d}.png')
            depth = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE).astype(float) / 255 * 5000
            save_path = os.path.join(save_folder,f'step_{step:05d}.png')
            
            if not use_torch_cuda:
                nav_mask = np_op(depth,x,z,r,h,nav_grids)        
            else:          
                nav_mask = pt_op(depth,x,z,r,h,device,nav_grids)
            
            nav_mask = nav_mask.astype(np.uint8) * 255            
            cv2.imwrite(save_path,nav_mask)
            
            if step == len(expert_actions):
                break
            else:
                action = expert_actions[step]
                if 'LookDown' in action:
                    h += 15
                elif 'LookUp' in action:
                    h -= 15
                elif 'RotateLeft' in action:
                    r = (r - 90) % 360
                elif 'RotateRight' in action:
                    r = (r + 90) % 360
                elif 'MoveAhead' in action:
                    if r == 0:
                        z += 0.25
                    elif r == 90:
                        x += 0.25
                    elif r == 180:
                        z -= 0.25
                    elif r == 270:
                        x -= 0.25
                    else:
                        raise ValueError(r)
                else:
                    assert 'Object' in action
                # print(action,x,z,r,h)
                if not use_torch_cuda:
                    assert (nav_grids == [x,z]).all(1).any(0)

        task_end_time = time.time()
        task_run_time = task_end_time - task_start_time    


        print(f'{utils.timestr()} Rank {args.rank} {task_index}. RemainTODO {n_tasks_remain}. Done. RunTime {task_run_time:.2f}. Step {n_steps}. FPS {n_steps/task_run_time :.2f}.')


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    args.remove_repeat = True

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
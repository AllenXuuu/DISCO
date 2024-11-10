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
import torch

USE_TORCH_CUDA = True
ROOT = '/home/allenxyxu/vlnidata/replay'
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
    for objectId,mask in env.last_event.instance_masks.items():
        if objectId in id2type and id2type[objectId] in ALFRED_INTEREST_OBJECTS:
            objectType = id2type[objectId]
            cnt += 1
            out['mask'][mask.astype(bool)] = cnt
            out['object'].append({
                'objectId'   : objectId,
                'objectType' : objectType,
                'label'      : cnt
            })
    with open(os.path.join(root,'step_%05d.pkl' % step), 'wb') as f:
        pkl.dump(out,f)


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

def pt_op(depth,x,z,r,h,nav_grids,device):
    height,width = depth.shape
    fov = 60
    
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
    
    bz = 100
    nav_mask = torch.zeros((300,300),dtype = torch.bool,device=device)
    for i in range(0, nav_grids.shape[0], bz):
        batch_nav_grids = nav_grids[i: i+bz]
        batch_nav_mask = (coords.unsqueeze(2).tile(1,1,batch_nav_grids.shape[0],1) == batch_nav_grids).all(-1).any(-1)
        nav_mask = torch.logical_or(nav_mask,batch_nav_mask)
    nav_mask = nav_mask.cpu().numpy()
    return nav_mask

def save_aff(root,step,env,nav_grids,x,z,r,h, device ,use_torch_cuda = USE_TORCH_CUDA):
    depth = env.last_event.depth_frame
    if not use_torch_cuda:
        nav_mask = np_op(depth,x,z,r,h,nav_grids)        
    else:         
        nav_mask = pt_op(depth,x,z,r,h,torch.from_numpy(nav_grids).to(device),device)
    out = {
        'Navigable' : nav_mask,
        'PickupObject'    : [],
        'PutObject'       : [],
        'OpenObject'      : [],
        'CloseObject'     : [],
        'ToggleObjectOn'  : [],
        'ToggleObjectOff' : [],
        'SliceObject'     : [],
        'CleanObject'     : [],
        'HeatObject'     : [],
        'CoolObject'     : [],
    }
    for obj in env.last_event.metadata['objects']:
        if obj['objectType'] in ALFRED_INTEREST_OBJECTS and obj['objectId'] in env.last_event.instance_masks and (not obj['isPickedUp']):
            if obj['pickupable']:
                out['PickupObject'].append(obj['objectId'])

            if obj['receptacle']:
                out['PutObject'].append(obj['objectId'])
            
            if obj['openable'] and (not obj['isOpen']) and obj['objectType'] in constants.OPENABLE_CLASS_SET:
                out['OpenObject'].append(obj['objectId'])

            if obj['openable'] and obj['isOpen'] and obj['objectType'] in constants.OPENABLE_CLASS_SET:
                out['CloseObject'].append(obj['objectId'])

            if obj['toggleable'] and (not obj['isToggled']) and obj['objectType'] in {'FloorLamp','DeskLamp','Faucet','Microwave'}:
                out['ToggleObjectOn'].append(obj['objectId'])
                
            if obj['toggleable'] and (obj['isToggled'])     and obj['objectType'] in {'FloorLamp','DeskLamp','Faucet','Microwave'}:
                out['ToggleObjectOff'].append(obj['objectId'])
            
            if obj['sliceable'] and (not obj['isSliced']) and obj['objectType'] in {'Apple','Bread','Lettuce','Potato','Tomato'}:
                out['SliceObject'].append(obj['objectId'])

            if obj['objectType'] == 'SinkBasin':
                out['CleanObject'].append(obj['objectId'])

            if obj['objectType'] == 'Microwave':
                out['HeatObject'].append(obj['objectId'])

            if obj['objectType'] == 'Fridge':
                out['CoolObject'].append(obj['objectId'])

    with open(os.path.join(root,'step_%05d_aff.pkl' % step), 'wb') as f:
        pkl.dump(out,f)
    




def work(args, tasks_queue, lock):
    n_tasks = tasks_queue.qsize()
    device = args.gpu[args.rank]
    device = torch.device('cuda:%d' % device)
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

        root = os.path.join(ROOT, args.split, task_info['task'])
        rgb_root = os.path.join(root,'rgb')
        depth_root = os.path.join(root,'depth')
        seg_root = os.path.join(root,'seg')
        aff_root = os.path.join(root,'aff')

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


        scene_id = task_info['scene']['floor_plan']
        nav_grids = pkl.load(open(f'./data/scene_parse/nav/{scene_id}.pkl','rb'))
        nav_grids = np.array(nav_grids)

        init_action = task_info['scene']['init_action']
        x,z,r,h = init_action['x'],init_action['z'],init_action['rotation'],init_action['horizon']
        x,z,r,h = process(x,z,r,h)
        
        if args.save_rgb:
            if os.path.exists(rgb_root):
                shutil.rmtree(rgb_root)
            os.makedirs(rgb_root)
            save_rgb(rgb_root,0,env)

        if args.save_depth:        
            if os.path.exists(depth_root):
                shutil.rmtree(depth_root)
            os.makedirs(depth_root)
            save_depth(depth_root,0,env)
        
        if args.save_seg:        
            if os.path.exists(seg_root):
                shutil.rmtree(seg_root)
            os.makedirs(seg_root)
            save_seg(seg_root,0,env)
        
        if args.save_aff:
            if os.path.exists(aff_root):
                shutil.rmtree(aff_root)
            os.makedirs(aff_root)
            save_aff(aff_root,0,env,nav_grids,x,z,r,h,device = device)
        
        
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
            
            if 'LookDown' in action['action']:
                h += 15
            elif 'LookUp' in action['action']:
                h -= 15
            elif 'RotateLeft' in action['action']:
                r = (r - 90) % 360
            elif 'RotateRight' in action['action']:
                r = (r + 90) % 360
            elif 'MoveAhead' in action['action']:
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
                assert 'Object' in action['action']
            
            if args.save_rgb:
                save_rgb(rgb_root,step + 1,env)
            
            if args.save_depth:
                save_depth(depth_root,step + 1,env)
            
            if args.save_seg:
                save_seg(seg_root,step + 1,env)
            
            if args.save_aff:
                save_aff(aff_root,step + 1,env,nav_grids,x,z,r,h,device = device)

            if args.step_report:
                print('step %d action %s' % (step,action['action']))

        task_end_time = time.time()
        task_run_time = task_end_time - task_start_time    

        print(f'{utils.timestr()} Rank {args.rank} {task_index} RemainTodo={n_tasks_remain}. Done. RunTime {task_run_time:.2f}. Step {n_steps}. FPS {n_steps/task_run_time :.2f}.')


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    args.remove_repeat = True
    print(f'******************** Replay Trajs ******************')
    print('args.cont       = ',args.cont)
    print('args.save_rgb   = ',args.save_rgb)
    print('args.save_depth = ',args.save_depth)
    print('args.save_seg   = ',args.save_seg)
    print('args.save_aff   = ',args.save_aff)

    mp.set_start_method('spawn')
    manager = mp.Manager()
    lock = manager.Lock()

    tasks = utils.load_alfred_tasks(args)
    tasks_queue = manager.Queue()
    for t in tasks:
        tasks_queue.put(t)
    args.gpu = args.gpu * (args.n_proc // len(args.gpu) + 1)
    args.gpu = args.gpu[:args.n_proc]
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

from utils import arguments, utils
import os
import torch.multiprocessing as mp
import importlib
from env import thor_env
import time
from agent.disco import Agent

def work(args, tasks_queue, lock):
    n_tasks = tasks_queue.qsize()
    env = thor_env.ThorEnv(x_display = args.x_display)
    agent = Agent(args,env)

    while True:
        lock.acquire()
        n_tasks_remain = tasks_queue.qsize()
        if n_tasks_remain == 0:
            lock.release()
            break
        task_info = tasks_queue.get()
        lock.release()

        task_start_time = time.time()
        result = agent.launch(task_info)
        task_end_time = time.time()
        task_run_time = task_end_time - task_start_time    

        task_index = task_info['task_index']
        goal_condition = '%d / %d' % (result['completed_goal_conditions'], result['total_goal_conditions'])
        print(f'{utils.timestr()} Rank {args.rank} {task_index} ({n_tasks - n_tasks_remain + 1}/{n_tasks}). RunTime {task_run_time:.2f}. Step {agent.steps}. FPS {agent.steps/task_run_time :.2f}. GC {goal_condition}')
    env.stop()

def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir, exist_ok=True)
    args.logging = logging_dir
    
    mp.set_start_method('spawn')
    manager = mp.Manager()
    lock = manager.Lock()

    tasks = utils.load_alfred_tasks(args)

    tasks_queue = manager.Queue()
    for t in tasks:
        tasks_queue.put(t)
    
    
    args.gpu = args.gpu * (args.n_proc // len(args.gpu) + 1)
    args.gpu = args.gpu[:args.n_proc]

    print(f'******************** Launch {len(tasks)} Tasks *** {args.n_proc} Processes *** Device {args.gpu}')
    if args.n_proc > 1:
        threads = []
        for rank in range(args.n_proc):
            args.rank = rank
            thread = mp.Process(
                target=work, 
                args= (args, tasks_queue, lock))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    else:
        work(args, tasks_queue, lock)

if __name__ == '__main__':
    main()
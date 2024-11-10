from utils import utils
import os
from termcolor import colored
from collections import OrderedDict
import json
import sys
import numpy as np
import torch
from utils import image_util
import shutil
import traceback

class Agent:
    def __init__(self, args, env):
        self.env = env
        self.args = args
        self.task_info = {}
        self.task_log_dir  = ''
        self.device = torch.device('cuda:%d' % args.gpu[args.rank])
        self.log_file_name = '_log.txt'

    def log(self,report=''):
        if type(report)!=str:
            report = str(report)
        if self.task_log_dir:
            with open(os.path.join(self.task_log_dir, self.log_file_name),'a+') as f:
                f.write(report)
                f.write('\n')
        
        if self.args.step_report:
            print(report)

    def reset_task(self):
        seed = self.args.seed + int(self.task_info['task_index'][-5:])
        utils.set_random_seed(seed)
        # benchmark
        self.env.reset(self.task_info['scene']['floor_plan'])
        self.env.restore_scene(
            object_poses    = self.task_info['scene']['object_poses'], 
            object_toggles  = self.task_info['scene']['object_toggles'], 
            dirty_and_empty = self.task_info['scene']['dirty_and_empty']
            )
        self.env.step(dict(self.task_info['scene']['init_action']))
        self.env.set_task(self.task_info, self.args, reward_type='dense')
        
        ### statistics
        self.steps = 0
        self.fails = 0

        ### for leaderboard
        self.api_actions = []

    def launch(self,task_info):
        self.task_info = task_info
        self.task_log_dir = os.path.join(self.args.logging, self.task_info['task_index'])
        if os.path.exists(os.path.join(self.task_log_dir)):
            shutil.rmtree(self.task_log_dir)
        os.makedirs(self.task_log_dir)
        self.log(f"scene {self.task_info['scene']['floor_plan']}" )
        self.log(f"Task {self.task_info['task']}")
        self.log(f"Repeat Index {self.task_info['repeat_idx']}")  
        self.log('=========================')
        self.log('Language')
        self.log('')
        self.log(self.task_info['turk_annotations']['anns'][self.task_info['repeat_idx']]['task_desc'])
        self.log('')
        self.log('\n'.join(self.task_info['turk_annotations']['anns'][self.task_info['repeat_idx']]['high_descs']))
        self.log('=========================')
        # if os.path.exists(os.path.join(self.task_log_dir, self.log_file_name)):
        #     os.remove(os.path.join(self.task_log_dir, self.log_file_name))
        self.reset_task()
        
        try:
            self.run()
        
        except KeyboardInterrupt:
            print('!!!!!!!!!!!! Ctrl+C pressed. Exit !!!!!!!!!')
            exit(0)
        except Exception as e:
            pass
            # traceback.print_exc()
            # task_index = self.task_info['task_index']
            # print(f"============== Error!!! task_index={task_index} ============")
            # self.log(repr(e))
        
        result = self.evaluate_alfred_task()
        with open(os.path.join(self.task_log_dir,'_result.json'),'w') as f:
            json.dump(result,f,indent=4)
        with open(os.path.join(self.task_log_dir,'_api_actions.json'),'w') as f:
            json.dump(self.api_actions,f,indent=4)
    
        
        
        
        return result

    def is_terminate(self):
        if self.steps >= self.args.max_steps:
            return True
        if self.fails >= self.args.max_fails:
            return True
        return False
    
    def va_interact(self, action, interact_mask=None, smooth_nav=None):
        if smooth_nav is None:
            smooth_nav = self.args.smooth_nav
        self.steps += 1
        success, _, target_instance_id, error_message, api_action = \
            self.env.va_interact(action,interact_mask,smooth_nav=smooth_nav,debug=False)
        
        if self.steps <= self.args.max_steps and self.fails < self.args.max_fails:
            self.api_actions.append(api_action)
        
        if not success:
            self.fails +=1

        
        # if target_instance_id:
        #     visible = None
        #     for target_object in self.last_event.metadata['objects']:
        #         if target_object['objectId'] == target_instance_id:
        #             visible = target_object['visib le']
        #             break
        #     assert visible is not None
        #     visible = 'Visible' if visible else 'NotVisible' 
        #     self.log('(x,z,r,h) = (%.2f,%.2f,%.2f,%.2f)' % \
        #         (self.last_event.metadata['agent']['position']['x'],self.last_event.metadata['agent']['position']['z'],self.last_event.metadata['agent']['rotation']['y'],self.last_event.metadata['agent']['cameraHorizon']))    
        # action_full = action + '_' + target_instance_id + '_' + visible if target_instance_id else action
        
        
        action = action + '_' + target_instance_id if target_instance_id else action
        if hasattr(self,'pose'):
            pose = f'Pose {self.pose}. ' # type: ignore
        else:
            pose = ''
        step_report = f'Step [{self.steps:d}]. {pose}Action [{action}]. Success [{success}].'
        if not success:
            step_report += f' Error [{error_message}].'

        self.log(step_report)

        return success, _, target_instance_id, error_message, api_action
    

    def evaluate_alfred_task(self):
        goal_satisfied = self.env.get_goal_satisfied()

        # goal_conditions
        pcs = self.env.get_goal_conditions_met()

        # SPL
        path_len_expert = len(self.task_info['plan']['low_actions'])
        path_len_agent  = int(self.steps)
        
        result = OrderedDict()
        # result = dict()
        result['success'] = 1 if goal_satisfied else 0
        result['completed_goal_conditions'] = int(pcs[0])
        result['total_goal_conditions'] = int(pcs[1])
        result['path_len_expert'] = path_len_expert
        result['path_len_agent']  = path_len_agent
        result['fails'] = self.fails
        
        return result
    


    def run(self):
        raise NotImplementedError
    

from utils.arguments import parse_args
from utils import utils
import os
import json

if __name__ == '__main__':
    args = parse_args()
    logging_dir = os.path.join('logs', args.name)
    tasks = utils.load_alfred_tasks(args)
    
    results = []
    for t in tasks:
        result = os.path.join(logging_dir, t['task_index'], '_result.json')
        if os.path.exists(result):

            result = json.load(open(result))
        else:
            result = None
        results.append(result)
    
    utils.get_alfred_metrics(tasks, results)
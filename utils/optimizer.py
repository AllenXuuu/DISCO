import torch.optim as optim
import math

def build_optimizer(args, params):

    params
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.base_lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=args.base_lr,
                               betas=(args.beta1,args.beta2), 
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=args.base_lr,
                                betas=(args.beta1,args.beta2), 
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'no':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1
        )
    elif args.lr_scheduler == 'warmup':
        WarmLambda = lambda x: (0.99 * x / args.warmup_step + 0.01) if x < args.warmup_step else  1
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,lr_lambda=WarmLambda
        )
    else:
        raise NotImplementedError

    return scheduler
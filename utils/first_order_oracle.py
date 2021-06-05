import torch

#stochastic first order oracle defining optimizer
from optimizer import *

# stochastic first order oracle

def SFO(model, args):
    if args.optimizer == 'SGD':
        optimizer = CustomSGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.optimizer == 'SGD+M': 

        # optimizer = torch.optim.SGD(model.parameters(), args.lr,
        #                         momentum=args.momentum,
        #                         weight_decay=args.weight_decay)
                                   
        optimizer = CustomSGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
    elif args.optimizer == 'Adam':
        optimizer = CustomAdam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    return optimizer
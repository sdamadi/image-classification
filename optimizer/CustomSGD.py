# the step method of the optimizer has changed to get
# masl as the input to freeze small elements

import torch

class CustomSGD(torch.optim.SGD):
  @torch.no_grad()
  def step(self, mask, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    idx = 0
    for group in self.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad

            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
            
            if mask != None: 
                layer_mask = mask[idx]
                d_p.mul_(layer_mask)

            p.add_(d_p, alpha=-group['lr'])
            
            idx += 1

    return loss
import torch
import torch.nn.functional as F

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    num_items = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / num_items))
    return res


def focal_loss(prob, targets, alpha: float = 0.75, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


class WarmupStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, warmup_steps, step_size, gamma=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupStepLR, self).__init__(optimizer, step_size, gamma=self.gamma, last_epoch=last_epoch)
    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        # e.g. warmup_steps = 10, case: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21...
        if self.last_epoch == self.warmup_steps or(self.last_epoch % self.step_size != 0 and self.last_epoch > self.warmup_steps):
            return [group['lr'] for group in self.optimizer.param_groups]
        # e.g. warmup_steps = 10, case: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        elif self.last_epoch < self.warmup_steps:
            return [group['initial_lr'] * float(self.last_epoch + 1) / float(self.warmup_steps) for group in self.optimizer.param_groups]
        
        
        # e.g. warmup_steps = 10, case: 10, 20, 30, 40...
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
    def _get_closed_form_lr(self):
        if self.last_epoch <= self.warmup_steps:
            return [base_lr * float(self.last_epoch) / (self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** ((self.last_epoch -  self.warmup_steps)// self.step_size) for base_lr in self.base_lrs]


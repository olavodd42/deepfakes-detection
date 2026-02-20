from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class CosineAnnealingWithWarmup:
    def __init__(
        self,
        optimizer,
        start_factor=1/3,
        end_factor=1,
        num_epochs=10,
        warmup_epochs=5
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs

        self.scheduler_warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=warmup_epochs
        )

        self.scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=(num_epochs - warmup_epochs),
            eta_min=1e-6
        )

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[self.scheduler_warmup, self.scheduler_cosine],
            milestones=[warmup_epochs]
        )

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
class BaseAttackMethod:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, trainer):
        return self.change_functionality(trainer)

    def change_functionality(self, trainer):
        return trainer

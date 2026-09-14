from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DocSegLossEpochHook(Hook):
    """Pass the current epoch to DocSegCombinedLoss modules."""

    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        for module in model.modules():
            if hasattr(module, 'set_epoch'):
                module.set_epoch(runner.epoch)

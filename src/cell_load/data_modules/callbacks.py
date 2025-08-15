from lightning import Callback


class ReshuffleSentencesEachEpoch(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dl = trainer.train_dataloader
        loaders = dl.loaders if hasattr(dl, "loaders") else [dl]
        for l in loaders:
            bs = getattr(l, "batch_sampler", None)
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(trainer.current_epoch)
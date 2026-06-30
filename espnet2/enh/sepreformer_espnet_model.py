from typing import Optional

from espnet2.enh.espnet_model import ESPnetEnhancementModel


class SepReformerESPnetEnhancementModel(ESPnetEnhancementModel):
    """Enhancement model that propagates epoch information to loss wrappers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = None

    def set_epoch(self, epoch: Optional[int]) -> None:
        self.current_epoch = None if epoch is None else int(epoch)
        if self.loss_wrappers is None:
            return
        for loss_wrapper in self.loss_wrappers:
            if hasattr(loss_wrapper, "set_epoch"):
                loss_wrapper.set_epoch(epoch)

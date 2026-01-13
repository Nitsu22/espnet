from espnet2.iterators.sequence_iter_factory_log import SequenceIterFactoryLog
from espnet2.train.dataset_log import ESPnetDatasetLog


def apply_log_patches() -> None:
    import espnet2.tasks.abs_task as abs_task
    import espnet2.train.dataset as dataset_mod
    import espnet2.train.iterable_dataset as iterable_dataset_mod

    abs_task.SequenceIterFactory = SequenceIterFactoryLog
    abs_task.ESPnetDataset = ESPnetDatasetLog

    dataset_mod.ESPnetDataset = ESPnetDatasetLog
    iterable_dataset_mod.ESPnetDataset = ESPnetDatasetLog

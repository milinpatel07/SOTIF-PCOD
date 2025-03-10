import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .kitti.kitti_tracking_dataset import KittiTrackingDataset
from .kitti.kitti_dataset_var import KittiDatasetVAR
from .kitti.kitti_tracking_dataset_var import KittiTrackingDatasetVAR
from .kitti.kitti_dataset_mimo_var import KittiDatasetMIMOVAR
from .kitti.kitti_dataset_mimo_var_a import KittiDatasetMIMOVARA
from .kitti.kitti_dataset_mimo_var_b import KittiDatasetMIMOVARB
from .kitti.kitti_dataset_mimo_var_c import KittiDatasetMIMOVARC
from .kitti.kitti_tracking_dataset_mimo_var2 import KittiTrackingDatasetMIMOVAR2
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .nuscenes.nuscenes_dataset_var import NuScenesDatasetVAR
from .nuscenes.nuscenes_dataset_mimo_var_a import NuScenesDatasetMIMOVARA
from .nuscenes.nuscenes_dataset_mimo_var_b import NuScenesDatasetMIMOVARB
from .nuscenes.nuscenes_dataset_mimo_var_c import NuScenesDatasetMIMOVARC
from .waymo.waymo_dataset import WaymoDataset
from .cadc.cadc_dataset import CadcDataset
from .cadc.cadc_dataset_var import CadcDatasetVAR
from .cadc.cadc_dataset_mimo_var_a import CadcDatasetMIMOVARA
from .cadc.cadc_dataset_mimo_var_b import CadcDatasetMIMOVARB
from .cadc.cadc_dataset_mimo_var_c import CadcDatasetMIMOVARC

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'KittiTrackingDataset': KittiTrackingDataset,
    'KittiDatasetVAR': KittiDatasetVAR,
    'KittiTrackingDatasetVAR': KittiTrackingDatasetVAR,
    'KittiDatasetMIMOVAR': KittiDatasetMIMOVAR, # This version is deprecated
    'KittiDatasetMIMOVARA': KittiDatasetMIMOVARA,
    'KittiDatasetMIMOVARB': KittiDatasetMIMOVARB,
    'KittiDatasetMIMOVARC': KittiDatasetMIMOVARC,
    'KittiTrackingDatasetMIMOVAR2': KittiTrackingDatasetMIMOVAR2,
    'NuScenesDataset': NuScenesDataset,
    'NuScenesDatasetVAR': NuScenesDatasetVAR,
    'NuScenesDatasetMIMOVARA': NuScenesDatasetMIMOVARA,
    'NuScenesDatasetMIMOVARB': NuScenesDatasetMIMOVARB,
    'NuScenesDatasetMIMOVARC': NuScenesDatasetMIMOVARC,
    'WaymoDataset': WaymoDataset,
    'CadcDataset': CadcDataset,
    'CadcDatasetVAR': CadcDatasetVAR,
    'CadcDatasetMIMOVARA': CadcDatasetMIMOVARA,
    'CadcDatasetMIMOVARB': CadcDatasetMIMOVARB,
    'CadcDatasetMIMOVARC': CadcDatasetMIMOVARC
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        if not hasattr(dataset, 'merge_all_iters_to_one_epoch'):
            raise AssertionError
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    # MIMO needs:
    # 1. Shuffle off and persistent_workers to True so we can detect when an epoch is complete
    # 2. Drop last set to true so that it doesn't fail at the end of a batch
    MIMO_MODE = dataset_cfg.DATASET == 'KittiDatasetMIMOVAR'

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training and (MIMO_MODE is False), collate_fn=dataset.collate_batch,
        drop_last=False or MIMO_MODE, sampler=sampler, timeout=0, persistent_workers=False or MIMO_MODE
    )

    return dataset, dataloader, sampler

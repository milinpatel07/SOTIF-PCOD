import copy, time
import numpy as np
from collections import defaultdict

from ...utils import box_utils, common_utils, mimo_utils
from .cadc_dataset_var import CadcDatasetVAR
from ..dataset import DatasetTemplate
from ..processor.data_processor import DataProcessor

class CadcDatasetMIMOVARB(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.NUM_HEADS = dataset_cfg.NUM_HEADS
        self.MAX_POINTS_PER_VOXEL = dataset_cfg.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL']
        self.INPUT_REPETITION = dataset_cfg.INPUT_REPETITION
        self.BATCH_REPETITION = dataset_cfg.BATCH_REPETITION

        self.rng = np.random.default_rng()

        self.cadc_dataset = CadcDatasetVAR(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )

        # Create two different dataset proccesors
        # First one has performs masking and second will shuffle pts and voxelize
        data_processor_cfg_masking = \
            [x for x in self.dataset_cfg.DATA_PROCESSOR if x.NAME == 'mask_points_and_boxes_outside_range']
        data_processor_cfg_shffl_voxelize = \
            [x for x in self.dataset_cfg.DATA_PROCESSOR if x.NAME != 'mask_points_and_boxes_outside_range']
        self.data_processor_masking = DataProcessor(
            data_processor_cfg_masking, point_cloud_range=self.point_cloud_range, training=self.training
        )
        self.data_processor_shffl_voxelize = DataProcessor(
            data_processor_cfg_shffl_voxelize, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.time_list = []
        self.frame_count = 0

    def __len__(self):
        return len(self.cadc_dataset)

    def __getitem__(self, index):
        self.start_time = time.time()
        info = copy.deepcopy(self.cadc_dataset.cadc_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.cadc_dataset.get_lidar(sample_idx)
        calib = self.cadc_dataset.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.cadc_dataset.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'sample_idx': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']

            # Create mask to filter annotations during training
            if self.training and self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (annos['num_points_in_gt'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            gt_names = annos['name'] if mask is None else annos['name'][mask]
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar'] if mask is None else annos['gt_boxes_lidar'][mask]
            else:
                # This should not run, although the code should look somewhat like this
                raise NotImplementedError

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        # Custom prepare data function
        # Which masks points but only performs shuffle & voxelize during testing
        data_dict = mimo_utils.prepare_data_b(self, data_dict=input_dict, head_dataset=self.cadc_dataset)

        data_dict['image_shape'] = img_shape
        return data_dict

    # This collate_batch function is modified from the one in dataset.py
    # Instead of receiving a list of data_dicts (N),
    # it receives a list of lists of data_dicts (N, number of heads)
    def collate_batch(self, batch_list, _unused=False):
        batch_repetitions = 1
        if self.training:
            batch_repetitions = self.BATCH_REPETITION

        if self.training:
            data_dict = mimo_utils.modify_batch_b(self, batch_list)
            # N * number of heads * batch repetition
            batch_size = len(batch_list) * self.NUM_HEADS * batch_repetitions
        else:
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    # Voxel stuff is added once
                    # Other stuff must be added for each head
                    if key in ['voxels', 'voxel_coords', 'voxel_num_points']:
                        data_dict[key].append(val)
                    else:
                        for head_id in range(self.NUM_HEADS):
                            data_dict[key].append(val)
            # N * number of heads * batch repetition
            batch_size = len(batch_list) * self.NUM_HEADS * batch_repetitions

        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        # Data processing timing only during testing
        if not self.training:
            self.frame_count += 1
            t1 = time.time()
            total_time = t1 - self.start_time
            self.time_list.append(total_time)
            if self.frame_count == self.__len__():
                print('Mean data processing time', np.mean(self.time_list))
                self.logger.info('Mean data processing time: %.4f s' % np.mean(self.time_list))

        return ret

    # We can use the method from one of our heads
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        FRAME_NUM = 0 # Must have eval set to batch size of 1
        ret_dict = self.cadc_dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path)

        # Generate prediction dicts for each head
        ret_dict_list = []
        for i in range(self.NUM_HEADS):
            ret_dict_list.append(self.cadc_dataset.generate_prediction_dicts( \
                batch_dict, pred_dicts[FRAME_NUM]['pred_dicts_list'][i], class_names, output_path)[FRAME_NUM])
        ret_dict[FRAME_NUM]['post_nms_head_outputs'] = ret_dict_list

        return ret_dict

    # Must also use this method from one of our heads
    def evaluation(self, det_annos, class_names, **kwargs):
        return self.cadc_dataset.evaluation(det_annos, class_names, **kwargs)

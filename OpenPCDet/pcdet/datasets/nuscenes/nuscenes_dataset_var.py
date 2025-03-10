import numpy as np

from .nuscenes_dataset import NuScenesDataset

class NuScenesDatasetVAR(NuScenesDataset):
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
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
        def get_template_prediction(num_samples, num_anchors):
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'score_all': np.zeros([num_samples, len(class_names)+1]),
                'boxes_lidar': np.zeros([num_samples, 7]), 
                'pred_labels': np.zeros(num_samples),
                'pred_vars': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            feature = box_dict['feature'].cpu().numpy()
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_scores_all = box_dict['pred_scores_all'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_vars = box_dict['pred_vars'].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0], 0)
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['score_all'] = pred_scores_all
            pred_dict['boxes_lidar'] = pred_boxes[:,:7]
            pred_dict['pred_labels'] = pred_labels
            pred_dict['pred_vars'] = pred_vars[:,:7]

            return pred_dict
        
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

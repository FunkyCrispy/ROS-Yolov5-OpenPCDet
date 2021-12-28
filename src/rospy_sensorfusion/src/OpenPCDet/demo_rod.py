import argparse
import glob
from pathlib import Path

# priority of using open3d
# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from .tools.visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch

from .pcdet.config import cfg, cfg_from_yaml_file
from .pcdet.datasets import DatasetTemplate
from .pcdet.models import build_network, load_data_to_gpu
from .pcdet.utils import common_utils
# from pcdet.datasets.rod.rod_dataset import RodDataset

import os
import time


### this script is based on /tools/demo_rod.py
### /tools/demo_rod.py is firstly realized to infer single pcl data
### /tools/demo_rod.py is based on /tools/demo.py
### /tools/demo.py is a openpcdet script to infer single pcl data


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', points=None):
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
        # self.root_path = root_path
        # self.ext = ext
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        # data_file_list.sort()
        # self.sample_file_list = data_file_list

        self.points = points

    def __len__(self):
        # return len(self.sample_file_list)
        return 1

    def __getitem__(self, index):
        # if self.ext == '.bin':
        #     # points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        #     points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 8)
        # elif self.ext == '.npy':
        #     points = np.load(self.sample_file_list[index])
        # else:
        #     raise NotImplementedError

        points = self.points

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/fengchen/projects/OpenPCDet-master/tools/cfgs/rod_models/pointpillar_fc.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='/home/fengchen/projects/OpenPCDet-master/output/kitti_models/pointpillar_fc/default/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--data_path', type=str, default='/home/fengchen/projects/OpenPCDet-master/data/rod/testing/velodyne',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def openpcdet_inference(model, demo_dataset):

    # args, cfg = parse_config()
    # logger = common_utils.create_logger()
    # logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # demo_dataset = DemoDataset(
    #         dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #         root_path=Path(args.data_path), ext=args.ext, logger=logger, points=pcl
    #     )
        
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    # model.cuda()
    # model.eval()

    det_result = []

    with torch.no_grad():

        data_dict = demo_dataset[0]

        # logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)

        # add to results
        # boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        det_result.append([pred_dicts[0]['pred_boxes'].cpu().numpy(), 
            pred_dicts[0]['pred_scores'].cpu().numpy(), 
            pred_dicts[0]['pred_labels'].cpu().numpy()])

        # stop showing the window
        # mlab.options.offscreen = True
        # V.draw_scenes(
        #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
        #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        # )

        # if not OPEN3D_FLAG:
        #     mlab.show(stop=True)

        # mlab.savefig(os.path.join(save_viz_path, pcl_name.split('/')[-1][:-4] + '.png'))
        # mlab.clf()
        # mlab.close()

    # logger.info('Demo done.')

    return det_result[0]


if __name__ == '__main__':
    main()

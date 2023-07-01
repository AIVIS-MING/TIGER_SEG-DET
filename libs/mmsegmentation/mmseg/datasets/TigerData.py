# Copyright (c) OpenMMLab. All rights reserved.
from fileinput import filename
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
from mmcv.utils import print_log
from ..utils import get_root_logger

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TIGERDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    # CLASSES = ('background','invasive tumor', 'tumor-asso stoma', 'in-situ tumor', 'healthy glands', 
    #         'necrosis not in-situ', 'inflamed stroma', 'rest')

    # PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], 
    #             [255, 255, 0], [0,255,255], [255, 0, 255]]

    # CLASSES = ('invasive tumor', 'tumor-asso stoma', 'in-situ tumor', 'healthy glands', 
    #         'necrosis not in-situ', 'inflamed stroma', 'rest')

    # PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], 
    #             [255, 255, 0], [0,255,255], [255, 0, 255]]

    # CLASSES = ('invasive tumor', 'tumor-asso stoma', 'rest')

    # PALETTE = [[255, 0, 0], [0, 255, 0], [255, 0, 255]]

    CLASSES = ('invasive tumor', 'tumor-asso stoma', 'rest',  'inflamed stroma')

    PALETTE = [[255, 0, 0], [0, 255, 0], [255, 0, 255], [0, 0, 255]]

    def __init__(self, **kwargs):
        super(TIGERDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files


    def load_annotations(self,
                         img_dir, # The txt file for *_folder
                         data_root):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to txt file with indicate the traing image
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        file_txt_fold = open(img_dir, 'r')
        Lines = file_txt_fold.readlines()
        for line in Lines:
            img_name = line.split('\n')[0]
            filename_with_folder = osp.join(data_root, 'images',img_name)
            # filename_with_folder = osp.join(data_root, img_name)
            img_info = dict(filename = filename_with_folder)

            mask_name_with_path = filename_with_folder
            mask_name_with_path = mask_name_with_path.replace('images', 'masks')
            # mask_name_with_path = mask_name_with_path.replace('.jpg', '.png')
            # mask_name_with_path = mask_name_with_path.replace('images_patch_512_4x', 'masks_patch_512_4x')
            
            img_info['ann'] = dict(seg_map=osp.join(data_root, mask_name_with_path))

            img_infos.append(img_info)
        
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())

        # for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
        #     img_info = dict(filename=img)
        #     if ann_dir is not None:
        #         seg_img = img
        #         seg_map = seg_img.replace(
        #             img_suffix, '_instance_color_RGB' + seg_map_suffix)
        #         img_info['ann'] = dict(seg_map=seg_map)
        #     img_infos.append(img_info)

        # print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

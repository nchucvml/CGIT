# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import tempfile
import time
import warnings
import cv2
import numpy as np
import tqdm


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
# constants
WINDOW_NAME = "mask2former demo"
# pd_size = []
pd_name = []
# pd_avg = []
global pd_size
#global pd_name
global pd_avg
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75, #0..9 #0.98, #0.5
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    pd_name = []
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    #print('in')
    demo = VisualizationDemo(cfg)
    #print(np.unique(demo))
    print("cfg.DATASETS.TEST[0]: ",cfg.DATASETS.TEST[0])
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")

            start_time = time.time()
            predictions, visualized_output, mask = demo.run_on_image_2(img)  # add_mask *** for binary ***
            # print(predictions.shape)
            # predictions, visualized_output, mask = demo.run_on_image(img)
            # print(mask)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            pd_name = path
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"

                    out_filename = args.output
                #print(visualized_output)
                #print(np.unique(visualized_output))

                # ** 2022-11-15 以下12行 ** #
                myoutput=np.zeros((visualized_output.shape[1],visualized_output.shape[2]))
                # print("myoutput.shape:")
                # print(myoutput.shape)#c*w*h
                # print(visualized_output.shape[1])#w
                # print(visualized_output.shape[0])#數量
                # print(visualized_output.shape[2])#h
                # print(visualized_output.shape) #(5,522,775)#5:細胞數
                for i in range(visualized_output.shape[0]):
                    myoutput = myoutput+visualized_output[i]*(i+1)

                cv2.imwrite(out_filename, myoutput)
                # ** 2022-11-15 以上12行 ** #

                # # ** 2022-11-15 以下3行 ** #
                # visualized_output.save(out_filename)
                # #print(cv2.cvtColor(visualized_output.get_image()[:, :, ::-1],cv2.COLOR_BGR2RGB))
                # cv2.imwrite(out_filename + '.png', mask) #add
                # # ** 2022-11-15 以上3行 ** #

            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit


#python demo/demo.py --config-file configs/coco/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_100ep.yaml --input datasets/glaspng/val2017/*.png --output post_processing/glas_1 --opts MODEL.WEIGHTS output/model_final_ablation1.pth
# python demo/demo.py --config-file configs/coco/instance-segmentation/dinat/maskformer2_dinat_large_IN21k_384_bs16_100ep.yaml --input datasets/Breast_cancer/breast_open/val/images/*.png --output post_processing/breast --opts MODEL.WEIGHTS output/breast.pth

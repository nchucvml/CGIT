# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import numpy as np #add
import cv2
import torch
import json
import os
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures.instances import Instances
pd_size = []
# pd_name = []
pd_avg = []
#global pd_size
global pd_name
# global pd_avg

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image) #type(predictions)=dict
        #print(predictions)
        # print("type:")
        # print(type(Instances))
        # import pprint
        # pprint.pprint(predictions)
        # tf = open("C:/Users/cvml-142/Desktop/myDictionary.json", "w")
        # json.dump(predictions, tf)
        # tf.close()
        import csv
        DATA_DIR = r'C:/Users/cvml-142/Desktop/'
        # with open(DATA_DIR+'output.csv', 'wb') as output:
        #     writer = csv.writer(output)
        #     for i in predictions.items():
        #         writer.writerow([key, value])
        # import json
        # a = json.dumps(predictions)
        # file = open(DATA_DIR+'jsonfile.json','w')
        # file.write(a)
        # file.close()

        # make mask
        pred = (predictions["instances"]._fields)["pred_masks"].cpu().numpy()  # bool 类型
        #print(pred)
        binary_mask = np.zeros((pred.shape[1], pred.shape[2]))
        #################################################################
        # instances = predictions["instances"].to(self.cpu_device)
        # flag = False
        # for index in range(len(pred)):  # 100
        #     # print(instances[index].scores)
        #     score = instances[index].scores[0]
        #     if score > 0.98:  # 置信度设置 #0.75
        #         if flag == False:
        #             binary_mask = pred[index]
        #             flag = True
        #################################################################
        #print(binary_mask)
        for i in range(pred.shape[0]):
            # instances = predictions["instances"].to(self.cpu_device)
            # flag = False
            # for index in range(len(instances)):  # 100
            #     # print(instances[index].scores)
            #     score = instances[index].scores[0]
            #     if score > 0.2:  # 置信度设置 #0.75
            #         if flag == False:
            #             binary_mask = pred[index]
            #             flag = True

            binary_mask = binary_mask + pred[i, :, :]
        #binary_mask[binary_mask > 0] = 255 #change to segmentation result
        #binary_mask[binary_mask > 0] += 100

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                #vis_output = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"),0.5)
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                #vis_output = visualizer.draw_instance_predictions(predictions=instances)
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                # 取得分大于阈值的实例
                instances_ = Instances(instances.image_size)
                flag = False
                for index in range(len(instances)):  # 100
                    # print(instances[index].scores)
                    score = instances[index].scores[0]
                    # if score > 0.9:  # 置信度设置 #0.75 #0.95 #無關 與run on img也猜
                    if score > 0.9:  # 置信度设置 #0.75 #0.95 #無關
                        ''''''''''''''''''''''''
                        # print(instances.pred_masks.shape)
                        # mask = torch.squeeze(instances[index].pred_masks).numpy() * 255
                        # import numpy as np
                        # mask = np.array(mask, np.uint8)  # 类型转换后才能输入查找轮廓
                        # contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # areas = []
                        # for cnt in contours:
                        #     areas.append(cv2.contourArea(cnt))
                        #
                        # for cnt in contours:
                        #     if cv2.contourArea(cnt) < max(areas):
                        #         cv2.drawContours(mask, [cnt], contourIdx=-1, color=0, thickness=-1)
                        # mask = torch.from_numpy(mask / 255).unsqueeze(0)
                        ''''''''''''''''''''''''
                        if flag == False:
                            instances_ = instances[index]
                            flag = True
                        else:
                            instances_ = Instances.cat([instances_, instances[index]])
                #print(instances_.pred_masks.shape)
                vis_output = visualizer.draw_instance_predictions(predictions=instances_)

        return predictions, vis_output, binary_mask  # add binary_mask

    def run_on_image_2(self, image):
        global pd_name

        vis_output = None
        predictions = self.predictor(image)  # type(predictions)=dict
        # print(predictions) #

        # make mask
        pred = (predictions["instances"]._fields)["pred_masks"].cpu().numpy()  # bool 类型
        # print(pred)

        binary_mask = np.zeros((pred.shape[1], pred.shape[2]))
        print(pred.shape)  # (100,442,581)

        # print(binary_mask)
        for i in range(pred.shape[0]):
            binary_mask = binary_mask + pred[i, :, :]
        # print(image)
        # print(image.shape)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)

                # 取得分大于阈值的实例
                #instances_ = Instances(instances.image_size)
                instances_ = None
                # print("_shape: ",instances_.shape) # 一堆矩陣
                flag = False
                a = 0
                i = 0

                for index in range(len(instances)):  # 100
                    # print(instances[index].scores)
                    score = instances[index].scores[0]

                    if score > 0.5:  # 0.9 # 0.93 # confidence setting #0.75
                        i = i + 1

                        # ************************
                        # print("-----")
                        # print("score: ", score)
                        # print("index: ", index)
                        # ************************

                        # 0.75/0.85/0.95/0.8/0.87/0.92/0.97/0.93
                        # a = 0
                        a = torch.add(a, score)
                        # print("a: ", a)
                        if flag == False:
                            instances_ = instances[index]
                            flag = True
                        else:
                            instances_ = Instances.cat([instances_, instances[index]])

                # print("final_a: ", a)
                # print("i: ", i)
                with open('test.txt', 'a') as file0:
                    if i == 0:
                        print("\navg: ", a, file=file0)
                        print("\navg: ", a)
                        # print('%d' % t, '%s' % s, file=file0)
                    else:
                        a = a/i
                        print("\navg: ", a)
                        print("\navg: ", a, file=file0)
                    # b = a / i
                    # print("avg: ", b)
                #pd_size = a

                # ddict = {'size': pd_size,
                #          'name': pd_name, #pd_name
                #          'avg': pd_size, #pd_avg
                #          }
                # dataframe = pd.DataFrame(ddict,index=[0])
                #
                # csv_path = "./_" + str(111) + ".csv"  # 存csv檔案的路徑
                # dataframe.to_csv(csv_path, index=False, sep=',')

                # if a == 0:
                #     print("avg: ", a)
                # else:
                #     # print("avg: ", a/i)
                #     print("avg: ", b)
                # print(instances_.pred_masks.shape)
                # torch.Size([5, 442, 581])
                if instances_ != None:
                    numpy_type = instances_.pred_masks.cpu().detach().numpy()
                else:
                    # create ndarray
                    numpy_type = np.empty((0,768,768))
                # print(numpy_type.shape)

        return predictions, numpy_type, binary_mask

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """
    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

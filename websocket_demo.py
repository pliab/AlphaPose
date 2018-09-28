import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_webcam import WebcamLoader, DetectionLoader, DetectionProcessor, DataViz, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import time
from fn import getTime
import cv2

import signal
from datetime import timedelta
import asyncio
import threading
import websockets

from pPose_nms import format_json

args = opt
args.dataset = 'coco'

env_name = 'alphapose-torch'
ws_port = 8612
msg_fps = 24
outputSize = (1920, 960)

window_title = os.path.basename(__file__)
script_start_time = time.time()
stopped = False
fps = 1
fps_history = []
humans_history = []

class DetectionThread (threading.Thread):
  def __init__(self, args):
    threading.Thread.__init__(self)
    self.data = []
    self.posebatch = args.posebatch

    # Load input video
    webcam = args.webcam
    mode = args.mode
    data_loader = WebcamLoader(webcam).start()

    (fourcc,fps,frameSize) = data_loader.videoinfo()
    self.frameSize = frameSize
    print(frameSize)

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    self.det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    self.pose_model.cuda()
    self.pose_model.eval()

  def run(self):
    self.dataviz = DataViz().start()
    sys.stdout.flush()
    batchSize = self.posebatch
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
        }

    while not stopped:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = self.det_processor.read()

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)

            humans = []

            if boxes is None or boxes.nelement() == 0:
                self.dataviz.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                if args.profile:
                    sys.stdout.flush()
                    sys.stdout.write(u"\u001b[1000D" + 'No humans detected')
                continue

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            hm = hm.cpu().data
            self.dataviz.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            self.data = self.dataviz.parse(boxes, scores, hm, pt1, pt2)

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        # Show some real time stats
        now = time.time()
        fps = round(1/(now - start_time))
        fps_history.append(fps)
        avg_fps = round(sum(fps_history) / max(len(fps_history), 1))

        humans = self.data
        human_count = len(humans)
        humans_history.append(human_count)
        avg_humans = round(sum(humans_history) / max(len(humans_history), 1))

        stats = '\u001b[45;1m det:{dt:.3f} pose:{pt:.2f} post-process:{pn:.4f} \u001b[44;1m FPS:{fps} ({lfps}>{avg_fps}>{hfps}) \u001b[46;1m Humans:{hc} ({lh}>{avg_humans}>{hh}) \u001b[47;1m {t} \u001b[0m'.format(
            fps=fps, avg_fps=avg_fps, lfps=min(fps_history), hfps=max(fps_history),
            dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']),
            hc=human_count, lh=min(humans_history), hh=max(humans_history), avg_humans=avg_humans,
            t=timedelta(seconds=int(now - script_start_time))
            )

        sys.stdout.flush()
        sys.stdout.write(u"\u001b[1000D" + stats)

  def stop(self):

      while(self.dataviz.running()):
          pass
      self.dataviz.stop()

  def get(self):
    return format_json(self.data, self.frameSize, outputSize)

class MSGThread (threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    self.connected = set()

  def run(self):
    while not stopped:
      data = detectionThread.get()
      if data:
        for websocket in self.connected.copy():
          coro = websocket.send(data)
          future = asyncio.run_coroutine_threadsafe(coro, loop)

      time.sleep(1/msg_fps)

  async def handler(self, websocket, path):
    self.connected.add(websocket)
    try:
      await websocket.recv()
    except websockets.exceptions.ConnectionClosed:
      pass
    finally:
      self.connected.remove(websocket)


if __name__ == "__main__":
    print('Server starting, Ctrl + c to stop')

    detectionThread = DetectionThread(args)
    msgThread = MSGThread()

    try:
        detectionThread.start()
        msgThread.start()

        ws_server = websockets.serve(msgThread.handler, '0.0.0.0', ws_port)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ws_server)
        loop.run_forever()
    except KeyboardInterrupt:
        stopped = True

        sys.stdout.flush()
        sys.stdout.write("\nStopping...\n")

        detectionThread.stop()

        loop.close()

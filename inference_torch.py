import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import psutil
import time
from queue import Queue, Empty

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--start',dest='start',type=int,default=0)
parser.add_argument('--resume', dest='resume', action='store_true', help='auto calc start index & resume render')

parser.add_argument('--device_id',dest='device_id',type=int,default=0)
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--rbuffer', dest='rbuffer', type=int, default=200)

parser.add_argument('--scene', dest='scene', type=float, default=50)
parser.add_argument('--rescene', dest='rescene', type=str, default="mix",help="copy/mix")
parser.add_argument('--exp', dest='exp', type=int, default=1)

args = parser.parse_args()
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
spent = time.time()

if not os.path.exists(args.output):
    os.mkdir(args.output)

if args.device_id != -1:
    device = torch.device("cuda")
    torch.cuda.set_device(args.device_id)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)    
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
else:
    device = torch.device("cpu")
    try:
        from model_cpu.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from model_cpu.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")

model.eval()
model.device()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

start = args.start
if args.resume:
    maxc = 0
    for f in os.listdir(args.output):
        tempcnt = int(os.path.splitext(f)[0])
        if tempcnt > maxc:
            maxc = tempcnt
    if maxc != 0:
        start = int(((maxc - 1) / (2 ** args.exp))) + 1
    start += args.start

videogen = [f for f in os.listdir(args.img) if 'png' in f]
if start != 0:
    templist = []
    pos = start - 1
    end = len(videogen)
    while pos != end:
        templist.append(videogen[pos])
        pos = pos + 1
    videogen = templist
tot_frame = len(videogen)
videogen.sort(key= lambda x:int(x[:-4]))
lastframe = cv2.imdecode(np.fromfile(os.path.join(args.img, videogen[0]), dtype=np.uint8), 1)[:, :, ::-1].copy()
videogen = videogen[1:]
h, w, _ = lastframe.shape

    
def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    cnt = 0 if start == 0 else (start - 1) * (2 ** args.exp) + 1
    cnt += 1
    while True:
        item = write_buffer.get()
        if item is None:
            break
        cv2.imencode('.png', item[:, :, ::-1])[1].tofile('{}/{:0>9d}.png'.format(user_args.output,cnt))
        cnt += 1

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             frame = cv2.imdecode(np.fromfile(os.path.join(user_args.img, frame), dtype=np.uint8), 1)[:, :, ::-1].copy()
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, exp):
    global model
    middle = model.inference(I0, I1, args.scale)
    if exp == 1:
        return [middle]
    first_half = make_inference(I0, middle, exp=exp - 1)
    second_half = make_inference(middle, I1, exp=exp - 1)
    return [*first_half, middle, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
write_buffer = Queue(maxsize=args.rbuffer)
read_buffer = Queue(maxsize=args.rbuffer)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
while True:
    frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    diff = cv2.absdiff(lastframe[:, :, ::-1], frame[:, :, ::-1]).mean()
    if diff > args.scene:
        output = []
        if args.rescene == "mix":
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose(
                    (cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()),
                    (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        else:
            for i in range((2 ** args.exp) - 1):
                output.append(I0)
    else:
        output = make_inference(I0, I1, args.exp)
    write_buffer.put(lastframe)
    for mid in output:
        mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
        write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
write_buffer.put(lastframe)
while(not write_buffer.empty()):
    time.sleep(1)
pbar.close()
print("spent {}s".format(time.time()-spent))

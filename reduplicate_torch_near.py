import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import math
import _thread
import psutil
import time
from queue import Queue, Empty
import sys
import shutil

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', type=str, default="")

parser.add_argument('--device_id',dest='device_id',type=int,default=0)
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--rbuffer', dest='rbuffer', type=int, default=200)

parser.add_argument('--scene', dest='scene', type=float, default=50)
parser.add_argument('--rescene', dest='rescene', type=str, default="mix",help="copy/mix")
parser.add_argument('--static', dest='static', type=int, default=24,help="when duplicate frames num >= static copy left frames")

args = parser.parse_args()
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
spent = time.time()

#加载设备和模型
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

#测试显存占用
print("测试显存占用....")
files = []
l1 = ""
l2 = ""

for f in os.listdir(args.img):
    files.append(os.path.join(args.img,f))

fs = []
pos = 1
maxc = 0
need = 0
im1 = files[0].replace(args.img+"\\","")
while(pos != len(files)):
    try:
        im2 = files[pos].replace(args.img+"\\","")
        n1 = int(os.path.splitext(im1)[0])
        n2 = int(os.path.splitext(im2)[0])
        need = n2 - n1 - 1
        if need != 0:
            fs.append(os.path.join(args.img,im1))
            fs.append(os.path.join(args.img,im2))
        if(need > maxc):
            if(need < args.static):
                maxc = need
                l1 = os.path.join(args.img,im1)
                l2 = os.path.join(args.img,im2)
        im1 = im2
        pos = pos + 1
    except:
        break
files = fs
#推导exp
exp = int(math.sqrt(need+1)) + 1 
if 2**(exp-1) - 1 == need:
    exp = exp - 1

padding = (0, 0, 0, 0)
try:
    l1 = cv2.imdecode(np.fromfile(l1, dtype=np.uint8), 1)[:, :, ::-1].copy()
    l2 = cv2.imdecode(np.fromfile(l2, dtype=np.uint8), 1)[:, :, ::-1].copy()
    h,w,_ = l1.shape
    tmp = max(32, int(32 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    I0 = torch.from_numpy(np.transpose(l1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = torch.from_numpy(np.transpose(l2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I0 = pad_image(I0)
    I1 = pad_image(I1)
    make_inference(I0,I1,exp)
except:
    print("可能没有需要补足的重复帧")
    print("或者显存/内存溢出")
    sys.exit(1)
print("测试成功")
torch.cuda.empty_cache()

def build_read_buffer(user_args, read_buffer, files):
    try:
        for frame in files:
             frame = cv2.imdecode(np.fromfile(os.path.join(user_args.img, frame), dtype=np.uint8), 1)[:, :, ::-1].copy()
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def clear_write_buffer(write_buffer,files):
    while True:
        item = write_buffer.get()
        if item is None:
            break
        output = item[0]
        st = item[1]
        end = item[2]
        drop = item[3]
        kpts = [] #保留帧
        if drop:
            need  = end - st - 1
            vgen = output
            #丢弃多余帧(保留最邻近时刻)

            #需要的时刻
            nts = 1 / (need+1)
            ntsc = []
            ntp = 0
            while ntp < 1:
                ntp = ntp + nts
                if (ntp + 0.000001) < 1:
                    ntsc.append(ntp)

            #补出来的时刻
            its = 1 / (len(vgen)+1)
            itsc = {}
            itp = 0
            n = 0
            while itp < 1:
                itp = itp +its
                if (itp + 0.000001) < 1:
                    itsc[itp] = vgen[n]
                    n += 1

            #要保留的时刻
            for i in ntsc:
                min = 1
                kpt = ""
                for k in itsc:
                    if abs(i-k) < min:
                        min = abs(i-k)
                        kpt = itsc[k]
                kpts.append(kpt)

        #写入文件
        cnt = st + 1
        if drop != True:
            kpts = output
        for s in kpts:
            cv2.imencode('.png', s[:, :, ::-1])[1].tofile('{}/{:0>9d}.png'.format(args.img,cnt))
            cnt = cnt + 1
        pbar.update(1)

read_buffer = Queue(maxsize=args.rbuffer)
write_buffer = Queue(maxsize=args.rbuffer)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, files))
_thread.start_new_thread(clear_write_buffer, (write_buffer,files))
        
pairs = int(len(files) / 2)

#print(len(files))
#print(pairs)
#print(files)
#sys.exit(0)
pbar = tqdm(total = pairs)
pos = 0
while pos != len(files):
    #print(pos,pos+1,files[pos],files[pos+1])

    fr0 = files[pos]
    fd0 = read_buffer.get()
    num1 = int(os.path.splitext(fr0.replace(args.img+"\\",""))[0])
    I0 = torch.from_numpy(np.transpose(fd0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I0 = pad_image(I0)
    pos = pos + 1

    fr1 = files[pos]
    fd1 = read_buffer.get()
    num2 = int(os.path.splitext(fr1.replace(args.img+"\\",""))[0])
    I1 = torch.from_numpy(np.transpose(fd1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    pos = pos + 1

    n = num2 - num1 - 1
    #print("num1: {} num2: {} need:{}".format(num1,num2,n))
    output = []
    drop = False
    if n >= args.static:
        for i in range(n):
            output.append(I0)
    else:
        exp = int(math.sqrt(n+1)) + 1 #推导exp
        if 2**(exp-1) - 1 == n:
            exp = exp - 1
        diff = cv2.absdiff(fd0[:, :, ::-1], fd1[:, :, ::-1]).mean()
        if diff > args.scene:
            if args.rescene == "mix":
                step = 1 / (n+1)
                alpha = 0
                for i in range(n):
                    alpha += step
                    beta = 1-alpha
                    output.append(torch.from_numpy(np.transpose(
                        (cv2.addWeighted(fd1[:, :, ::-1], alpha, fd0[:, :, ::-1], beta, 0)[:, :, ::-1].copy()),
                        (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            else:
                for i in range(n):
                    output.append(I0)
        else:
            #print(num1,num2)
            output = make_inference(I0,I1,exp)
            drop = True
    op = []
    for mid in output:
         mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
         op.append(mid[:h,:w])
    if n == len(op):
         drop = False
    #print(n," in",len(op))
    write_buffer.put([op,num1,num2,drop])
print("等待文件写入")
while not os.path.exists(os.path.join(args.img,'{}/{:0>9d}.png'.format(args.img,num2-1))):
    time.sleep(1)
pbar.close()
print("spent {}s".format(time.time() - spent))

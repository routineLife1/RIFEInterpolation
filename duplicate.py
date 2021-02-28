import subprocess as sp
import argparse
import os

parser = argparse.ArgumentParser(description='除去重复帧')
parser.add_argument('--img', dest='img', type=str, default=None,help="图片路径")
parser.add_argument('--dup', dest='dup', type=float, default=1,help="除去重复帧数值 小于场景切换识别数值，大于0")
parser.add_argument('--threads', dest='threads', type=float, default=16,help="识别进程数 最大128")
args = parser.parse_args()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
l = []
for f in os.listdir(args.img):
    l.append(f)
c  = bytes("{}\n{}\n{}\n{}\n".format(args.img,args.dup,len(l),args.threads),'ansi')

pipe = sp.run("{}\\delgen.exe".format(dname),bufsize=-1,input=c)
@echo off&chcp 65001 >nul
cd /d %~dp0
set /p input=输入文件:
ffmpeg -i %input% -f image2 input\%%^09d.png
py\python.exe duplicate.py --img %~dp0\input --dup 1
py\python.exe reduplicate_torch_near.py --img %~dp0\input --rbuffer 300 --scene 50 --rescene mix --static 24 --device_id 0 --model train_log
py\python.exe inference_torch.py --img %~dp0\input --output out --start 0 --device_id 0 --model train_log --scale 1.0 --rbuffer 200 --scene 50 --rescene mix --exp 2
ffmpeg -r 95.904 -i out\^%%09d.png -c:v h264 -crf 16 -color_range tv -color_primaries bt709 -colorspace bt709 out.mp4
pause

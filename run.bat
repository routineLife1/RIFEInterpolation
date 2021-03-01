@echo off&chcp 65001 >nul
set /p input=输入文件:
ffmpeg -i %input% -f image2 %~dp0\input\%%^09d.png
python.exe duplicate.py --img %~dp0\input --dup 2
python.exe reduplicate_torch_near.py --img %~dp0\input --rbuffer 300 --scale 0.25 --scene 50 --rescene mix --static 24 --device_id 0 --model train_log
python.exe inference_torch.py --img %~dp0\input --output %~dp0\out --scale 0.25 --exp 2 --start 0 --device_id 0 --model train_log --rbuffer 500 --scene 50 --rescene mix
ffmpeg -r 95.904 -i out\^%%09d.png -c:v h264 -crf 16 -color_range tv -color_primaries bt709 -colorspace bt709 %~dp0\out.mp4
pause

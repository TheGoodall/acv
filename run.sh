docker run -m 10G --rm -it --runtime=nvidia  --ipc=host -e NVIDIA_VISIBLE_DEVICES=0 -v ~/projects/cv/:/workspace cv python cv.py

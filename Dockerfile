FROM pytorch/pytorch
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -c "import torch, torchvision; a = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)"
RUN python -c "import torch, torchvision; a = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)"
ADD data /workspace/data

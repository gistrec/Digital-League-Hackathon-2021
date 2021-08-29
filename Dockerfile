FROM python:3.8-slim-buster

RUN pip3 install ipython albumentations pandas tqdm seaborn torchsummary \
                 torch torchvision matplotlib tqdm sklearn pytorch_lightning \
                 albumentations pathlib numpy timm

ADD resnet.onnx .
ADD run.py .

ENTRYPOINT ["python3", "run.py"]
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.8.0

WORKDIR /usr/src/app

COPY ./* ./

RUN pip install pytorch-lightning torchvision jupyterlab

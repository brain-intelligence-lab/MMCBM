FROM python:3.11-slim
LABEL authors="liuyang"

VOLUME ["/intervention"]
WORKDIR /intervention
COPY . .
RUN mkdir -p  /root/.cache/torch/hub/checkpoints &&\
    mv efficientnet-b0-355c32eb.pth /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth && \
    apt-get update && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers matplotlib monai openai seaborn tqdm scikit-learn   \
    albumentations tencentcloud-sdk-python torchmetrics opencv-python-headless gradio==3.47 && \
    pip uninstall opencv-python -y
CMD ["python", "interface.py"]
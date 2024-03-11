FROM python:3.11-slim
LABEL authors="liuyang"

VOLUME ["/intervention"]
WORKDIR /intervention
COPY . .
RUN mkdir -p  /root/.cache/torch/hub/checkpoints &&\
    mv efficientnet-b0-355c32eb.pth /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth && \
    apt-get update && apt-get install -y ffmpeg libsm6 && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers matplotlib monai openai seaborn tqdm scikit-learn gradio  \
    albumentations tencentcloud-sdk-python torchmetrics
CMD ["python", "interface.py"]
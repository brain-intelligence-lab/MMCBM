FROM python:3.11-slim
LABEL authors="liuyang"

VOLUME ["/gradio"]
WORKDIR /gradio
COPY efficientnet-b0-355c32eb.pth .
RUN mkdir -p  /root/.cache/torch/hub/checkpoints &&\
    mv efficientnet-b0-355c32eb.pth /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth && \
    apt-get update && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers matplotlib monai openai seaborn tqdm scikit-learn   \
    albumentations tencentcloud-sdk-python torchmetrics opencv-python-headless gradio && \
    pip uninstall opencv-python -y

EXPOSE 7860
CMD ["/bin/bash", "web_human.sh"]
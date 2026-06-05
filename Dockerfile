# Container image for the pneumonia classification Lambda.
# Based on the official AWS Lambda Python base image.

FROM public.ecr.aws/lambda/python:3.12

# Keep the Hugging Face cache inside the image so cold starts don't download.
ENV HF_HOME=${LAMBDA_TASK_ROOT}/hf_home

# Install CPU-only PyTorch (smaller, no GPU needed) plus the other libraries.
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-lambda.txt

# Pre-download the model at build time so it's baked into the image.
RUN python -c "from transformers import AutoImageProcessor, AutoModelForImageClassification; \
m='nickmuchi/vit-finetuned-chest-xray-pneumonia'; \
AutoImageProcessor.from_pretrained(m); \
AutoModelForImageClassification.from_pretrained(m)"

# Run offline at request time (model is already in the image).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# OPTIONAL: to use your fine-tuned weights, drop the .pth file next to this
# Dockerfile and uncomment the next line before building.
# COPY best_cxrpretrained_vit_3000_clean.pth ${LAMBDA_TASK_ROOT}/

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.handler"]

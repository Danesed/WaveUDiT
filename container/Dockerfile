FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir nibabel==5.3.2 scipy==1.15.* \
    && rm -rf /root/.cache/pip

WORKDIR /opt/algorithm
ENV PYTHONPATH=/opt/algorithm \
    BRATS_INPUT=/input \
    BRATS_OUTPUT=/output \
    BRATS_CKPT=/opt/algorithm/model/model.pth \
    BRATS_TTA=crop_only \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY wavedit /opt/algorithm/wavedit
COPY predict.py /opt/algorithm/predict.py
COPY tta_ensemble.py /opt/algorithm/tta_ensemble.py
COPY model /opt/algorithm/model

ENTRYPOINT ["python", "-u", "predict.py"]
CMD ["--input-dir", "/input", "--output-dir", "/output"]

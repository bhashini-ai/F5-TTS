#!/usr/bin/env bash
docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p 7860:7860 -v /home/models/huggingface/hub:/root/.cache/huggingface/hub -v /home/models/huggingface/modules:/root/.cache/huggingface/modules -v /home/models/huggingface/samples:/samples f5-tts

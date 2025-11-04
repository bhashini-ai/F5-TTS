#!/usr/bin/env bash
docker run -it --rm --name "f5-server" --gpus all --net host -v /home/models:/models -v ~/git/F5-TTS/:/workspace/F5-TTS/ --shm-size=2g triton-f5-tts:24.12

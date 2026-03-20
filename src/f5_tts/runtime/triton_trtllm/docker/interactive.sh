#!/usr/bin/env bash
docker run -it --rm --name "f5-server" --gpus '"device=1"' --net host -v /home/models:/models -v ~/git/F5-TTS/:/workspace/F5-TTS/ -v /data3/TTS_Speech_Recording/:/training -v /home/reference-audios:/reference-audios --shm-size=2g triton-f5-tts:24.12

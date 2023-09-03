# syntax=docker/dockerfile:1

FROM python:3.10-bullseye

EXPOSE 8080

WORKDIR /app

COPY . .

RUN apt update && apt install -y -qq ffmpeg aria2

RUN pip3 install -r requirements.txt

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/pretrained/D40k.pth -d /content/RVC-Train/pretrained -o D40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/pretrained/G40k.pth -d /content/RVC-Train/pretrained -o G40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/pretrained/f0D40k.pth -d /content/RVC-Train/pretrained -o f0D40k.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/pretrained/f0G40k.pth -d /content/RVC-Train/pretrained -o f0G40k.pth

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/uvr5_weights/HP2-human-vocals-non-human-instrumentals.pth -d /content/RVC-Train/uvr5_weights -o HP2-human-vocals-non-human-instrumentals.pth
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/uvr5_weights/HP5-main-melody-vocals-other-instruments.pth -d /content/RVC-Train/uvr5_weights -o HP5-main-melody-vocals-other-instruments.pth

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/hubert_base.pt -d /content/RVC-Train -o hubert_base.pt

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Goutham03/sp37/resolve/main/rmvpe.pt -d /content/RVC-Train -o rmvpe.pt

VOLUME [ "/app/weights", "/app/opt" ]

CMD ["python3", "infer-web.py"]
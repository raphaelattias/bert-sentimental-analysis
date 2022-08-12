FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime	

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m raphael

RUN chown -R raphael:raphael /home/raphael/

COPY --chown=raphael . /home/raphael/app/

USER raphael

RUN cd /home/raphael/app/ && pip3 install -r requirements.txt

WORKDIR /home/raphael/app

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
RUN chmod +x ./Anaconda3-2024.02-1-Linux-x86_64.sh
RUN bash ./Anaconda3-2024.02-1-Linux-x86_64.sh -b -p /opt/anaconda
RUN rm Anaconda3-2024.02-1-Linux-x86_64.sh

RUN echo ". /opt/anaconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
    
ENV PATH /opt/anaconda/bin:$PATH

RUN conda install -y python=3.10.13

RUN mkdir /app
WORKDIR /app

COPY . /app

RUN conda env update -f ./environment.yml

RUN mkdir model_store

RUN torch-model-archiver --model-name expansion-net-v2 \
    --version 0.15.5 --model-file /app/ExpansionNetV2-serve/End_ExpansionNet_v2.py \
    --serialized-file /app/rf_model.pth --handler /app/ExpansionNetV2-serve/image_captioning_handler.py \
    --extra-files "/app/ExpansionNetV2-serve/demo_coco_tokens.pickle,/app/ExpansionNetV2-serve/layers.py,/app/ExpansionNetV2-serve/captioning_model.py,/app/ExpansionNetV2-serve/swin_transformer_mod.py,/app/ExpansionNetV2-serve/image_utils.py,/app/ExpansionNetV2-serve/language_utils.py,/app/ExpansionNetV2-serve/masking.py"

RUN mv /app/expansion-net-v2.mar /app/model_store/

CMD ["torchserve", "--start", \
      "--model-store", "model_store", \
      "--models", "expansion-net-v2=expansion-net-v2.mar", \
      "--ts-config", "/app/ExpansionNetV2-serve/config.properties", \
      "--ncs"]

EXPOSE 8080
EXPOSE 8081
EXPOSE 8082
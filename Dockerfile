FROM pytorch/torchserve:0.10.0-gpu
WORKDIR /home/model-server/

COPY . /home/model-server/
ENV APP_PATH /home/model-server
ENV MODEL_BASE_PATH $APP_PATH/ExpansionNetV2-serve
ENV MODEL_STORE_PATH $APP_PATH/model-store

RUN python3 -m pip install h5py==3.8.0 nvgpu
# RUN python3 -m pip install --upgrade torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
# RUN python3 -m pip install -r requirements.txt
# RUN mkdir $MODEL_STORE_PATH

RUN torch-model-archiver --model-name expansion-net-v2 \
    --version 0.15.5 --model-file $MODEL_BASE_PATH/End_ExpansionNet_v2.py \
    --serialized-file $MODEL_BASE_PATH/rf_model.pth --handler $MODEL_BASE_PATH/image_captioning_handler.py \
    --extra-files "$MODEL_BASE_PATH/demo_coco_tokens.pickle,$MODEL_BASE_PATH/layers.py,$MODEL_BASE_PATH/captioning_model.py,$MODEL_BASE_PATH/swin_transformer_mod.py,$MODEL_BASE_PATH/image_utils.py,$MODEL_BASE_PATH/language_utils.py,$MODEL_BASE_PATH/masking.py"

RUN mv $APP_PATH/expansion-net-v2.mar $MODEL_STORE_PATH/expansion-net-v2.mar

CMD ["torchserve", "--start", \
      "--models", "expansion-net-v2=expansion-net-v2.mar", \
      "--model-store", "$MODEL_STORE_PATH",\
      "--ts-config", "$MODEL_BASE_PATH/config.properties", \
      "--ncs"]

EXPOSE 8080
EXPOSE 8081
EXPOSE 8082
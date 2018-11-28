USER=ycl
model_dir=/home/aigroup/share/ycl/update/wxbot/export
#docker pull tensorflow/serving:latest-gpu

#NV_GPU="0" nvidia-docker run -p 8501:8501 --name tfserving_bert \
#--mount type=bind,source=/home/aigroup/share/ycl/wxbot/export,target=/models/bert \
#-e MODEL_NAME=bert -t tensorflow/serving:latest-gpu &


# optimized for CPU computation

git clone https://github.com/tensorflow/serving
cd serving

docker build --pull -t $USER/tensorflow-serving-devel-gpu \
  -f tensorflow_serving/tools/docker/Dockerfile.devel-gpu .

docker build -t $USER/tensorflow-serving-gpu \
  --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel-gpu \
  -f tensorflow_serving/tools/docker/Dockerfile.gpu .

NV_GPU="0" nvidia-docker run -p 8501:8501 --name tfserving_bert \
--mount type=bind,source=$model_dir,target=/models/bert \
-e MODEL_NAME=bert -t ycl/tensorflow-serving-gpu &

NV_GPU='0' nvidia-docker start -i tfserving_bert &


#docker start -p 8502:8502 --name tfserving_bert -e MODEL_NAME=nert -t tensorflow/serving

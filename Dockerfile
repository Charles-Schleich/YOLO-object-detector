FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y libopencv-dev

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# COPY class.names .
# COPY image.jpg .
# COPY yolov7-tiny_480x640.onnx .
# COPY main.py .
# COPY . .

# CMD [ "python", "./main.py" ]

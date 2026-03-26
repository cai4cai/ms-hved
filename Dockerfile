FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ARG USER_ID
ARG GROUP_ID
ARG USER

RUN addgroup --gid $GROUP_ID $USER \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

WORKDIR /workspace

EXPOSE 8888

# COPY . .
# list all files for debugging
RUN ls -la .
# Copy requirements.txt from the current directory and install packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

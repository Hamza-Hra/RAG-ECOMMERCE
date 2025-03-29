#building docker image from python base image

FROM python:3.11

# Set the working directory in the container

WORKDIR /RagApp

# Copy the content of the local src directory to the working directory

COPY . /RagApp

# Install any dependencies

RUN pip install -r requirements.txt


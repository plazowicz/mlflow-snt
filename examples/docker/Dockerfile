# syntax=docker/dockerfile:experimental
FROM continuumio/miniconda3

WORKDIR /home

COPY conda.yaml conda.yaml

ARG conda_env_name=template

# download public key for github.com
RUN conda env create --name $conda_env_name -f conda.yaml && conda init bash

RUN echo "source activate $conda_env_name" > ~/.bashrc
ENV PATH /opt/conda/envs/$conda_env_name/bin:$PATH

FROM nvcr.io/nvidia/pytorch:22.04-py3

# 言語の設定
RUN apt-get update && apt-get install -y language-pack-ja-base language-pack-ja
ENV LANG='ja_JP.UTF-8'

# WORKDIRを設定
WORKDIR /root

# Miacisの導入
RUN git clone https://github.com/SakodaShintaro/Which_are_you

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh

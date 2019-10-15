## This code is modified from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa

## Script for downloading data
download_gg_large () {
  FID=$1
  DATAPATH=$2
  wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id="$FID -O tmp.html
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(cat tmp.html | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FID -O $DATAPATH
  rm -rf /tmp/cookies.txt tmp.html
}

DATAPATH=data

mkdir -p $DATAPATH
cd $DATAPATH
download_gg_large "1gpOaOl0BcUvYpgoOA2JpZY2z-BUhuBLX" word-embeddings.zip
download_gg_large "0B0ZXk88koS2KbDhXdWg1Q2RydlU" ko.zip
wget "https://www.dropbox.com/s/stt4y0zcp2c0iyb/ko.tar.gz?dl=1" -O ko.tar.gz

# ratsgo
unzip word-embeddings.zip

# Kyubong Park
mkdir -p fasttext
tar -zxvf ko.tar.gz -C fasttext
unzip ko.zip -d word2vec

cd ..
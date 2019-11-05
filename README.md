# 한국어 시각 질의응답을 위한 Bilinear Attention Networks (BAN)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)
![cuDNN 7.5](https://img.shields.io/badge/cudnn-7.5-green.svg?style=plastic)

이 코드 저장소는 우리말로 시각적 질의응답을 수행할 수 있도록 수집된 _KVQA_ 데이터셋을 학습할 수 있도록 [Bilinear Attention Networks](http://arxiv.org/abs/1805.07932) 모델을 구현하였습니다.

![Examples of KVQA](docs/assets/img/kvqa_examples.png)
![Overview of bilinear attention networks](docs/assets/img/ban_overview.png)

5회 교차 검증을 한 후 평균 점수는 다음 표와 같습니다.

| Embedding | Dimension |          All          |  Yes/No   |  Number   |   Other   | Unanswerable |
| --------- | :-------: | :-------------------: | :-------: | :-------: | :-------: | :----------: |
| [Word2vec](https://arxiv.org/abs/1310.4546)  | [200](https://github.com/Kyubyong/wordvectors)       |   37.23 ± 0.11    | **66.95** |   20.47   |   20.08   |  **93.57**   |
| [GloVe](https://nlp.stanford.edu/projects/glove/)     | [100](https://ratsgo.github.io/embedding)       |   37.91 ± 0.08    |   65.98   |   20.76   |   21.97   |    93.18     |
| [fastText](https://arxiv.org/abs/1607.04606)  | [200](https://github.com/Kyubyong/wordvectors)       | **38.16 ± 0.13**  |   66.05   | **20.79** | **22.45** |    92.72     |
| [BERT](https://arxiv.org/abs/1810.04805)      | [768](https://github.com/google-research/bert)       | 37.95  ± 0.10 |   63.77   |   20.46   |   22.35   |    92.92     |


이 코드 저장소의 일부 코드는 @hengyuan-hu의 [저장소](https://github.com/hengyuan-hu/bottom-up-attention-vqa)의 코드 일부를 차용 또는 변형하였음을 알려드립니다. 해당 코드를 사용할 수 있게 허락해주셔서 감사드립니다.


### 미리 준비할 사항

타이탄 급 그래픽카드, 64기가 CPU 메모리가 장착된 서버 또는 워크스테이션이 필요합니다. `Python3` 기반의 `PyTorch v1.1.0`가 필요하며 [이 도커 이미지](https://hub.docker.com/layers/pytorch/pytorch/1.1.0-cuda10.0-cudnn7.5-runtime/images/sha256-299bfb9e54db1b2640d59caa6b7432a2b63002ec00154fd9dca4a08796a5f54a)를 사용하실 것을 강력히 추천드립니다.

```bash
pip install -r requirements.txt
```

mecab 설치를 위해서 다음 명령어를 실행하십시오.
```bash
sudo apt-get install default-jre curl
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

### KVQA 데이터셋 내려받기

KVQA 데이터셋은 [이 링크](https://drive.google.com/drive/folders/1hqnCxlWq5JAxnj_wsXjteH0UFhS7RMHW?usp=sharing)를 이용하여 내려받으실 수 있습니다. 별도 라이센스(Korean VQA License)가 적용되므로 유의하시기 바랍니다.

### 전처리

이 구현은 [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)에서 추출된 미리 학습된 이미지 특징을 사용합니다. 이미지 한 장 당 10개에서 100개의 가변적인 개수의 객체에 대한 이미지 특징들을 미리 구할 수 있습니다. 한글 단어 벡터를 위해서 다음의 코드 저장소를 참고하여 주십시오: [Word2vec](https://github.com/Kyubyong/wordvectors), [GloVe](https://ratsgo.github.io/embedding), [fastText](https://github.com/Kyubyong/wordvectors), 그리고 [BERT](https://github.com/google-research/bert). 

#### 다음 과정을 따르면 데이터를 쉽게 준비할 수 있습니다.

1. [KVQA 데이터셋 내려받기](#kvqa-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EB%82%B4%EB%A0%A4%EB%B0%9B%EA%B8%B0)에서 다운받은 데이터의 경로를 아래와 같이 지정해주세요.

```bash
data
├── KVQA_annotations.json
└── features
    ├── KVQA_resnet101_faster_rcnn_genome.tsv
    └── VizWiz_resnet101_faster_rcnn_genome.tsv
```
전처리된 이미지 특징 파일(tsv)들을 다운받으시면 이미지 파일들을 다운 받으실 필요없이 학습을 진행할 수 있습니다.

2. `download.sh`와 `process.sh` 스크립트를 실행해주세요.

```bash
./tools/download.sh
./tools/process.sh
```


### 학습하기 

학습을 시작하기 위해서 다음 명령을 실행하십시오.

```bash
python3 main.py
```

매 학습 주기마다 학습 점수와 검증 점수를 확인하실 수 있습니다. 가장 좋은 모델은 `saved_models` 디렉토리 아래 저장될 것입니다. 만약 다른 질의 임베딩을 이용하여 학습하고자 한다면 다음 명령어를 실행하십시오.

```bash
python3 main.py --q_emb glove-rg
```


### 논문 인용

연구 목적으로 이 코드 저장소의 일부를 사용하신다면 다음 논문들을 인용해주시면 감사하겠습니다.

```
@inproceedings{Kim_Lim2019,
author = {Kim, Jin-hwa and Lim, Soohyun and Park, Jaesun and Cho, Hansu},
booktitle = {AI for Social Good workshop at NeurIPS},
title = {{Korean Localization of Visual Question Answering for Blind People}},
year = {2019}
}
@inproceedings{Kim2018,
author = {Kim, Jin-Hwa and Jun, Jaehyun and Zhang, Byoung-Tak},
booktitle = {Advances in Neural Information Processing Systems 31},
title = {{Bilinear Attention Networks}},
pages = {1571--1581},
year = {2018}
}
```

### 라이센스

* Korean VQA License for the KVQA Dataset
* Creative Commons License Deed ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko)) for the VizWiz subset
* GNU GPL v3.0 for the Code

### 감사의 글

데이터 수집에 도움을 주신 [테스트웍스](http://www.testworks.co.kr/page/overview) 관계자 분들께 감사의 말씀을 드립니다.

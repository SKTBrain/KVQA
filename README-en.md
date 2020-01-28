Back to [Korean](README.md).

SK Telecom pursues a goal of adding social value to our society through various projects. We believe that the first step for sustainable development in management is to take the lead and discover social issues, and then offer responsible solutions.

Since April 2019, we initiated a project in cooperation with [Testworks](http://www.testworks.co.kr), a social enterprise in Korea, and started to collect data from the blind people who volunteered to participate in this project. Furthermore, we translated parts of the published [VizWiz dataset](https://vizwiz.org/tasks-and-datasets/vqa/) which can be well fit into the Korean context, and create a full dataset to train Visual Question Answering (VQA) models in Korean language.

# Paper

## AI for Social Good workshop at NeurIPS (Kim & Lim et al., 2019)

[PDF](https://drive.google.com/open?id=1CB9DwXTI1UqpsCEN9a5V9yOLS_T_HTu_)
![AI for Social Good workshop at NeurIPS](docs/assets/img/AISG_NeurIPS_2019_KVQA.png)

# Visual question answering

VQA understands a provided image and if a person asks question about this, it provides an answer after analyzing (or reasoning) the image via natural language. 

![VQA](docs/assets/img/vqa.png)

# KVQA dataset

As part of T-Brain’s projects on social value, KVQA dataset, a Korean version of VQA dataset was created. KVQA dataset consists of photos taken by Korean visually impaired people, questions about the photos, and 10 answers from 10 distinct annotators for each question.
Currently, it consists of 30,000 sets of images and questions, and 300,000 answers, but by the end of this year, we will increase the dataset size to 100,000 sets of images and questions, and 1 million answers.
This dataset can be used only for educational and research purposes. Please refer to the attached license for more details. We hope that the KVQA dataset can simultaneously provide opportunities for the development of Korean VQA technology as well as creation of meaningful social value in Korean society.

You can download KVQA dataset via [this link](https://drive.google.com/drive/folders/1IQazOJtNTBql51woveN4zb6NplxH7eVl?usp=sharing).

![Examples of KVQA](docs/assets/img/kvqa_examples.png)

## Data statistics

### v1.0 (Jan. 2020)

|             | Overall (%)    | Yes/no (%)   | Number (%)   | Etc (%)        | Unanswerable (%) |
|:------------|:---------------|:-------------|:-------------|:---------------|:-----------------|
| # images    | 100,445 (100)  | 6,124 (6.10) | 9,332 (9.29) | 69,069 (68.76) | 15,920 (15.85)   |
| # questions | 100,445 (100)  | 6,124 (6.10) | 9,332 (9.29) | 69,069 (68.76) | 15,920 (15.85)   |
| # answers   | 1,004,450 (100)| 61,240 (6.10)| 93,320 (9.29)| 690,690 (68.76)| 159,200 (15.85)  |

## Evaluation

We measure the model's accuracy by using answers collected from 10 different people for each question. If the answer provided by a VQA model is equal to 3 or more answers from 10 annotators, it gets 100%; if less than 3, it gets a partial score proportionately. To be consistent with ‘human accuracies’, measured accuracies are averaged over all 10 choose 9 sets of human annotators. Please refer to [VQA Evaluation](https://visualqa.org/evaluation.html) which we follow.

## Data

### Data field description

| Name                             | Type     | Description                                                    |
|:---------------------------------|:---------|:---------------------------------------------------------------|
| VQA                              | `[dict]` | `list` of `dict` holding VQA data                              |
| +- image                         | `str`    | filename of image                                              |
| +- source                        | `str`    | data source `["kvqa" | "vizwiz"]`                              |
| +- answers                       | `[dict]` | `list` of `dict` holding 10 answers                            |
| +--- answer                      | `str`    | answer in `string`                                             |
| +--- answer_confidence           | `str`    | `["yes" | "maybe" | "no"]`                                     |
| +- question                      | `str`    | question about the image                                       |
| +- answerable                    | `int`    | answerable? `[0 | 1]`                                          |
| +- answer_type                   | `str`    | answer type `["number" | "yes/no" | "unanswerable" | "other"]` |

### Data example

```json
[{
        "image": "KVQA_190712_00143.jpg",
        "source": "kvqa",
        "answers": [{
            "answer": "피아노",
            "answer_confidence": "yes"
        }, {
            "answer": "피아노",
            "answer_confidence": "yes"
        }, {
            "answer": "피아노 치고있다",
            "answer_confidence": "maybe"
        }, {
            "answer": "unanswerable",
            "answer_confidence": "maybe"
        }, {
            "answer": "게임",
            "answer_confidence": "maybe"
        }, {
            "answer": "피아노 앞에서 무언가를 보고 있음",
            "answer_confidence": "maybe"
        }, {
            "answer": "피아노치고있어",
            "answer_confidence": "maybe"
        }, {
            "answer": "피아노치고있어요",
            "answer_confidence": "maybe"
        }, {
            "answer": "피아노 연주",
            "answer_confidence": "maybe"
        }, {
            "answer": "피아노 치기",
            "answer_confidence": "yes"
        }],
        "question": "방에 있는 사람은 지금 뭘하고 있지?",
        "answerable": 1,
        "answer_type": "other"
    },
    {
        "image": "VizWiz_train_000000008148.jpg",
        "source": "vizwiz",
        "answers": [{
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "티비 리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "maybe"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }, {
            "answer": "리모컨",
            "answer_confidence": "yes"
        }],
        "question": "이것은 무엇인가요?",
        "answerable": 1,
        "answer_type": "other"
    }
]
```

# Licenses

* [Korean VQA License](LICENSE) for the KVQA Dataset
* Creative Commons License Deed ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko)) for the VizWiz subset
* GNU GPL v3.0 for the Code

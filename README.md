[Adversarial Personalized Ranking for Recommendation](https://dl.acm.org/doi/abs/10.1145/3209978.3209981) -PyTorch
====================================================

The repository implement the [Adversarial Personalized Ranking for Recommendation](https://dl.acm.org/doi/abs/10.1145/3209978.3209981) with PyTorch.

I am still in the testing phase of the data set. If there is any problem with the code, please contact me. 🌏 🪐☄️ 

## Environment

* python==3.6
* pytorch==1.3.1

You can install these package by executing the following command or through anaconda.

```bash
pip install -r requirements.txt
```



## Usage

### 1. Preprocess data

In order to better compare with the experimental effect of the original author, this repository adopted the processed datasets provided in the authors' source code. The processed dataset are:

* MovieLens 1M
* Yelp
* Pinterest

Execute following command line to preprocess the data.

```bash
python3.6 preprocess.py --dataset ml-1m --output_data preprocessed/ml-1m.pickle
python3.6 preprocess.py --dataset yelp --output_data preprocessed/yelp.pickle
python3.6 preprocess.py --dataset pinterest --output_data preprocessed/pinterest.pickle
```

### 2. Training AMF

```bash
python3.6 train.py --data preprocessed/ml-1m.pickle 
python3.6 train.py --data preprocessed/yelp.pickle
python3.6 train.py --data preprocessed/pinterest.pickle 
```

### 3.Evaluation

The result was evaluated by Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG).

<img src="https://p6-tt-ipv6.byteimg.com/origin/pgc-image/ece30982c0e4430f86a9fdc328d46535" width="60%" height="60%" />

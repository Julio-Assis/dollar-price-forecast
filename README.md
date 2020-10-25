# dollar-price-forecast

The focus of this repository is to implement different machine learning algorithms that have access to the timeseries data of different derivatives and try to predict the price of the dollar in D+1. Also, we want to evaluate these different models against each other by using relevant validation metrics on the timeseries forecast problem.

# setting up development environment

To install the packages required for running code in this repository you'll need to install the packages in `requirements.txt`. We suggest using a `virtualenv` for that. First, install  `virtualenv` globally with.

```sh
sudo apt install virtualenv
```

Then from the root of the repository, create the virtualenv with

```sh
virtualenv -p python3 virtualenv
```

Activate the environment with

```sh
source virtualenv/bin/activate
```

Finally, install the packages in `requirements.txt` with

```sh
pip install -r requirements.txt
```

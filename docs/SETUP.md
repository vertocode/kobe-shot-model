## Setup

0. Set up the virtual environment (Optional, but recommended)

First, let’s create a virtual environment. I will use pyenv, but feel free to use any other tool you prefer.

- macOS/Linux:
```shell
pyenv virtualenv 3.11.6 shotmodel
```

> If you encounter the error: pyenv: no such command 'virtualenv', you need to install pyenv-virtualenv. To install it on macOS, run: `brew install pyenv-virtualenv`

I used the name “shotmodel” for the environment, but you can choose any name you prefer.

Make sure you have created the environment using Python 3.11.6.

Now, activate the newly created virtual environment by running the following command:

- macOS/Linux:
```shell
pyenv activate shotmodel
```

> Note: The command may vary depending on the name you chose for your virtual environment.


1. Install dependencies

```shell
pip install -r requirements.txt
```

2. Run kedro

```shell
python -m kedro run
```
## Setup

0. Set up the virtual environment (Optional, but recommended)

First, let’s create a virtual environment. I will use venv, but feel free to use any other tool you prefer.

- macOS/Linux:
```shell
python3 -m venv shotmodel
```

- Windows:
```shell
python -m venv shotmodel
```

I used the name “shotmodel” for the environment, but you can choose any name you prefer.

Now, activate the newly created virtual environment by running the following command:

- macOS/Linux:
```shell
source shotmodel/bin/activate
```

- Windows: 
```shell
shotmodel\Scripts\activate
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
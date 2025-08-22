# Deep Learning with PyTorch, 2nd Edition

This repository contains code for the book Deep Learning with PyTorch, Second Edition by Howard Huang, Eli Stevens, Luca Antiga, and Thomas Viehmann, published by Manning Publications.

The Manning site for the book is: https://www.manning.com/books/deep-learning-with-pytorch-second-edition

The errata for the book can be found on the Manning website.

## About Deep Learning with PyTorch

This book has the aim of providing the foundations of deep learning with PyTorch and
showing them in action in a real-life project. We strive to provide the key concepts underlying deep learning and show how PyTorch puts them in the hands of practitioners. In
the book, we try to provide intuition that will support further exploration, and in doing
so we selectively delve into details to show what is going on behind the curtain.
Deep Learning with PyTorch doesn’t try to be a reference book; rather, it’s a conceptual companion that will allow you to independently explore more advanced material
online. As such, we focus on a subset of the features offered by PyTorch.

## Who should read this book

This book is meant for developers who are or aim to become deep learning practitioners and who want to get acquainted with PyTorch. We imagine our typical reader
to be a computer scientist, data scientist, or software engineer, or an undergraduate or later student in a related program. Since we don’t assume prior knowledge of deep
learning, some parts in the first half of the book may be a repetition of concepts that
are already known to experienced practitioners. For those readers, we hope the exposition will provide a slightly different angle to known topics.
 We expect readers to have basic knowledge of imperative and object-oriented programming. Since the book uses Python, you should be familiar with the syntax and
operating environment. Knowing how to install Python packages and run scripts on
your platform of choice is a prerequisite. Readers coming from C++, Java, JavaScript,
Ruby, or other such languages should have an easy time picking it up but will need to
do some catch-up outside this book. Similarly, being familiar with NumPy will be useful. We also expect familiarity with some basic linear algebra,
such as knowing what matrices and vectors are and what a dot product is.

## Setting Up the Project Environment with Poetry or pip

This project uses [Poetry](https://python-poetry.org/) for Python dependency management and environment setup, but you can also use pip with a requirements.txt file. Follow these steps to get started:

### Option 1: Using Poetry (Recommended)

#### 1. Install Poetry

If you don't have Poetry installed, run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or see the [Poetry installation guide](https://python-poetry.org/docs/#installation) for more options.

#### 2. Install Project Dependencies (Except PyTorch)

In the project root (where `pyproject.toml` is located), run:

```bash
poetry install
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml` (except PyTorch).

#### 3. Install PyTorch, TorchVision, and TorchAudio

PyTorch and related libraries are not included in the default dependencies. Please follow the official instructions for your system and hardware:

- Go to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/)
- Select your OS, package manager (pip), Python version, and compute platform (CPU or CUDA)
- **If you have a GPU and want CUDA acceleration, be sure to select the CUDA option for your version.**
- Copy the provided install command and run it inside your Poetry shell:

```bash
poetry shell
# Then paste the command from the PyTorch website, e.g.:
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For CUDA (GPU) support, use the CUDA command provided by the PyTorch website, e.g.:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Using pip and requirements.txt (Alternative)

#### 1. Install dependencies with pip

In the project root, run:

```bash
pip install -r requirements.txt
```

#### 2. Install PyTorch, TorchVision, and TorchAudio

As above, follow the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) and install the correct versions for your system (CPU or CUDA) with pip.

---

For more information, see the [Poetry documentation](https://python-poetry.org/docs/) and [PyTorch installation guide](https://pytorch.org/get-started/locally/).

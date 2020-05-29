# Project 1. Board Games

This folder contains the notebook, input and output files for the first project of the Udacity Data Scientist NanoDegree program (Term 2).

## Files

The key notebook file is `Board_Games.ipynb`, which looks at the characterisics of a large set of board-games.  We explore the relationship between games' age-ranges and play-times, as well as attempting to predict the games rating.  We eventually reach an R2 value (for linear regression) of just over 0.3.

It uses a data set from [Kaggle](https://www.kaggle.com/mshepherd/board-games), stored in the `BGG.csv` file.  This dataset was created by Markus Shepherd.

It also requires two json dictionaries, `categories.json` & `mechanics.json`, as certain features are provided in a non-human-readable way.  This were copied from the BoardGameAtlas [docs](https://www.boardgameatlas.com/api/docs/game/mechanics).

There are several graphical outputs saved here, which are used in the [Medium Post](https://medium.com/p/c31340859bef) describing this work.

## Requirements

The notebook runs in a python3 [Jupyter Kernel](https://jupyter.org/install)

Required python libraries to run the notebook:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import plotly.express as px
import json
```

# numlp
## _Neural Network - Numerical Prediction application_

[![N|Solid](https://pypi.org/static/images/logo-small.8998e9d1.svg)](https://www.python.org/)

[![Generic badge](https://img.shields.io/badge/version-v1.1.00-<>.svg)](https://shields.io/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
## Introduction
Do you have an excel file with header labeled, otherwise numerical data?
|MONTH|DAY|AVGTEMP|MAXTEMP|
|--|--|--|--|
|01|01|09|12|
|01|02|11|14|
|--|--|--|--|
|10|21|18|19|

Would you like to predict **ANY**✨ column value with an AI neural network model?
|MONTH|DAY|AVGTEMP|MAXTEMP|
|--|--|--|--|
|12|08|?|08|

This application provides an accessible, easy to understand and click-through interface to make that happen.

> If you'd like to tinker around with some advanced Neural Network model
> attributes, such as epochs, hidden layers, optimizers, increased precision..

You can!

> Want to keep track of the built model parameters?

The sidebar display all selected, all built attributes.

> Would you like to download the created model for later use?

No problem. Model files can be downloaded in .pt format.

## Features

- Configurable Neural Network: Customize activation functions, optimizers, hidden layers, and training epochs.
- User-Friendly Interface: Built with Streamlit for easy interaction.
- Hardware Agnostic: Automatically uses CPU or GPU.
- Advanced Tuning: Fine-tune with options like dropout rate and batch size.
- Data Flexibility: Processes numerical data from XLSX files, automatically mapping inputs to outputs.
- Robust Evaluation: Model performance assessed on a 30% test set.
- Uncertainty in Predictions: "Predict Range" uses Monte Carlo Dropout for mean predictions and 95% confidence intervals.
- Persistent Session: Model parameters maintained until dataset change or reload.
- Key Performance Metrics: Displays RMSE, MAE, R², and visual graphs of predictions and training loss.
## How to use
1. Prepare an excel (xlsx) file with <row1> labels and <row2,3,...> with only numerical values
2. Upload data via the GUI sidebar - upload button
3. Model will automatically analyze excel file and display header labels
4. Select the output from the list (y1) [all other labels will be taken as x1, x2, x3..]
5. OPTIONAL: Select 'Advanced Options' to fine-tune mode parameters
6. Train model
7. Examine precision metrics
8. Specify new input value and predict a new output value
9. OPTIONAL: Download model .pt file


## Cloud

numlp is available online, check it out!

| Service   | URL                |
|-----------|--------------------|
| Streamlit | [NUMLPGUI][gcloud] |


## Installation

numlp uses several python packages and based on [numpy][numpyver] v1.26.4.
Additional packages are included in the requirements.txt

For local run git clone and step into the cloned repository:

```sh
git clone https://github.com/th3pajay/numlp
cd numlp
```

Start application:

```sh
streamlit run app.py
```
or alternatively:
```sh
python -m streamlit run app.py
```
Local GUI now accessible on localhost:8501


## License

MIT

**Free Software, Hell Yeah!**

   [streamlitgui]: <https://streamlit.io/>
   [gcloud]: <https://numlpgui.streamlit.app/>
   [numpyver]: <https://numpy.org/devdocs/release/1.26.4-notes.html>


[![N|Solid](https://user-images.githubusercontent.com/74038190/216649421-9e9387cc-b2d3-4375-97e2-f4c43373d3ae.gif)](https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub)
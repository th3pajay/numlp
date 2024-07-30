# numlp
## _Neural Network - Numerical Prediction application_

[![N|Solid](https://pypi.org/static/images/logo-small.8998e9d1.svg)](https://www.python.org/)

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
## Introduction
Do you have an excel file with labeled, numerical data?
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
> attributes, such as epochs or hidden layers, increased precision..

You can!

> Want to keep track of the model built?

The sidebar will note and display the running model parameters.

> Would you like to download the created model for later use?

No problem.

## Features

- The neural network model offers selectable activation, hidden layer and train loop count (epoch) on numerical data inputs from xlsx file format
- User interaction via [streamlit][streamlitgui] GUI
- By default uses ReLU activation function, 3 hidden layers and 100 epochs to train and make a baseline comparison on trained (70%), with test data (30%_true) against predicted test data (30%_pred)
- Model -on data set load- automatically derives the <row1> labels, on which the user can range specify the N:1 relation of input (x1, x2, x3..) and output value (y1)
- Model parameter selection is fixed for session, until data set or page is re-loaded
- Training establishes correlation between input values and with activation and weights shifted feeds forward towards precision metrics at the end of epoch run
- Metrics are displayed along with visualized graph of predicted test data against true test data values
## How to use
1. Prepare an excel (xlsx) file with <row1> labels and <row2,3,...> only numerical values
2. Upload data set via the GUI
3. Determine x values (input) and y value (output)
4. Specify advanced neural network parameters if needed
5. Train model
6. Examine precision metrics
7. Specify new input value and predict a new output value


## Cloud

numlp will be linked to Google Cloud to provide an *out-of-the-box*, free service.

| Service | URL |
| ------ | ------ |
| Google Cloud | [TBD][gcloud] |


## Installation

numlp uses several python packages and [numpy][numpyver] v1.26.4.

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
Will automatically open webpage with default steramlit port: 8501


## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [streamlitgui]: <https://streamlit.io/>
   [gcloud]: <UPDATE>
   [numpyver]: <https://numpy.org/devdocs/release/1.26.4-notes.html>


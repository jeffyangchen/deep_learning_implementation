import numpy as np
import progressbar
import plotly
from plotly import graph_objs as go

bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

def batch_iterator(X,y = None,batch_size = 64):
	"""Generator function that divides data into batches for mini-batch SGD"""
	points = X.shape[0]
	for i in np.arange(0,points,batch_size):
		begin,end = i,min(i+batch_size,points)
		if y is not None:
			yield X[begin:end],y[begin:end]
		else:
			yield X[begin:end]

def unwind_data(data):
	"""
	Unwinds tuple data into two vectors X,y
	:param data:
	:return:
	"""
	X = []
	y = []
	for row in data:
		X.append(row[0])
		y.append(row[1])

	return np.asarray(X),np.asarray(y)

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def plotter(x,Y,labels,title = None,filename = 'plot.html'):
	data = []
	for row,name in zip(Y,labels):
		trace = go.Scatter(
			x = x,
			y = row,
			name = name,
			mode = 'lines+markers'
		)
		data.append(trace)
	layout = go.Layout(
		title = title
	)
	fig = go.Figure(data = data,layout=layout)
	plotly.offline.plot(fig, filename=filename)
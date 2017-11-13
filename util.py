import numpy as np
import progressbar
import plotly

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

def plotter(x,Y,title = None,filename = None):
	data = []
	for row in Y:
		trace = go.Scatter(
			x = x,
			y = row[0],
			name = row[1],
			mode = 'lines+markers'
		)
		data.append(trace)
	layout = go.Layout(
		title = title
	)
	fig = go.Figure(data = data,layout=layout)
	plotly.offline.plot(fig, filename=filename)
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column


x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend="Temp.", line_width=2)


page_layout = column(p)

curdoc().add_root(p)

curdoc().title = "tester"

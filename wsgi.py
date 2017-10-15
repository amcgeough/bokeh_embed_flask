import os
from app import app as application
import subprocess


if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('localhost',  application)
    httpd.serve_forever()

#subprocess.Popen(
#    ['bokeh', 'serve','--allow-websocket-origin=127.0.0.1:5000', '--port=5000','bokeh_plot.py'], stdout=subprocess.PIPE)
#    ['bokeh', 'serve','bokeh_plot.py'], stdout=subprocess.PIPE)

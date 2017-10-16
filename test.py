import subprocess
import atexit

subprocess.Popen(["bokeh", "serve", "--allow-websocket-origin=127.0.0.1:8080", "--port=8080", "--use-xheaders", "bokeh_plot.py"],stdout=subprocess.PIPE)

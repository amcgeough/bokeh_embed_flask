#flask_app.py
import subprocess
import atexit
from flask import render_template, render_template_string, Flask
from bokeh.embed import autoload_server
from bokeh.client import pull_session

app_html="""
<!DOCTYPE html>
<html lang="en">
  <body>
    <div class="bk-root">
      {{ bokeh_script|safe }}
    </div>
  </body>
</html>
"""

app = Flask(__name__)

bokeh_process = subprocess.Popen(["bokeh", "serve", "--allow-websocket-origin=bokeh-flask-bokeh-flask.a3c1.starter-us-west-1.openshiftapps","--port=$PORT", "--host=bokeh-flask-bokeh-flask.a3c1.starter-us-west-1.openshiftapps.com", "--host=*", "--address=0.0.0.0", "--use-xheaders", "bokeh_plot.py"],stdout=subprocess.PIPE)

"""
subprocess.Popen(
#    ['bokeh', 'serve','--allow-websocket-origin=127.0.0.1:5000', '--port=5000','bokeh_plot.py'], stdout=subprocess.PIPE)
    #['bokeh', 'serve','bokeh_plot.py'], stdout=subprocess.PIPE)
    ['bokeh-server', '--port 8080', '--ip 127.0.0.1', '--backend memory']
"""

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route('/')
def index():
    session=pull_session(app_path='/bokeh_plot')
    bokeh_script=autoload_server(None,app_path="/bokeh_plot",session_id=session.id)
    # return render_template('app.html', bokeh_script=bokeh_script)
    return render_template_string(app_html, bokeh_script=bokeh_script)

if __name__ == '__main__':
    app.run(debug=True)

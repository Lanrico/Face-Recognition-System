from recognition import app

import socketio
from waitress import serve
import socket
import eventlet.wsgi

hostname=socket.gethostname()   
IPAddr=socket.gethostbyname(hostname)

sio = socketio.Server()
appServer = socketio.WSGIApp(sio, app)

if __name__ == '__main__':

    print("Face recognition system server start, please access http://127.0.0.1:8080 or http://"+IPAddr+":8080")
    serve(appServer, host='0.0.0.0', port=8080, url_scheme='http', threads=3)
    # eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 8080)), appServer)


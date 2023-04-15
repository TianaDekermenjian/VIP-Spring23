import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('start_recording')
def my_response():
    print('Started recording')

if __name__ == '__main__':
    sio.connect('http://192.168.100.61:5000')



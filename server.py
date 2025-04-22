import socket
s = socket.socket()
port = 8282
s.bind(('',port))
print(f'socket binded to port {port}')
s.listen(5)
while True:
    c, addr = s.accept()
    print(f"got connection from {addr}")
    message = "connected"
    c.send(message.encode())
    c.close()
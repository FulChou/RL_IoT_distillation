import socket
import time 

#创建套接字
tcpClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket---%s'%tcpClientSocket)
#链接服务器
serverAddr = ('127.0.0.1', 5005)
tcpClientSocket.connect(serverAddr)
print('connect success!')
# matlab_config = ['log_path', 1e3, 0.9, 200]




def set_data(data):
    # 发送数据
    input = data.obs[:, -1]
    input = input.tolist()
    input_s = str(input) + '\n'
    # sendData = 'please input the send message:'
    # s_t = time.perf_counter()
    if len(input_s)>0:
        # print(len(input_s.encode('ascii')), type(input_s.encode()))
        tcpClientSocket.send(input_s.encode('ascii'))
    # e_t = time.perf_counter() - s_t
    # print('sent:', e_t)

    # 接收数据
    recvData = tcpClientSocket.recv(102400)
    # 打印接收到的数据
    # print(recvData)
    recvData = recvData.decode('utf8')
    # print('the receive message is: %s'% recvData, type(recvData))
    return recvData.split()

# 关闭套接字
# tcpClientSocket.close()
# print('close socket!')
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/3 15:59 
import struct
import pickle
import socket


def recv_msg(sock):
    # 读取消息长度并将其解压缩为整数
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # 读取消息数据
    return recvall(sock, msglen)


def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def decode_msg(msg):
    res = pickle.loads(msg)
    return res


def encode_msg(data):
    msg = pickle.dumps(data)
    return msg


class clientCommunication():
    """客户端通信"""
    def __init__(self, host_self, port_self):
        self.host_self = host_self
        self.port_self = port_self
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host_self, self.port_self))
        self.s.listen()

    def send_msg(self, msg, host_conn, port_conn):
        msg = encode_msg(msg)
        self.s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s1.connect((host_conn, port_conn))
        send_msg(self.s1, msg)

    def receive_msg(self, conn):
        received = recv_msg(conn)
        received = decode_msg(received)
        return received

    def accept_conn(self):
        conn, addr = self.s.accept()
        return conn, addr

    def close_channel(self):
        self.s.close()


class cloudCommunication():
    """云通信"""
    def __init__(self, host_self, port_self):
        self.host_self = host_self
        self.port_self = port_self
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host_self, self.port_self))
        self.s.listen()

    def send_msg(self, msg, host_conn, port_conn):
        msg = encode_msg(msg)
        self.s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s1.connect((host_conn, port_conn))
        send_msg(self.s1, msg)

    def receive_msg(self, conn):
        received = recv_msg(conn)
        received = decode_msg(received)
        return received

    def accept_conn(self):
        conn, addr = self.s.accept()
        return conn, addr

    def close_channel(self):
        self.s.close()


class edgeCommunication():
    """边缘通信"""
    def __init__(self, host_self, port_self):
        self.host_self = host_self
        self.port_self = port_self
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host_self, self.port_self))
        self.s.listen()

    def send_msg(self, msg, host_conn, port_conn):
        msg = encode_msg(msg)
        self.s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s1.connect((host_conn, port_conn))
        send_msg(self.s1, msg)

    def receive_msg(self, conn):
        received = recv_msg(conn)
        received = decode_msg(received)
        return received

    def accept_conn(self):
        conn, addr = self.s.accept()
        return conn, addr

    def close_channel(self):
        self.s.close()

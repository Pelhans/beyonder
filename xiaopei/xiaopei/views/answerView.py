# -*- coding: utf-8 -*-

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse
from ..models import Question
from ..models import Question_audio
import os
from datetime import datetime
import requests
import json
import socket
import struct

serverIp = '0.0.0.0'; #问答模块网络ip地址
serverPort = 9999; #问答模块网络端口
addr = (serverIp, serverPort);

def answer(question):
#    addr = (serverIp, serverPort);
    res = requests.get("http://el.pelhans.com/api/entity_annotation?txt={}".format(question)).text
    res = json.dumps(eval(res)["data"]).decode('unicode_escape')
#    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
#    sock.connect(addr);
    print("Watting for response.....")
#    send = question;
#    sock.send(send);  # 发送信息
#    answer = sock.recv(1024)
    return res

@api_view(['POST'])
def deal_question( request ):
    result = {}
    requestData = json.loads(request.body)
    question = requestData['talkwords']
    result["message"] = answer(question)
    with open("./logs/log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt", "a+") as l:
        l.write("Current time:\t{};\nRequest is:\t{};\nResult is: {}".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"), question, result["message"]))

    result["statusCode"] = 200
    return Response(result,status=status.HTTP_200_OK)


# -*- coding: utf-8 -*-
 
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from datetime import datetime
from django.core import serializers
from .models import User, Meeting
from django.http import FileResponse
import socket
import struct
import io
import json

def getMeetingObject(meeting):
	objectMeeting = {};
	objectMeeting['id'] = meeting.id;
	objectMeeting['meetingName'] = meeting.meetingName;
	objectMeeting['description'] = meeting.description;
	objectMeeting['resultType'] = meeting.resultType;
	objectMeeting['createDate'] = meeting.createDate;
	objectMeeting['finishDate'] = meeting.finishDate;
	objectMeeting['userName'] = meeting.userName;
	objectMeeting['status'] = meeting.status;
	objectMeeting['language'] = meeting.language;
	objectMeeting['audioType'] = meeting.audioType;
	objectMeeting['audioBit'] = meeting.audioBit;
	objectMeeting['resultUrl'] = meeting.resultUrl;
	objectMeeting['audioUrl'] = meeting.audioUrl;
	objectMeeting['audioFrequency'] = meeting.audioFrequency;
	return objectMeeting;

@api_view(['POST'])
def openCreateMeeting(request):
	result = {};
	requestData = json.loads(request.body);
	userName = requestData['userName'];
	password = requestData['password'];
	meetingName = requestData['meetingName'];
	description = requestData['description'];
	resultType = requestData['resultType'];
	statusCode = judgePermission(userName, password);
	if (statusCode != 200):
		result['statusCode'] = statusCode;
		result["message"] = "error";
		return Response(result,status=status.HTTP_200_OK);
	if (resultType != "txt"):
		result['statusCode'] = 400;
		result["message"] = "error resultType";
		return Response(result,status=status.HTTP_200_OK);
	meeting = saveMeeting(userName, meetingName, description, resultType);
	result['meeting'] = getMeetingObject(meeting);
	result['statusCode'] = 200;
	result["message"] = "ok";
	return Response(result,status=status.HTTP_200_OK);

@api_view(['POST'])
def openGetAllMeetings(request):
	result = {};
	requestData = json.loads(request.body);
	userName = requestData['userName'];
	password = requestData['password'];
	statusCode = judgePermission(userName, password);
	if (statusCode != 200):
		result['statusCode'] = statusCode;
		result["message"] = "error";
		return Response(result,status=status.HTTP_200_OK);
	meetings = Meeting.objects.all();
	meetingList = [];
	for meeting in meetings:
		meetingList.append(getMeetingObject(meeting));
	result["meetings"] = meetingList;
	result['statusCode'] = 200;
	result["message"] = "ok";
	return Response(result,status=status.HTTP_200_OK);

@api_view(['POST'])
def openGetUserAllMeetings(request):
	result = {};
	requestData = json.loads(request.body);
	userName = requestData['userName'];
	password = requestData['password'];
	statusCode = judgePermission(userName, password);
	if (statusCode != 200):
		result['statusCode'] = statusCode;
		result["message"] = "error";
		return Response(result,status=status.HTTP_200_OK);
	meetings = Meeting.objects.filter(userName=userName);
	meetingList = [];
	for meeting in meetings:
		meetingList.append(getMeetingObject(meeting));
	result["meetings"] = meetingList;
	result['statusCode'] = 200;
	result["message"] = "ok";
	return Response(result,status=status.HTTP_200_OK);

@api_view(['POST'])
def openDeleteMeetings(request):
	result = {};
	requestData = json.loads(request.body);
	userName = requestData['userName'];
	password = requestData['password'];
	statusCode = judgePermission(userName, password);
	if (statusCode != 200):
		result['statusCode'] = statusCode;
		result["message"] = "error";
		return Response(result,status=status.HTTP_200_OK);
	meetings = Meeting.objects.all();
	meetingList = [];
	for meeting in meetings:
		meetingList.append(getMeetingObject(meeting));
	result["meetings"] = meetingList;
	result['statusCode'] = 200;
	result["message"] = "ok";
	return Response(result,status=status.HTTP_200_OK);

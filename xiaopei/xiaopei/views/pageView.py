# -*- coding: utf-8 -*-
 
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ..models import Question_audio

def index(request):
	data = {};
	message = request.session.get('message', None);
	request.session["message"]="ok";
	if message is None:
		data["message"] = "ok";
	data["message"] = message;
	return render(request, 'answer.html', data)

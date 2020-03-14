from django.db import models
import django.utils.timezone as timezone

class User(models.Model):
    id = models.AutoField(primary_key=True) 
    userName = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    position = models.CharField(max_length=50)
    email = models.CharField(max_length=50)

class Meeting(models.Model):
    id = models.AutoField(primary_key=True) 
    meetingName = models.CharField(max_length=50)
    description = models.CharField(max_length=50)
    resultType = models.CharField(max_length=50)
    language = models.CharField(max_length=50)
    audioType = models.CharField(max_length=50)
    createDate = models.DateTimeField(default = timezone.now)
    finishDate = models.DateTimeField(auto_now = True)
    audioBit = models.IntegerField()
    audioFrequency = models.IntegerField()
    userName = models.CharField(max_length=50)
    resultUrl = models.CharField(max_length=50)
    audioUrl = models.CharField(max_length=50)
    status = models.CharField(max_length=50)
    txtname = models.CharField(max_length=50)
    wavname = models.CharField(max_length=50)
    txtdate = models.DateTimeField(default = timezone.now)
    wavdate = models.DateTimeField(default = timezone.now)

class Question(models.Model):
    id = models.AutoField(primary_key=True)
    talkWord = models.CharField(max_length = 50)

class Question_audio(models.Model):

    id = models.AutoField(primary_key=True)
    audioUrl = models.CharField(max_length=50)
    wavname = models.CharField(max_length=50,default = "none")

#!/usr/bin/env python3
# coding=utf-8

import logging
import json
import sys
import traceback
import tornado
import tornado.web
import tornado.ioloop
from predict import Client
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-api_port", type=str, default="6000")
parser.add_argument("-model_name", type=str, default="")
args = parser.parse_args()


class ResultBean(object):
    def __init__(self, msg='操作成功', debug_msg='', **data):
        self.msg = msg
        self.debug_msg = debug_msg
        if data:
            self.update(data)

    def __str__(self):
        return json.dumps(vars(self), ensure_ascii=False)

    def update(self, kv):
        for k, v in kv.items():
            setattr(self, k, v)


class UserException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class UnloginException(UserException):
    def __init__(self, msg='使用该功能需要先登录帐号'):
        Exception.__init__(self, msg)


class PermissiomDeniedException(UserException):
    def __init__(self, msg='您的权限不足'):
        Exception.__init__(self, msg)


def getlogger(name, show='WARNING'):
    """
    :param name: 一般填写__name__
    :param show: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    :return: logger
    """
    f = '[%(levelname)1.1s %(asctime)s.%(msecs)03d %(module)s:%(lineno)d] %(message)s'
    formatter = logging.Formatter(f, datefmt='%y%m%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.parent = None
    logger.handlers = []
    showhandler = logging.StreamHandler()
    showhandler.setLevel(show)
    showhandler.setFormatter(formatter)
    logger.addHandler(showhandler)
    return logger


log = getlogger(__name__, 'INFO')
log.info('sys.version = ' + str(sys.version))


def check_param(params, body):
    if params:
        for key in params:
            if key not in body:
                raise UserException('缺少参数：%s' % (key))


def prepare(self, must=None):
    # 预处理获得的客户端请求，返回 body字典（包含所有url参数和body参数）
    method = self.request.method
    request_info = self.request.host + self.request.uri + ' ({})'.format(
        method)
    body = {}
    if method == 'POST':
        if self.request.body_arguments:
            request_info += 'Type=application/x-www-form-urlencoded'
            for k, vs in self.request.body_arguments.items():
                if isinstance(vs, list) and  len(vs) == 1:
                    body[k] = vs[0].decode('utf8')
                else:
                    body[k] = vs
        else:
            request_info += 'Type=application/json'
            body = self.request.body.decode('utf8')
            body = json.loads(body)
    for k, vs in self.request.arguments.items():
        if isinstance(vs, list) and  len(vs) == 1:
            body[k] = vs[0].decode('utf8')
        else:
            body[k] = vs
    if must:
        check_param(must, body)
    log.info(request_info)
    log.info('request_body = ' + json.dumps(body, ensure_ascii=False))
    return body


def write_error(self, status_code, **kwargs):
    url = self.request.host + self.request.uri
    error = '\n'.join(traceback.format_exception(*sys.exc_info()))
    error = 'url={url} ({method}) code={code}\n{error}\n'.format(
        url=url, method=self.request.method, code=status_code, error=error)
    log.error(error)

    self.set_header("Content-Type", "application/json")
    result = ResultBean(msg=str(sys.exc_info()[1]), debug_msg=error)
    if isinstance(sys.exc_info()[1], UnloginException):
        self.set_status(401)
    elif isinstance(sys.exc_info()[1], PermissiomDeniedException):
        self.set_status(403)
    else:
        self.set_status(400)
    self.write(str(result))


client = Client(args.model_name, server_ip=args.model_name,
               server_port=6000,)

# 服务器接口
class Api(tornado.web.RequestHandler):
    def post(self):
        log.info("I RECEIVE A MSG")
        body = prepare(self)
        #########################  调用功能函数  ########################
        start = time.time()
        querys = body["query"].split("|")
        querys_entity = body["query_entity"].split("|")
        tmp_all_candicates = body["candicate"].split("|")
        tmp_all_candicates_entity = body["candicate_entity"].split("|")
        lengths = body["length"].split("|")
        lengths = [int(l) for l in lengths]
        count = 0
        all_candicates = []
        all_candicates_entity = []
        for leni in lengths:
            all_candicates.append(tmp_all_candicates[count:count+leni])
            all_candicates_entity.append(tmp_all_candicates_entity[count:count+leni])
            count += leni
        assert len(querys) == len(all_candicates)
        assert len(querys) == len(all_candicates_entity)
        assert len(querys) == len(querys_entity)
        examples = []
        for idx, query in enumerate(querys):
            for cidx, candicate in enumerate(all_candicates[idx]):
                examples.append(str({"query_text": query,
                                     "query_entity": querys_entity[idx],
                                     "candi_abstract": candicate,
                                     "candi_entity": all_candicates_entity[idx][cidx],
                                     "tag": 0}))
        result = client.query_parsing(examples)
        result = json.loads(result)

        out = ResultBean(data=result)
        log.info("共耗时 {} s".format(time.time()-start))
        log.info('out_bean = {}'.format(out))
        ###############################################################
        sys.stdout.flush()
        self.set_header("Content-Type", "application/json")
        self.write(str(out))


def make_app():
    handlers = [
        (r"/api/kg_rerank", Api),
    ]
    return tornado.web.Application(handlers)


def run_server(port):
    # 定制 tornado的异常处理函数
    tornado.web.RequestHandler.write_error = write_error
    # 构建服务器
    app = make_app()
    app.listen(port)
    log.info('app listen port = {}'.format(port))
    log.info('app ioloop start ...\n')
    tornado.ioloop.IOLoop.current().start()
    log.info('app ioloop finish')
    return app


run_server(port=args.api_port)

#!/bin/bash
curl -s -XGET 'http://47.104.240.64:9919/_cat/indices/kg_baidu?v' | awk -F ' ' {'print $7'} | grep -v docs.count

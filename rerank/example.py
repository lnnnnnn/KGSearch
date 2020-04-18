from rerank import predict
import json

with open('log.txt',encoding='utf-8') as fin:
	item=next(fin)#取第一行做测试
	item=json.loads(item)
	scores=predict(item['results'],item['intention_calc'],item["timestamp"])
	print(scores)
	inds=(-scores).argsort()[:10]
	for i in inds :
		print(item['results'][i].get("DRETITLE"))


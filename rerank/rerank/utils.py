def get_dataset(fn):
	with open(fn,'r',encoding='utf-8') as fin:
		lines=fin.readlines()
		qs={}
		for line in lines:
			eid,query,intention,title,url=line.strip().split('\t')[:5]
			# print(query)
			if query.strip():
				qs.setdefault((eid,query),[]).append((title,url))

	return qs

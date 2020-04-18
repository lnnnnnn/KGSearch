import os, sys, re, math, ljqpy, json
import kw_api as api
from search import GetTags
from collections import defaultdict

api.SetApiKey('')

def GetEdgeFromCNDB():
	nodes = ljqpy.LoadList('saved_graph/graph_nodes.txt')
	node_set = set(nodes)
	step = 100
	ems = {}
	with open('gen_rels/edges_kg.txt', 'w', encoding='utf-8') as fout:
		for ii in range(0, len(nodes), step):
			print('%d/%d' % (ii, len(nodes)))
			nslice = nodes[ii:ii+step]
			m2e = api.Ment2Ent(nslice)
			mes = {}
			for mm in nslice:
				ees = m2e.get(mm, [])
				if len(ees) == 0: continue
				ee = ees[0]
				if '歌' in ee or '影' in ee: continue
				mes[mm] = ee
				ems.setdefault(ee, []).append(mm)
			tris = api.GetEntTriplesMulti(list(mes.values()), keephref=True, nospecial=0)
			edges = []
			for mm, ee in mes.items():
				tri = tris.get(ee, [])
				ww = 0.5 + 0.5 / len(m2e.get(mm, []))
				for p, o in tri:
					olinks = re.findall('<a.+?>(.+?)</a>', o)
					for olink in olinks:
						if olink == mm: continue
						if olink in node_set:
							edges.append(('KG', mm, olink, ww))
							edges.append(('KGi', olink, mm, ww))
			for x in edges: 
				ljqpy.WriteLine(fout, x)
		for ee, mms in ems.items():
			for i, m1 in enumerate(mms):
				for m2 in mms[:i]:
					ljqpy.WriteLine(fout, ['KGm', m1, m2, 1])
					ljqpy.WriteLine(fout, ['KGm', m2, m1, 1])

def GetEdgeFromCoocc():
	global datalist, datadict, idf, tags, r1cnt
	datalist = [];  datadict = {}
	df = defaultdict(int)
	for jj in ljqpy.LoadList('training/all_data.txt'):
		jj = json.loads(jj)
		datadict[jj['id']] = jj['text']
		tf = GetTags(jj['text'])
		for t in tf.keys(): df[t] += 1
		jj['tf'] = tf
		datalist.append(jj)
	N = len(datalist)
	idf = {x:math.log(N/s) for x,s in df.items()}
	#ljqpy.SaveCSV(ljqpy.FreqDict2List(idf), 'saved_graph/idf.txt')
	tags = {x for x,s in df.items() if s > 2 and idf[x] > 2 and len(x) > 1}
	tags = {x for x in tags if not x.isdigit()}

	print('docu segs:', N)
	print('tags:', len(tags))
	lasttts = [] 

	r2cnt = defaultdict(int)
	r1cnt = defaultdict(int)

	for i, jj in enumerate(datalist):
		id, words = jj['id'], jj['tf']
		tt = [x for x in words.keys() if x in tags]
		if i % 1000 == 0: print('datalist %d/%d' % (i, len(datalist)))

		for mi in range(3):
			if mi >= i: continue
			lid, lasttt = (id, tt) if i == 0 else lasttts[-mi]
			if lid.split('@')[0] != id.split('@')[0]: break

			for w1 in tt:
				for w2 in lasttt:
					if w1 in w2 or w2 in w1: continue
					if w2 < w1: w1, w2 = w2, w1
					r2cnt[(w1, w2)] += 1
					r1cnt[w1] += 1
					r1cnt[w2] += 1

		lasttts.append( (id, tt) )
		if len(lasttts) > 10: lasttts = lasttts[5:]

	relscs = {}
	for g2, ng2 in ljqpy.FreqDict2List(r2cnt):
		for i, w in enumerate(g2):
			relscs[(w, g2[1-i])] = ng2 / r1cnt[w]
		#print(g2, ng2, ng2/r1cnt[g2[0]], ng2/r1cnt[g2[1]])
		if ng2 < 100: break

	with open('gen_rels/edges_coocc.txt', 'w', encoding='utf-8') as fout:
		for g2, rel in ljqpy.FreqDict2List(relscs):
			if rel < 0.2: break
			ljqpy.WriteLine(fout, ['coocc', g2[0], g2[1], rel])
			
if __name__ == '__main__':
	if not os.path.isdir('gen_rels'): os.mkdir('gen_rels')
	#GetEdgeFromCNDB()
	GetEdgeFromCoocc()
	ljqpy.MergeFiles('gen_rels', 'saved_graph/graph_edges.txt')
	print('complete')
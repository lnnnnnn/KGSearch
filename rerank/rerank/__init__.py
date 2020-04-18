# coding=utf-8

import json, os, time, traceback
from .ranknet import build_model
from keras.callbacks import ModelCheckpoint
import numpy as np
from .utils import get_dataset
from os.path import dirname, abspath, join

rm = None
ddir = join(dirname(abspath(__file__)), "./data")
ftfile = join(ddir, "feat_list.txt")
tpfile = join(ddir, "type_list.txt")
mdfile = join(ddir, "model.h5")
def predict(items, intent="product", reftime=None):
	global rm
	if rm is None: rm = RankModel(model_file=mdfile)
	return rm.predict(items, intent, reftime)
def train(log_file="log.txt", click_file="data.xlsx"):
	global rm
	if rm is None: rm = RankModel()
	rm.fit(log_file, click_file)
def test(log_file="log.txt", click_file="data.xlsx"):
	global rm
	if rm is None: rm = RankModel(model_file=mdfile)
	rm.test(log_file, click_file)

def inset(url, urlset, title, titleset):
	return url and url.lower().strip() in urlset or title and title.lower().strip() in titleset
def get_attributes(item):
	return item["uid"], item["query"], item["results"], item["intention_calc"], item["timestamp"]
def get_title(item):
	return item.get("doc_title", item.get("DRETITLE", ""))
def get_url(item):
	return item.get("doc_url", item.get("DOC_URL", ""))
class RankModel:
	def __init__(self, feat_list=ftfile, type_list=tpfile, model_file=None):
		if type(feat_list) == str:
			feat_list = [line.strip("\r\n").split("\t")[0] for line in open(feat_list, encoding="utf-8")]
		self.feat_inds = {y:x for x, y in enumerate(feat_list)}
		self.feat_list = feat_list

		if type(type_list) == str:
			type_list = [line.strip("\r\n") for line in open(type_list, encoding="utf-8")]
		self.type_inds = {y:x for x, y in enumerate(type_list)}
		self.type_list = type_list

		self.model, self.lmodel = build_model(len(feat_list), len(type_list))
		if model_file and os.path.exists(model_file):
			self.lmodel.load_weights(model_file)
		if model_file is None: self.model_file = mdfile
	def extract_feat(self, item, default=0):
		return [float(item.get(f, default)) for f in self.feat_list]
	def extract_type(self, item, default=0):
		typ = item.get("doc_type", item.get("type"))
		return self.type_inds.get(typ, default)
	def extract_bm25(self, item, default=1):
		return item.get("bm25_score", item.get("score", default))
	def extract_timediff(self, item, reftime):
		last_time = max(float(item.get("DREDATE", 1e9)), float(item.get("DOC_MODIFY_DATE", 1e9)))
		return max((reftime - last_time)/86400, 0)
	def extract(self, X, r, intent, reftime):
		X[0].append([self.extract_bm25(r)])
		X[1].append([int(intent.lower().strip()=="product")])
		X[2].append(self.extract_feat(r))
		X[3].append([self.extract_type(r)])
		X[4].append([self.extract_timediff(r, reftime)])
	def check_features(self, X):
		print(X)
		for j, x in enumerate(X):
			sx = x.mean(axis=0)
			print(sx)
			for i, s in enumerate(sx):
				print(i,s)
				if all(x[:,i]==s):
					print("empty feature %d, %d"%(j, i))
	def makeX(self, X):
		for i in range(len(X)): X[i] = np.array(X[i])
	def initializeX(self): 
		return [[] for _ in range(5)]
	def fit(self, log_file, click_file, n_negs=500):
		qs = get_dataset(click_file)
		print('qs:',qs)
		X = self.initializeX()
		pairs = []
		with open(log_file, encoding="utf-8") as fin:
			for line in fin:
				print(line)
				try: item = json.loads(line.strip())
				except Exception:traceback.print_exc()
				print('item：',item)
				uid, q, results, intent, reftime = get_attributes(item)
				print(uid, q)
				if (uid, q) in qs:
					print('(uid, q):',(uid, q))
					titleset, urlset = zip(*qs[uid, q])
					titleset = {t.lower() for t in titleset if t.strip()}
					urlset = {u.lower() for u in urlset if u.strip()}
					start = len(X[0])
					pos_ids = set()
					print('results:',results)
					for rr in results:
						self.extract(X, rr, intent, reftime)
						if inset(get_url(rr), urlset, get_title(rr), titleset):
							pos_ids.add(len(X[0])-1)
					for pos in pos_ids:
						num = 0
						for i in range(start, len(X[0])):
							if i not in pos_ids: 
								pairs.append((pos, i))
								num += 1
								if num >= n_negs: break
		print('X:',X)
		# [[[43.280008177087154], [41.190123208617685], [36.84539707788369], [50.37942892056295], [55.37942892056295], [51.37942892056295]], [[1], [1], [1], [1], [1], [1]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0], [0], [0], [0], [0], [0]], [[161744.00902777776], [167942.59259259258], [161639.54513888888], [209.24652777777777], [209.24652777777777], [209.24652777777777]]]
		self.makeX(X)
		print('X after make:', X)
		self.check_features(X)
		X_ind_1, X_ind_2 = zip(*pairs)

		X_train = [x[list(X_ind_1)] for x in X] + [x[list(X_ind_2)] for x in X]
		Y_train = np.ones(X_train[0].shape)
		init_weights = [w.copy() for w in self.model.get_weights()]
		try:
			self.lmodel.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.1, \
				callbacks=[ModelCheckpoint(self.model_file, save_weights_only=True, save_best_only=True)])
		except Exception: traceback.print_exc()
		except: print("Interupted")
		self.lmodel.load_weights(self.model_file)
		final_weights = self.model.get_weights()
		for i, (w0, w1) in enumerate(zip(init_weights, final_weights)):
			print("weight %d, same num %d/%d"%(i, (w0==w1).sum(), (w0==w0).sum()))
	def predict(self, items, intent="product", reftime=None):
		X = self.initializeX()
		if reftime is None: reftime = time.time()
		for r in items:
			self.extract(X, r, intent, reftime)
		self.makeX(X)
		return self.model.predict(X, batch_size=len(X[0])).reshape(-1)
	def test(self, log_file, click_file=""):
		qs = get_dataset(click_file)
		acc = {10:0, 20:0, 50:0, 100:0, 500:0, 1000:0}
		acc_orig = acc.copy()
		acc_match = acc.copy()
		tot = 0
		with open(log_file, encoding="utf-8") as fin:
			for line in fin:
				try: item = json.loads(line.strip())
				except Exception: continue
				uid, q, results, intent, reftime = get_attributes(item)
				if (uid, q) in qs:
					ngts = len(qs[uid, q])
					titleset, urlset = zip(*qs[uid, q])
					titleset = {t.lower() for t in titleset if t.strip()}
					urlset = {u.lower() for u in urlset if u.strip()}

					match_scores = np.array([self.extract_bm25(r) for r in results], dtype="float32")
					sorted_indices_match = (-match_scores).argsort(axis=-1)

					scores = self.predict(results, intent, reftime)
					sorted_indices = (-scores).argsort(axis=-1)
					n_rr = n_or = n_mr = 0
					for rank, idx in enumerate(sorted_indices):
						if n_or < ngts and inset(get_url(results[rank]), urlset, get_title(results[rank]), titleset):
							n_or += 1
							for i in acc_orig:
								if rank < i: acc_orig[i] += 1
						if n_mr < ngts and inset(get_url(results[sorted_indices_match[rank]]), urlset, get_title(results[sorted_indices_match[rank]]), titleset):
							n_mr += 1
							for i in acc_match:
								if rank < i: acc_match[i] += 1
						if n_rr < ngts and inset(get_url(results[idx]), urlset, get_title(results[idx]), titleset):
							n_rr += 1
							for i in acc:
								if rank < i: acc[i] += 1
					tot += ngts
		print("acc@k\torigin\tonly bm25\tranknet")
		for k,v in sorted(acc.items()):
			print("%d\t%.4f\t%.4f\t%.4f"%(k, acc_orig[k]/tot, acc_match[k]/tot, v/tot))


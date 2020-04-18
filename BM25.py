import os, sys, time, utils, math, struct, random, json, re
import jieba, shutil
import numpy as np
import traceback
import h5py

try:
    from pymongo import MongoClient
    from bson import ObjectId
except:
    pass
time.clock()

datadir = 'data/'


class BM25:
    def __init__(self):
        if not os.path.exists(datadir): os.mkdir(datadir)
        self.num_invfile = 100
        self.Initialize()

    def Initialize(self):
        self.docu_lengths = []
        self.docu_list = []
        self.word_docu_freq = []
        self.id2w = [];
        self.w2id = {}
        self.inv = {};
        self.woffset = {}
        self.cache = [];
        self.coffset = {}
        self.idf_fn = None

    def ClearData(self):
        shutil.rmtree(datadir)
        time.sleep(0.5)
        os.mkdir(datadir)
        self.Initialize()

    def GetFileno(self, w):
        return w % self.num_invfile

    def LoadData(self):
        self.id2w = utils.LoadList(datadir + 'i_id2w.txt')
        self.w2id = {y: x for x, y in enumerate(self.id2w)};
        with h5py.File(datadir + 'i_counts.h5') as dfile:
            self.docu_lengths = dfile['dl'][:]
            self.word_docu_freq = dfile['df'][:]
        self.docu_lengths = list(self.docu_lengths)
        self.word_docu_freq = list(self.word_docu_freq)
        self.avgdl = np.mean(self.docu_lengths)
        with h5py.File(datadir + 'i_offset.h5') as dfile: self.woffset = dfile['woffset'][:]
        self.docu_list = utils.LoadList(datadir + 'i_docuid.txt')
        self.ComputeIDF()
        print('Load OK')

    def AddDocument(self, docu_id, tokens):
        index = len(self.docu_list)
        self.docu_list.append(docu_id)
        tf = {}
        if type(tokens) is type([]):
            for wd in tokens: tf[wd] = tf.get(wd, 0) + 1
        elif tokens.items is not None:
            for wd, weight in tokens.items():
                tf[wd] = tf.get(wd, 0) + weight
        self.docu_lengths.append(sum(tf.values()))
        tf = {x: y for x, y in tf.items() if x not in '\r\t\n '}
        for w in tf.keys():
            if not w in self.w2id:
                self.w2id[w] = len(self.id2w)
                self.id2w.append(w);
                self.word_docu_freq.append(0)
        for ww, num in tf.items():
            w = self.w2id[ww]
            self.word_docu_freq[w] += 1
            self.inv.setdefault(w, []).append(index)
            self.inv[w].append(min(num, 65535))
            if len(self.inv[w]) > 20000: self.inv[w] = self.inv[w][10000:]

    def Defrag(self, fileno):
        print('ReConstruct %d ...' % fileno)
        fn = datadir + 'i_inv_%d.bin' % fileno
        dat = []
        with open(fn, 'rb') as finv:
            for wi in range(fileno, len(self.id2w), self.num_invfile):
                if self.woffset[wi] >= 0:
                    finv.seek(self.woffset[wi])
                    nm = struct.unpack('i', finv.read(4))[0]
                    bts = finv.read(4 * nm * 2)
                    dat.append((wi, nm, bts))
        with open(fn, 'wb') as finv:
            for wi, nm, bts in dat:
                self.woffset[wi] = finv.tell()
                finv.write(struct.pack('i', int(nm)))
                if nm > 0: finv.write(bts)
        with h5py.File(datadir + 'i_offset.h5', 'w') as dfile:
            dfile.create_dataset('woffset', data=np.array(self.woffset))

    def ComputeIDF(self):
        if self.idf_fn: return self.LoadIDF()
        print('Computing IDF ...')
        tot_docus = len(self.docu_list)
        idf = (tot_docus - np.array(self.word_docu_freq) + 0.5) / \
              (np.array(self.word_docu_freq) + 0.5)
        idf[idf <= 1.0001] = 1.0001
        self.idf = np.log(idf)

    def computeIDFByDf(self,df):

        # tot_docus = len(self.docu_list)
        tot_docus=len(self.docu_lengths)
        idf = (tot_docus - df + 0.5) / \
              (df + 0.5)
        idf=max(idf,1)
        self.idf = np.log(idf)



    def LoadIDF(self):
        print('Load IDF from file %s' % self.idf_fn)
        idf = np.zeros(len(self.id2w))
        for ww, nn in utils.LoadCSVg(self.idf_fn):
            if ww in self.w2id: idf[self.w2id[ww]] = float(nn)
        self.idf = idf

    def Commit(self):
        self.ComputeIDF()
        self.avgdl = np.mean(self.docu_lengths)
        print('Saving Inv Tables ...')
        if type(self.woffset) is not type({}):
            self.woffset = {x: y for x, y in enumerate(self.woffset)}
        lastfileno = -1
        ws = sorted([(self.GetFileno(x), x) for x in self.inv.keys()])
        for fileno, wi in ws:
            fn = datadir + 'i_inv_%d.bin' % fileno
            if lastfileno != fileno:
                if lastfileno >= 0: finv.close()
                finv = open(fn, 'ba+')
                lastfileno = fileno
            rlist = self.inv[wi]
            num = len(rlist) // 2;
            nm = 0
            if self.woffset.get(wi, -1) >= 0:
                finv.seek(self.woffset[wi])
                nm = struct.unpack('i', finv.read(4))[0]
                bts = finv.read(4 * nm * 2)
                num += nm
                finv.seek(0, 2)
            self.woffset[wi] = finv.tell()
            finv.write(struct.pack('i', int(num)))
            if nm > 0: finv.write(bts)
            for z in rlist: finv.write(struct.pack('i', int(z)))
            if wi in self.coffset: self.coffset.pop(wi)
        if lastfileno >= 0: finv.close()
        self.inv = {}
        self.woffset = [self.woffset.get(i, -1) for i in range(len(self.id2w))]
        print('Saving ...')
        with h5py.File(datadir + 'i_counts.h5', 'w') as dfile:
            dfile.create_dataset('dl', data=np.array(self.docu_lengths))
            dfile.create_dataset('df', data=np.array(self.word_docu_freq))
        utils.SaveList(self.docu_list, datadir + 'i_docuid.txt')
        utils.SaveList(self.id2w, datadir + 'i_id2w.txt')
        with h5py.File(datadir + 'i_offset.h5', 'w') as dfile:
            dfile.create_dataset('woffset', data=np.array(self.woffset))
        self.Defrag(random.randint(0, self.num_invfile - 1))

    def GetList(self, w):
        if not w in self.coffset:
            if len(self.cache) > 1048576 * 100:
                self.cache = [];
                self.coffset = {}
            self.coffset[w] = len(self.cache)#长度？
            fileno = self.GetFileno(w)
            fn = datadir + 'i_inv_%d.bin' % fileno
            with open(fn, 'rb') as fin:
                fin.seek(self.woffset[w])
                num = struct.unpack('i', fin.read(4))[0]
                num = min(num, 10000)
                self.cache.append(num)
                for i in range(num * 2):
                    zz = struct.unpack('i', fin.read(4))[0]
                    self.cache.append(zz)
        return self.coffset[w]

    def Query(self, Q, k1=1.5, b=0.75, limit=100):
        qlst = []
        for qq in Q.split(' '):
            qlst += jieba.lcut(qq)
        sc = {}
        for q in qlst:
            if not q in self.w2id: continue
            w = self.w2id[q]
            qo = self.GetList(w)
            num = self.cache[qo]
            for i in range(num):
                docid, tf = self.cache[qo + i * 2 + 1:qo + i * 2 + 3]
                sc[docid] = sc.get(docid, 0) + \
                            self.idf[w] * (tf * (k1 + 1)) / (
                                        tf + k1 * (1 - b + b * self.docu_lengths[docid] / self.avgdl))
        ret = utils.FreqDict2List(sc)[:limit]
        ret = [(self.docu_list[x], y) for x, y in ret]
        return ret

    def QueryDict1(self, Qs, k1=1.5, b=0.75, limit=100):
        sc = {}#[(docID,score)......]
        for q, weight in Qs.items():
            if not q in self.w2id: continue
            w = self.w2id[q]#词频(整个文档库中包含此语素的文档数) 用于计算idf
            qo = self.GetList(w)
            num = self.cache[qo]#检索出的文档个数
            for i in range(num):
                docid, tf = self.cache[qo + i * 2 + 1:qo + i * 2 + 3]#tf 语素在此文档出现的频数
                #文档得分累加
                sc[docid] = sc.get(docid, 0) + weight * \
                            (self.idf[w] * (tf * (k1 + 1)) / (
                                        tf + k1 * (1 - b + b * self.docu_lengths[docid] / self.avgdl)))
        ret = utils.FreqDict2List(sc)[:limit]
        ret = [(self.docu_list[x], y) for x, y in ret]#映射到documents的docID
        return ret



    #从数据库表中查询倒排索引



    def QueryDict(self, Qs, k1=1.5, b=0.75, limit=100):
        sc = {}#[(docID,score)......]
        for w, weight in Qs.items():
            df,docs,freqs= self.GetList(w)
            if not docs:continue
            idf=self.ComputeIDF(df)
            for i in range(num):
                docid, tf = self.cache[qo + i * 2 + 1:qo + i * 2 + 3]#tf 语素在此文档出现的频数
                #文档得分累加
                sc[docid] = sc.get(docid, 0) + weight * \
                            (self.idf[w] * (tf * (k1 + 1)) / (
                                        tf + k1 * (1 - b + b * self.docu_lengths[docid] / self.avgdl)))
        ret = utils.FreqDict2List(sc)[:limit]
        ret = [(self.docu_list[x], y) for x, y in ret]#映射到documents的docID
        return ret

def LoadConfig():
    global config
    try:
        with open('config.json', encoding='utf-8') as fin:
            config = json.load(fin)
        for x, y in config.items(): print('Config %s: %s' % (x, str(y)))
    except:
        traceback.print_exc()


def ConnectDB():
    client = MongoClient('mongodb://%s/' % config.get('mongo_host', ''))
    if config.get('mongo_user', '') != '':
        client.admin.authenticate(config.get('mongo_user', ''), config.get('mongo_passwd', ''),
                                  mechanism=config.get('mongo_auth_mech', 'SCRAM-SHA-1'))
    db = client.get_database(config.get('mongo_db', ''))
    collection = db.get_collection(config.get('mongo_collection', ''))
    return collection


### CreateIndexXXX
def CreateIndexCSV(fn):
    for ii, z in enumerate(utils.LoadCSVg(fn)):
        if ii > 0 and ii % 1000 == 0: print(ii)
        words = jieba.lcut(z[-1])
        bm25.AddDocument(z[0], words)
    bm25.Commit()
    print('Create index completed')


def CreateIndexMongo(collection, filter, fields):
    for ii, docu in enumerate(collection.find(filter)):
        if ii > 0 and ii % 1000 == 0: print(ii)
        docu_id = str(docu['_id'])
        words = {}
        for field, ww in fields.items():
            for wd in jieba.lcut(docu.get(field, '')):
                words[wd] = words.get(wd, 0) + ww
        bm25.AddDocument(docu_id, words)
    bm25.Commit()
    print('Create index completed')


def CreateIndexDir(rootdir):
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            fn = os.path.join(parent, filename)
            try:
                print(fn)
                with open(fn, encoding='utf-8') as fin:
                    text = fin.read()
                chstxt = re.sub('[^\n\t \u4e00-\u9fa5]', '', text).strip()
                words = jieba.lcut(chstxt + ' ' + fn)
                bm25.AddDocument(fn, words)
            except:
                pass
    bm25.Commit()
    print('Create index completed')


### CreateIndexXXX

if __name__ == '__main__':
    bm25 = BM25()
    if os.path.exists(datadir + 'i_id2w.txt'): bm25.LoadData()
    LoadConfig()


# bm25.ClearData()
# CreateIndexCSV('wenshus.txt')
    collection = ConnectDB()
    CreateIndexMongo(collection, config.get('document_filter', {}), config.get('fields', {}))
# CreateIndexDir('z:/data')

# def Test(x):
#     ret = bm25.Query(x)
#     rr = []
#     for fn, s in ret[:3]:
#         with open(fn, encoding='utf-8') as fin:
#             text = fin.read()
#         chstxt = re.sub('[^\n\t \u4e00-\u9fa5]', '', text).strip()
#         chstxt = re.sub('[\r\t\n ]+', ' ', chstxt)
#         print(s, fn)
#         print(chstxt)
#
#
# import bottle
# from bottle import request
#
# app = bottle.Bottle()
#
#
# @app.route('/query')
# def query():
#     Q = request.params.q
#     ret = Query(Q)
#     ret = [(x, y) for x, y in ret]
#     return json.dumps(ret, ensure_ascii=False)
#
#
# if __name__ == '__main__':
#     if 'server' in sys.argv:
#         bottle.run(app, 'tornado', host=config.get('host', '127.0.0.1'), port=config.get('port', 13333))
#     print('completed')

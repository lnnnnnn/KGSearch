import os, sys, time, utils, json, re, math, jieba, random
from collections import defaultdict
import html
from gen_tags import DocumentTags, Trie
from wtrie import WTrie
import BM25


class TagGraph:
    def __init__(self):
        self.edges = {}
        self.nodes = set()
        self.db = None

    def set_nodes(self, nodes):
        self.nodes = nodes

    def add_edge(self, u, v, t, w):
        self.edges.setdefault(u, []).append((t, v, w))

    def _get_edge(self, u):
        return self.edges.get(u, [])

    def query(self, us, etps={}):
        ret = {}
        for u in us:
            q = [u];
            qh = 0
            w = defaultdict(float);
            w[u] = 1
            pp = {u: -1};
            pt = {}
            while qh < len(q) and len(q) < 100:
                z = q[qh];
                qh += 1
                for e_type, v, e_weight in self._get_edge(z):
                    tpscore = etps.get(e_type, 0)
                    sc = w[z] * e_weight * tpscore
                    if sc < 0.05: continue
                    if w[v] == 0: q.append(v)
                    if sc > w[v]:
                        w[v], pp[v], pt[v] = sc, z, e_type
            w = {k: min(v, 1) for k, v in w.items()}
            ret[u] = w;
            ret[(u, 0)] = pp;
            ret[(u, 1)] = pt
        return ret

    def get_path(self, z, pp, pt=None):
        zs = []
        while z != -1:
            zs.append(z);
            if z in pt: zs.append(pt[z])
            z = pp[z]
        return list(reversed(zs))

    def is_node(self, sr):
        return sr in self.nodes

    def save_graph(self):
        if not os.path.isdir('saved_graph'): os.mkdir('saved_graph')
        utils.SaveList(self.nodes, 'saved_graph/graph_nodes.txt')
        with open('saved_graph/graph_edges.txt', 'w', encoding='utf-8') as fout:
            for tp, ee in self.edges.items():
                for u, vw in ee.items():
                    for v, w in vw.items():
                        utils.WriteLine(fout, [tp, u, v, w])
        print('Saved')

    def load_graph(self):
        self.nodes = utils.LoadSet('saved_graph/graph_nodes.txt')
        for edge in utils.LoadCSVg('saved_graph/graph_edges.txt'):
            tp, u, v, w = edge
            self.add_edge(u, v, tp, float(w))
        print('Loaded')


class TagGraphDBCache(TagGraph):
    def __init__(self, cache_size=1000000):
        self.cache_size = cache_size
        self.outdate_nodes = set()
        return super().__init__()

    def _get_edge(self, u):
        if u in self.edges and u not in self.outdate_nodes: return self.edges[u]
        self.load_edges(u)
        return self.edges.get(u, [])

    def save_graph(self):
        pass

    def load_graph(self, db):
        self.db = db

    def add_outdated_node(self, node):
        self.outdate_nodes.add(node)

    def is_node(self, sr):
        if sr in self.nodes: return True
        yes = self.db.nodes.find_one({'_id': sr}) is not None
        if yes: self.nodes.add(sr)
        return yes

    def load_edges(self, u):
        if len(self.edges) > self.cache_size * 0.5:
            if random.random() > 0.5: dt.popitem()
        if len(self.edges) > self.cache_size:
            for _ in range(1000): dt.popitem()
        self.edges[u] = []
        for item in self.db.edges.find({'u': u}):
            self.edges.setdefault(u, []).append((item['type'], item['v'], item['w']))
        if u in self.outdate_nodes:    self.outdate_nodes.remove(u)


dt = DocumentTags()
dt.load_phrases()


def GetTags(text):
    text = re.sub('[ ]+', ' ', text).lower()
    # if re.search('[\u4e00-\u9fa5]+', text) is not None:
    words = jieba.lcut(text)
    words = [x for x in words if x.strip() != '']
    tt = dt.get_tags(text)
    for t in words: tt[t] += 1
    return tt


def BuildBM25(graph, makebm25=False):
    global idf, tags
    if not makebm25: return
    N = 0
    for jj in utils.LoadList('training/all_data.txt'):
        jj = json.loads(jj)
        tf = GetTags(jj['text'])
        N += 1
        bm25.AddDocument(jj['id'], tf)
    bm25.Commit()
    print('docu segs:', N)
    if len(graph.nodes) == 0 and self.db is None:
        tags = set()
        for i, df in enumerate(bm25.word_docu_freq):
            w = bm25.id2w[i]
            if len(w) > 1 and df > 2 and bm25.idf[i] > 2: tags.add(w)
        graph.set_nodes(tags)
        print('tags:', len(tags))


def Query(q):
    qs = [x for x in jieba.lcut(q) if x != ' ']
    qs += dt.get_tags(q)
    qtags = [x for x in qs if graph.is_node(x)]#标签词
    qwords = [x for x in qs if x not in set(qtags)]#普通词
    print('qtags:',qtags)
    edgew = {'self': 1, 'KG': 0.2, 'coocc': 0.2, 'KGm': 0.7, 'KGi': 0.15}
    '''
    rnodes:{'解决方案':{'解决方案'：1，'solution':0.35},('解决方案',0):{'解决方案'：-1,'solution':'解决方案'},
    ('解决方案'，1):{'solution':'encn'},'技术':{},('技术',0):{},('技术',1):{}}
    '''
    rnodes = graph.query(qtags, edgew)#扩展节点
    Qws = {}#保存标签词的权重{节点:[权重]}
    Qs = defaultdict(float)#所有词的得分
    for u, ww in rnodes.items():
        if type(u) is type(tuple()): continue
        for v, w in ww.items():
            Qws.setdefault(v, []).append(w)
    #Qws:{'技术':[1],'technology':[0.35],'解决方案':[1],'solution':[0.35]}
    # merge, max(a,b,c) < r < sum(a,b,c)
    for x, ys in Qws.items():
        Qs[x] = (max(ys) + sum(ys)) * 0.5
    for word in qwords: Qs[word] += 0.5

    print('Qs:',Qs)
    Qlist = utils.FreqDict2List(Qs)
    #Qlist:[('解决方案',1.2),('iot',1.2),('技术'，1.0),('solution':0.55)......]
    # print('Qlist:',Qlist)

    # print(Qlist[:10])

    gends = [x[0] for x in Qlist[:10]]
    #gends:['技术','解决方案','technology','solution'] 降序节点列表
    #ends:['解决方案','solution'] pp:{'解决方案'：-1,'solution':'解决方案'} pt:{'solution':'encn'}
    paths = []
    for u, ww in rnodes.items():
        if type(u) is type(tuple()): continue
        pp, pt = rnodes[(u, 0)], rnodes[(u, 1)]
        ends = [x for x in gends if ww.get(x, 0) > 0.1]
        for end in ends:
            path = graph.get_path(end, pp, pt)
            if len(path) == 1: continue
            paths.append((path, ww[end]))
    paths.sort(key=lambda x: -x[1])

    ret = bm25.QueryDict(Qs, b=0.3)
    print('doc:',ret)
    # doc: [('blog_11113', 6.981760701226563), ('blog_11111', 6.967036468215555), ('blog_11112', 6.955284710133972),
    #       ('blog_11114', 6.933101299616559), ('blog_11115', 6.848454952712912)]
    #得到一系列文档的bm25分数
    #[('docID':5.8987),('docID':5.8987),......]
    # for zz, sc in ret[:10]: print(sc, datadict[zz])
    return ret, Qs, paths


def GetDocuText(ii):
    '''
    This function must be changed for documents.
    '''

    def _LoadDocumentDetails():
        global datadict
        datadict = {}
        for jj in utils.LoadList('training/all_data.txt'):
            jj = json.loads(jj)
            datadict[jj['id']] = jj['text']

    if not 'datadict' in dir(): _LoadDocumentDetails()
    return datadict[ii]


import bottle
from bottle import route, template, request, redirect, static_file

app = bottle.app()


@app.route('/style/<filename:path>')
def server_static(filename):
    return static_file(filename, root='style/')


@app.route('/')
def index():
    return template('suggest_server.html')


@app.route('/outdate')
def add_outdated():
    node = request.params.node
    if node != '': graph.add_outdated_node(node)
    return 'ok'


def coloring(text, tc):
    trie = WTrie(tc)
    ret = trie.search(text.lower())
    locs = sorted(ret.keys(), key=lambda x: (x[0], -x[1]))
    tt = [];
    ii = 0
    ws = set()
    for loc in locs:
        if loc[0] < ii: continue
        tt.append(text[ii:loc[0]])
        tt.append('<span style="color:rgb(%d,0,0)">' % tc[ret[loc]])
        tt.append(text[loc[0]:loc[1]])
        ws.add(text[loc[0]:loc[1]])
        tt.append('</span>')
        ii = loc[1]
    tt.append(text[ii:])
    return ''.join(tt), ws


def select_best_paths(paths, wweights, num=3):
    pps = [(pp, w * wweights[pp[-1]]) for pp, w in paths if pp[-1] in wweights]
    pps.sort(key=lambda x: -x[1])
    return [x[0] for x in pps[:num]]


@app.route('/rec', methods=['GET', 'POST'])
def rec():
    text = request.params.data
    text = text.lower()
    sent = text.strip('。')
    ret, Qs, paths = Query(sent)
    for x in paths: print(x)
    rlist = []
    for x in ret[:10]:
        text = GetDocuText(x[0])
        text = html.escape(text)
        ts = {};
        ws = {}
        for word, weight in sorted(Qs.items(), key=lambda x: len(x[0])):
            if not word in bm25.w2id: continue
            wmult = bm25.idf[bm25.w2id[word]] * 0.25
            weight = weight * wmult
            ws[word] = wmult * 0.25
            ts[word] = int(min(max(weight, 0), 1) * 200 + 55)
        text, wws = coloring(text, ts)
        ws = {x: y for x, y in ws.items() if x in wws}
        path = select_best_paths(paths, ws, 3)
        path = '<br/>'.join([' '.join([('-' + x + '->' if i % 2 else x) for i, x in enumerate(x)]) for x in path])
        zz = '<a>%s</a> %.2f<br/>' % (x[0], x[1]) + text
        zz += '<br/><br/>' + path
        rlist.append(zz)
    return json.dumps({'ret': rlist}, ensure_ascii=False)


if __name__ == '__main__':
    bm25 = BM25.BM25()
    # bm25.idf_fn = 'training/pretrain_idf.txt'
    if 'clear' in sys.argv: bm25.ClearData()
    if os.path.exists('data/i_id2w.txt'): bm25.LoadData()
    first = len(bm25.id2w) == 0
    graph = TagGraph()
    if not first: graph.load_graph()
    if 'build' in sys.argv:
        BuildBM25(graph, first)
        if first: graph.save_graph()
    if 'server' in sys.argv:
        bottle.run(app, host='0.0.0.0', port=41324, server='tornado')
    # print('done')

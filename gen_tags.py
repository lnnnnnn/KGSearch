import os, sys, time, re, ljqpy, math, json
from collections import defaultdict
import jieba.analyse
from kg_utils import get_collections

graph_nodes=get_collections(['nodes'])[0]
# print(graph_nodes)
# print(graph_nodes.find())



def TextRank():
    for sen in ljqpy.LoadList('training/merged_text.txt'):
        if not '，' in sen: continue
        print(sen)
        for x, w in jieba.analyse.textrank(sen, withWeight=True, allowPOS=None):
            print('%s %s' % (x, w))
        print('-' * 30)


def GetTags(text):
    text = re.sub('[ ]+', ' ', text).lower()
    tags = jieba.lcut(text)
    tags = [x for x in tags if x.strip() != '']
    tt = defaultdict(int)
    for t in tags: tt[t] += 1
    return tt


def log_sum_exp(ys):
    mm = max(ys)
    return mm + math.log(sum(math.exp(y - mm) for y in ys))


from wtrie import WTrie

Trie = WTrie


class DocumentTags:
    def gen_new_tags(self, corpusfn, numlim=1000):
        global ng1, ng2, ng3, pg1, pg2, pg3, pdict, ndict, scores

        def _HH(p):
            return -p * math.log(p) if p > 0 else 0

        def _HY(g3, g2):
            return _HH(ng3[g3] / ng2[g2])

        ng1 = defaultdict(int)
        ng2 = defaultdict(int)
        ng3 = defaultdict(int)
        pdict, ndict = {}, {}
        cnum = 0
        for ii, lines in enumerate(ljqpy.LoadCSVg(corpusfn)):
            line = lines[0]
            if ii % 100000 == 0: print('counting', ii)
            if line == '': continue
            if len(line) < 10: continue
            if re.search('[a-zA-Z\u4e00-\u9fa5]{2,}', line) is None: continue
            lln = jieba.lcut(line)
            lln = ['^'] + lln + ['$']
            for i, wd in enumerate(lln):
                ng1[wd] += 1
                if i > 0: ng2[tuple(lln[i - 1:i + 1])] += 1
                if i > 1: ng3[tuple(lln[i - 2:i + 1])] += 1
                if i > 1:
                    pdict.setdefault(tuple(lln[i - 1:i + 1]), set()).add(lln[i - 2])
                    ndict.setdefault(tuple(lln[i - 2:i]), set()).add(lln[i])
            cnum += len(lln)
        log_all_ng1 = math.log(sum(ng1.values()))
        log_all_ng2 = math.log(sum(ng2.values()))
        log_all_ng3 = math.log(sum(ng3.values()))
        pg1 = {k: math.log(v) - log_all_ng1 for k, v in ng1.items()}
        pg2 = {k: math.log(v) - log_all_ng2 for k, v in ng2.items()}
        pg3 = {k: math.log(v) - log_all_ng3 for k, v in ng3.items()}
        print('COUNT ok')

        # base_wp = {x:float(y) for x,y in ljqpy.LoadCSV('resources/base_wcounts.txt')}
        # pg1 = {k:(log_sum_exp([base_wp[k],v])-math.log(2) if k in base_wp else v) for k,v in pg1.items()}

        scores = {}
        ii = 0
        for k, v in ljqpy.FreqDict2List(pg2):
            ii += 1
            if ii % 10000 == 0: print('%d/%d' % (ii, len(pg2)))
            if max(ng1[k[0]], ng1[k[1]]) <= 3: continue
            pmi = v - pg1[k[0]] - pg1[k[1]]
            if pmi < 2: continue
            Hl, Hr = 0, 0
            Hlr, Hrl = 0, 0
            for ll in pdict.get(k, []):
                Hl += _HY((ll, k[0], k[1]), k)
                Hlr += _HY((ll, k[0], k[1]), (ll, k[0]))
            for rr in ndict.get(k, []):
                Hr += _HY((k[0], k[1], rr), k)
                Hrl += _HY((k[0], k[1], rr), (k[1], rr))
            score = pmi - min(Hlr, Hrl) + min(Hl, Hr)
            if not ljqpy.IsChsStr(k[0] + k[1]): continue
            scores[k] = score * ng2[k]

        phrases = []
        for k, v in ljqpy.FreqDict2List(scores)[:numlim]:
            print(k, v)
            phrases.append(''.join(k))
        self.newtags = phrases
        self.newtagtrie = Trie({x: 1 for x in self.newtags})
        return phrases

    def save_phrases(self):
        ljqpy.SaveList(self.newtags, 'training/phrases.txt')

    def load_phrases(self,table_nodes=graph_nodes):
        # self.newtags = ljqpy.LoadList('training/phrases.txt')
        self.newtags=set(x['node'].strip().lower() for x in table_nodes.find() if x['node'].strip())
        self.newtagtrie = Trie({x: 1 for x in self.newtags})

    def get_tags(self, docu):
        tf = defaultdict(int)
        for wd in self.newtagtrie.search(docu).values():
            tf[wd] += 1
        return tf


dt = DocumentTags()
if __name__ == '__main__':
    # TextRank()
    # dt.gen_new_tags('training/merged_text.txt', 100)
    # dt.save_phrases()
    print(dt.get_tags('深度学习和知识图谱的融合.'))
    print('done')

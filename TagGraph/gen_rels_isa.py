import jieba
from jieba import posseg
from collections import defaultdict

class UnionSet(object):
    def __init__(self):
        self.parent = {}

    def init(self,key):
        if key not in self.parent:
            self.parent[key] = key

    def findRoot(self,key):

        if self.parent[key]!= key:
            self.parent[key] = self.findRoot(self.parent[key])

        return self.parent[key]

    def find(self,key):
        if key not in self.parent:
            self.parent[key] = key
        else:
            self.findRoot(key)
        return self.parent[key]

    def join(self,key1,key2):
        p1 = self.find(key1)
        p2 = self.find(key2)

        if p1 != p2:
            self.parent[p2] = p1

    def getSets(self):#按类别划分
        res = defaultdict(list)
        for child in self.parent:
            res[self.findRoot(child)].append(child)

        return res
class ISA:
    def __init__(self,suffixFlag = False,addNewWordFlag = False):

        self.count = defaultdict(int)
        self.KOSword = set([])
        self.keyword = {}
        self.addNewWord = []
        self.flag = {}
        self.addNewWordFlag = addNewWordFlag
        self.suffixFlag = suffixFlag

        if self.suffixFlag:
            self.ngram = self.suffix
        else:
            self.ngram = self.ngram_all

        if addNewWordFlag:
            self.isValidWord = self.isValidWord_soft
        else:
            self.isValidWord = self.isValidWord_strict
    '''
       规则1：中心词必须存在于模型输入的词表中。
       规则2：中心词需满足以名词或动词类词性的词结尾，且除了词尾不包含单字的词

    '''
    def isValidWord_soft(self,word):
        #print('isValidWord_soft')
        if len(word) < 2: return False
        word_posseg = posseg.lcut(word)

        if 'n' !=  word_posseg[-1].flag[0] and 'v' != word_posseg[-1].flag[0]:
            return False
        if 'v'  == word_posseg[0].flag[0] :
            return False

        for w,f in word_posseg[:-1]:#除了结尾不能包含单字 水产品
            if len(w) < 2 :
                return False

        if word not in self.words:
            self.addNewWord.append(word)

        return True

    def isValidWord_strict(self,word):
        if word in self.words or word in self.KOSword:
            return True
        return False



    def suffix(self,word):
            '''
            基于后缀法的候选中心词生成。首先使用jieba对输入word短语进行分词，对分词后的列表组合，生成所有可能的后缀词，最后利用上述两个规则判断所生成的词或短语是否是有效的词。如短语“银行理财产品”分词后形成【“银行”、“理财”、“产品”】三个词，组合后生成【“产品”、“理财产品”】两个可能的后缀，其中后缀词【“理财产品”】通过上述两个规则判断，输出该短语。

            '''
            res = []
            wcut = jieba.lcut(word)
            l = len(wcut)
            for start in range(len(wcut)):
                w = "".join(wcut[start:l])
                if w not in self.flag:
                    self.flag[w] = self.isValidWord(w)
                if self.flag[w]:
                    res.append(w)
                    self.count[w] += 1

            return res

    def ngram_all(self,word):
            '''
         基于ngram的候选中心词生成。首先使用jieba对输入word短语进行分词，ngram生成所有可能短语组合，最后通过上述两个规则判断所生成的词或短语是否是有效的词。如短语“银行理财产品”分词后形成【“银行”、“理财”、“产品”】三个词，组合后生成【“产品”、“理财产品”、“银行理财”】三个可能的后缀，其中后缀词【“理财产品”、“银行理财”】通过上述两个规则判断，输出该短语列表。

            '''
            #print('in ngram all')
            res = []
            wcut = jieba.lcut(word)
            #print('word',word,'wcut:',wcut)
            l = len(wcut)
            for start in range(l):
                w = ""
                for ind in range(start,l):
                    w+ = wcut[ind]
                    if w not in self.flag:
                        self.flag[w] = self.isValidWord(w)
                    #print(w,self.flag[w])
                    if self.flag[w]:
                            res.append(w)
                            self.count[w] +=  1

            return res

    def checkKOSexist(self,class2words):
        us = UnionSet()

        for classname in class2words:
            for w in class2words[classname]:
                if w !=  classname:
                    us.join(classname,w)
        words_us = us.getSets()

        print('words_us',words_us)

        #判断每个连通图中是否有KOS
        ws_confirmed = []
        for p in words_us:
            for w in words_us[p]:
                if w in self.KOSword:
                    ws_confirmed.extend(words_us[p])
                    ws_confirmed.append(p)
                    break

        print('ws_confirmed:',ws_confirmed)

        self.KOSexistFlag = {}

        for classname in class2words:
            if classname in ws_confirmed:
                self.KOSexistFlag[classname] = True
                continue

            for w in class2words[classname]:
                if w in ws_confirmed:
                    self.KOSexistFlag[classname] = True
                    break

        print('self.KOSexistFlag:',self.KOSexistFlag)

        return self.KOSexistFlag


    def print_triple(self,class2words):
        triples = defaultdict(list)
        for classname in class2words:
            for w in set(class2words[classname]):
                if w  ==  triples:
                    continue
                triples[w].append(classname)
        tot = 0
        for s in triples:
            for o in triples[s]:
                if s == o:
                    continue
                print({'s':s,'p':'isA','o':o})
                tot+ = 1
        print('insert {} isA relationships'.format(tot))


    def parse(self,words,minfreq=0):
        #print('in parse words',words)
        word2class = {}
        class2word = defaultdict(list)

        self.words = words

        for w in words:
            word2class[w] = self.ngram(w)
            #print('word2class--11', word2class)
        print('word2class--1',word2class)
        if self.addNewWordFlag:
            for w in self.addNewWord:
                word2class[w] = self.ngram(w)

        print('word2class--2',word2class)

        for word in word2class:
            for each in word2class[word]:
                if self.count[each]> =  minfreq:
                    class2word[each].append(word)

        print('class2word',class2word)
        self.checkKOSexist(class2word)
        self.print_triple(class2word)
        return class2word


if __name__ =  = '__main__':
    keyword_file = 'new_word.txt'
    label = ISA(suffixFlag = False,addNewWordFlag = True) #False
    label.KOSword = {'企业BG':0}
    label.keyword = {'消费者BG':0,'运营商BG':0,'企业BG':0,'云BU':0,'税务信息化':0,'关联交易管理':0,'税务规划':0,
                   '直接税管理':0,'消费者BG财经':0,'消费者BG资金':0}

    kws = {'KOS':label.KOSword,'new':label.keyword}
    words = []
    for kw in kws:
        words.extend(kws[kw])
    label.parse(words,minfreq=0)

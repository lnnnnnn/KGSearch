from collections import defaultdict
import re,jieba,math

NGRAM_SIZE = 4
# ENTROPY_THRESHOLD = 0.01
# MUTUALINFO_THRESHOLD = 4
# FREQ_THRESHOLD = 2

ENTROPY_THRESHOLD = 0.0
MUTUALINFO_THRESHOLD = 0
FREQ_THRESHOLD = 0

node_index = -1
'''
按ngram插入可保证所有字符都在根节点的children中
'''
class TrieNode(object):
    def __init__(self,
                 frequence=0,
                 children_frequence=0,
                 parent=None,
                 index=-1):
        self.index = index
        self.parent = parent
        self.frequence = frequence#自身状态出现的频率
        self.children = {}
        self.children_frequence = children_frequence#真实子节点的个数

    def insert(self,char):
        global node_index
        self.children_frequence += 1
        if self.children.get(char) is None:
            node_index += 1#新增节点
        self.children[char] = self.children.get(char,TrieNode(parent=self,index=node_index))
        self.children[char].frequence += 1
        return self.children[char]

    def fetch(self,char):
        return self.children[char]

class TrieTree(object):
    def __init__(self,size = 6):
        self._root = TrieNode(index=-1)
        self.size = size

    def get_root(self):
        return self._root

    def insert(self,chunk):
        node  = self._root

        for char in chunk:
            node = node.insert(char)
            print('insert node info(index,parent index,f,cf,children)',
                  node.index,node.parent.index,node.frequence,node.children_frequence,node.children)

        if len(chunk) < self.size:
            node.insert('EOS')

    def fetch(self,chunk):
        node = self._root
        for char in chunk:
            node = node.fetch(char)
        return node


def preprocess(text):
    lines = []
    for line in text.strip().split('\n'):
        line = line.strip()
        #跳过没有中文字符的行、空行
        if len(re.findall('[\u4e00-\u9fa5]',line)) =  = 0:
            continue
        words = jieba.lcut(line,HMM = False)
        lines.append(words)

    return lines

def get_words_count(lines):
    words_count = defaultdict(int)
    tot = 0
    for line in lines:
        for word in line:
            words_count[word] += 1
            tot += 1
    return words_count,tot

def concate_chunk(chunk):
    word = '-'.join(chunk)
    #若新词里没有中文，用空格分割
    if len(re.findall('[\u4e00-\u9fa5]',word)) =  = 0:
        word = word.replace('-',' ')
    else:
        word = word.replace('-','')

    return word


def get_ngram(lines,words_count):
    '''
    前缀树的深度为ngram_size+1,用来保存ngram右边词的信息，方便计算信息熵和点互信息
    '''
    chunks_freq = defaultdict(int)
    tree_depth = NGRAM_SIZE + 1
    ngram_trie = TrieTree(tree_depth)
    for line in lines:
        for start in range(len(line)):#保证所有char都在字典树的根节点中
            end = start + NGRAM_SIZE
            chunk = line[start:end]#单位为词 e.g ['中'，'射频','与'，'基站']
            ngram_trie.insert(line[start:start + tree_depth])

            # chunk = line 用于调试
            ngram_trie.insert(chunk)

            while len(chunk) > 1:
                chunks_freq[tuple(chunk)] += 1
                chunk = chunk[:-1]#每次删除最后一个词

    return chunks_freq,ngram_trie

def calc_entropy(chunks,ngram_trie):

    def entropy(sample,tot):
        s = float(sample)
        t = float(tot)
        result = - s / t * math.log(s / t)
        return result

    def parse(chunk,ngram_trie):
        node = ngram_trie.fetch(chunk)
        print('in entropy parse , node index',node.index)
        tot = node.children_frequence
        return sum([entropy(sub_node.frequence,tot) for sub_node in node.children.values()])

    word2entropy = {}
    for chunk in chunks:
        sc = parse(chunk,ngram_trie)
        if sc > ENTROPY_THRESHOLD:
            word2entropy[chunk] = sc

    return word2entropy

def calc_pmi(chunks,ngram_trie,words_count,total_words):
    '''
     pmi:log(p(x,y)/(p(x)*p(y))) = log(p(y|x)/p(y))
     y为chunk最后一个词的频率
    '''
    def parse(chunk):
        node_xy = ngram_trie.fetch(chunk)
        p_xy = node_xy.parent
        yf = words_count.get(chunk[-1],0) #chunk最后一个字符的频率
        pyx = float(node_xy.frequence) / p_xy.children_frequence
        py = float(yf) / total_words
        pmi = math.log(pyx / py)
        return pmi

    word2pmi = {}
    for chunk in chunks:
        sc = parse(chunk)
        if sc > MUTUALINFO_THRESHOLD:
            word2pmi[chunk] = sc

    return word2pmi

def fparse(lines,words_count,tot):

    chunks_freq,ngram_trie = get_ngram(lines,words_count)
    word_entropy = calc_entropy(chunks_freq.keys(),ngram_trie)
    word_pmi = calc_pmi(chunks_freq.keys(),ngram_trie,words_count,tot)

    return chunks_freq,word_entropy,word_pmi


def parse(text):
    all_lines = preprocess(text)
    words_count,total_count = get_words_count(all_lines)#words_count为词频字典 total_count为总词数
    chunks_freq,entropy,pmi = fparse(all_lines,words_count,total_count)

    print('chunks_freq,entropy,pmi',chunks_freq,entropy,pmi)

    #选取词频、信息熵、点互信息均符合要求的短语
    new_words = {}

    for chunk,freq in chunks_freq.items():
        if freq >= FREQ_THRESHOLD and \
            chunk in entropy and \
            chunk in pmi:
                new_words[concate_chunk(chunk)] = (freq,pmi,entropy)

    return new_words,words_count


if __name__ == '__main__':

    text = '全部分类'
    all_lines = [['全部','分类'],['分类'],['基站','平台','软件'],['基站','北','研','组']]
    words_count = {'全部':1,'分类':2,'基站':2,'平台':1,'软件':1,'北':1,'研':1,'组':1}
    # print(parse(text))
    chunks_freq, entropy, pmi  =  fparse(all_lines, words_count, 10)

    print('chunks_freq,entropy,pmi', chunks_freq,'\n', entropy,'\n', pmi)

'''
chunks_freq,entropy,pmi defaultdict(<class 'int'>, {('全部', '分类'): 1, ('基站', '平台', '软件'): 1, ('基站', '平台'): 1, ('基站', '北', '研', '组'): 1, ('基站', '北', '研'): 1, ('基站', '北'): 1}) 
 {} 
 {('全部', '分类'): 1.6094379124341003, ('基站', '平台', '软件'): 2.302585092994046, ('基站', '平台'): 1.6094379124341003, ('基站', '北', '研', '组'): 2.302585092994046, ('基站', '北', '研'): 2.302585092994046, ('基站', '北'): 1.6094379124341003}

'''
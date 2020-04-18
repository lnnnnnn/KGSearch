import string
import ahocorasick

class TagsTrie:
    MERGE_CHARS=set (string.digits+string.ascii_letters)

    def __init__(self,tags):
        self.tree=ahocorasick.Automaton()
        for tag in tags:
            self.tree.add_word(TagsTrie.cut_sent(tag),tag)
        self.tree.make_automaton()

    @staticmethod
    def cut_sent(sent):
        '''
        sent:996项目asd计划
        ret:['','996','项’,'目','asd','计','划','']

        '''
        print('in cut_sent,sent:',sent)

        sent=sent.upper()
        ret=[""]

        for c in sent:
            if c in TagsTrie.MERGE_CHARS and ret[-1] and ret[-1][-1] in TagsTrie.MERGE_CHARS:
                ret[-1]+=c
                continue
            ret.append(c)
        ret.append("")
        print('ret:'," ".join(ret))
        return " ".join(ret)

    def search(self,text):
        print('in search')
        print('cut sent:',TagsTrie.cut_sent(text))
        print('iter',self.tree.iter(TagsTrie.cut_sent(text)))
        for match in self.tree.iter(TagsTrie.cut_sent(text)):
            yield match


def tag_documents(docs,tags_trie):
    '''
    [({'docID':'xxxxx'},'全部分类敏捷项目管理'),...]
    '''
    res=[]
    for (meta,text) in docs:
        tags=list()
        for _,tag in tags_trie.search(text):
            tags.append(tag)
        meta['TAGS']=tags

        res.append(meta)
    return res

if __name__ =='__main__':
    tags=['项目','软件','项目计划']
    text='项目项目软件'
    tags_trie=TagsTrie(tags)
    for _,tag in tags_trie.search(text):
        print(_,tag)

    '''
     4 项目
     8 项目
     12 软件
     '''
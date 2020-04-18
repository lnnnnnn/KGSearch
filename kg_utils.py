import pymongo
import json ,os,re
def get_collections(cols,config='config.json'):
    config=json.load(open(config))
    client=pymongo.MongoClient('mongodb://%s' % config.get('host',""))
    db=client.get_database(config.get('db',""))
    # print(config.get(cols,cols))
    return [db[config.get(col,col)] for col in cols]


class Cache:
    def __init__(self,func):
        self.memory={}
        self.func=func

    def get(self,*keys):
        '''

        :param keys: ('iot',100,1,1,") type:<class 'tuple'>
               *keys iot 100 1 1 1   type()报错
        :return:
        '''
        if keys not in self.memory:
            self.memory[keys]=self.func(*keys)
        return self.memory[keys]
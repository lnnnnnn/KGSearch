from rerank import train,test
logfile ='log.txt'
dataset ='data.txt' #xlsx 转txt

if __name__=='__main__':
	train(logfile,dataset)
	test(logfile,dataset)
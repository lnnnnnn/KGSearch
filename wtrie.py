import os, sys, time, re, math
import string
#得到字典词在原句中的索引
class WTrie:
	def __init__(self, wdict):
		self.dic = {}
		self.num_items = 0
		self.sum_items = 0
		self.merge_chars = set(string.digits+string.ascii_letters)
		for k, v in wdict if type(wdict) is type([]) else wdict.items(): 
			self.__setitem__(k, v)

	def cut_sent(self, sent):
		ret = [' ']
		for c in sent:
			if c in self.merge_chars and ret[-1][-1] in self.merge_chars:
				ret[-1] += c
				continue
			ret.append(c)
		return ret[1:]

	def search(self, sent):
		'''
		sent:iot技术解决方案
		scut:['iot','技','术','解','决','方','案']
		{(0,3):'iot',(3,4):'技',(3,5):'技术',(3,9):'技术解决方案',...解决 ，方案， 解决方案}
		:param sent:
		:return:
		'''
		scut = self.cut_sent(sent)
		ret = {};  ii = 0#记录当前处理的query位置
		for i, ch in enumerate(scut):
			z = self.dic; jj = 0
			for j, c in enumerate(scut[i:]):
				if c not in z: break
				z = z[c]; jj += len(c)#当前字符c的字典
				#{'中':{'案':{0:'案中案'，1:1}},'案':{'例':{0:'案案例',1:1}}}
				if 0 in z: ret[ii,ii+jj] = z[0]
			ii += len(ch)
		return ret

	def build_fail(self):
		Q = [self.dic]; qh = 0
		self.dic[9] = None
		while qh < len(Q):
			z = Q[qh];  qh += 1
			for ch, next in z.items():
				if type(ch) is type(0): continue
				if z != self.dic:
					p = z[9]
					while p:
						if ch in p:
							next[9] = p[ch]
							break
						p = p[9]
					if p is None: next[9] = self.dic
				else: next[9] = self.dic
				Q.append(next)
		#print('fail-pointer built')
		self.fail_built = True

	def fast_search(self, sent):
		if not self.fail_built: self.build_fail()
		scut = self.cut_sent(sent)
		ret = {};  ii = 0
		z = self.dic
		for ch in scut:
			ii += len(ch)
			while not ch in z and z != self.dic: z = z[9]
			if not ch in z: continue
			z = z[ch];  p = z
			while p != self.dic:
				if 0 in p: ret[(ii-len(p[0]),ii)] = p[0]
				p = p[9]
		return ret

	def __setitem__(self, item, cnt):
		z = self.dic
		icut = self.cut_sent(item)
		for c in icut: z = z.setdefault(c, {})
		if 0 not in z: 
			self.num_items += 1
			self.sum_items += cnt
		else: self.sum_items += cnt - z[1]
		z[0], z[1] = item, cnt
		self.log_sum_items = math.log(self.sum_items) if self.sum_items > 0 else -100
		self.fail_built = False

	def __getitem__(self, w):
		z = self.dic
		wcut = self.cut_sent(w)
		for c in wcut:
			if c not in z: return 0
			z = z[c]
		return z.get(1, 0)

	def __delitem__(self, w):
		z = self.dic
		wcut = self.cut_sent(w)
		for c in wcut:
			if c not in z: return None
			z = z[c]
		if 0 in z:
			self.num_items -= 1
			self.sum_items -= z[1]
			self.log_sum_items = math.log(self.sum_items) if self.sum_items > 0 else -100
			del z[0]; del z[1]
			self.fail_built = False

if __name__ == '__main__':
	wtrie = WTrie({'you are':1, 'are':23, '我们':12, '我们de飞':3})

	#{'you': {' ': {'are': {0: 'you are', 1: 1}}}, 'are': {0: 'are', 1: 23}, '我': {'们': {0: '我们', 1: 12, 'de': {'飞': {0: '我们de飞', 1: 3}}}}}

	print(wtrie.dic)
	print(wtrie.search('you are我们de飞机'))
	print(wtrie.fast_search('you are我们de飞机'))
	print('done')
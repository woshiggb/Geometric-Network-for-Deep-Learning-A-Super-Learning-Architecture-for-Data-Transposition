
	# 进行计算
AN = 100
qpdoto = 1
feiquxiands = 0 #非曲线ds,作用是选择耗材最小的函数模型
Tanh = 0

	
import random as r
import math
import threading


cntstod = 1
if (cntstod):
	#import numpy as math
	from mpmath import mp
	# 设置精度
	mp.dps = 200

print_ = 1
rn2 = 0

def is_zero_beto(num1 , num2):
	if (num2 == 0):
		return float("inf")
	return num1 / num2

fto = r.random
rand_to = [-1 , 1]
def random():
	return mp.mpf(str(fto()))
r.random = random
class model:
	def __init__(self,N,new_num , atrs =  r.randint(0 , 3)):
		if (atrs == 0):
			a = r.randint(1 , 3)
		else:
			a = atrs
		if (a == 1):
			Be_to = 0.5
			#if (new_num <= 20):
			#	Be_to = 1/math.pow(new_num+1,3)
			#	print(Be_to)
			#	# print(Be_to) 
			#self.Num = [[0,r.random()*rand_to[r.randint(1 , 2) - 1]]] + [[i/new_num,r.random()*rand_to[r.randint(1 , 2) - 1]] for i in range(1,new_num+1)] # i/new_num
			self.Num = [[0,r.random()*rand_to[r.randint(1 , 2) - 1]]] + [[r.random() * rand_to[r.randint(0 , 1)],r.random()*rand_to[r.randint(1 , 2) - 1]] for i in range(1,new_num+1)] # i/new_num
			#for i in range(1,new_num):
			#	self.Num[i][1] = self.Num[i-1][1] + self.Num[i][1]
			#max_in , min_in = 0 , self.Num[0][1]
			#for i in self.Num:
			#	if (i[1] > max_in):
			#		max_in = i[1]
			#	elif (min_in > i[1]):
			#		min_in = i[1]
			#max_Go = 1/(max_in - min_in)
			#self.Num = [[i[0] , i[1]-min_in]]
		elif (a == 2):
			self.Num = [[i , r.random()] for i in range(0 , new_num+1)]
			self.mean_go = sum([i[1] for i in self.Num])/2
			self.Num[0][1] = -1 + self.Num[0][1] / self.mean_go
			for i in range(len(self.Num)): self.Num[i][1] = self.Num[i-1][1] + self.Num[i][1] / self.mean_go
		elif (a == 3):
			self.Num = [[0 , r.random()]] + [[i / new_num , r.random()] for i in range(1 , new_num+1)]
		else:
			self.Num = [[i , r.random()] for i in range(0 , new_num+1)]
			self.mean_go = sum([i[1] for i in self.Num])/2
			self.Num[0][1] = 1 - self.Num[0][1] / self.mean_go
			for i in range(len(self.Num)): self.Num[i][1] = self.Num[i-1][1] - self.Num[i][1] / self.mean_go	
		if (new_num == 0):
			self.Num = []
		new_num += 1
		self.Nuum = [[self.Num[i][0],self.Num[i+1][0],self.Num[i][1],self.Num[i+1][1],(self.Num[i+1][1] - self.Num[i][1]) / self.is_zero_(self.Num[i+1][0] - self.Num[i][0])] for i in range(new_num-1)]
		self.new_num,self.N,self.Num_in = new_num,N,new_num

	def Numtobeinn(self , a = 1.5):
		self.Num_betgo = [[math.pow( self.Num[i] / self.Num[i-1] , a ) , abs(self.Num[i] - self.Num[i-1]) / (self.Num[i] - self.Num[i-1]) ] for i in range(1 , self.Num_in)]
		return self.Num_betgo

	def forward(self,cin):
		if (cin != 1): cin = cin
		self.Num_in = len(self.Nuum)
		if (self.Num_in == 0): return cin
		IN = self.find_one(cin)
		if (cntstod):
			#return self.Nuum[IN][2] + mp.mpf(f'{cin - self.Nuum[IN][0]}') * mp.mpf(f'{self.Nuum[IN][4]}')
			return self.Nuum[IN][2] + mp.mpf(f'{cin - self.Nuum[IN][0]}') * mp.mpf(f'{self.Nuum[IN][4]}')
		else:
			return self.Nuum[IN][2] + (cin - self.Nuum[IN][0]) * self.Nuum[IN][4]


	def backward(self,cin , out):
		if (cin == 'T'):
			well = 1
			#print(f"GeR_New T is treu:{well}") 
			Ee_a , Ee_b = out[0][0] , out[0][1]
			out = out[1]
			self.Num_in = len(self.Num)
			self.Num = [[self.Num[i][0] * Ee_a + out[i][0] * Ee_b, self.Num[i][1] * Ee_a + out[i][1] * Ee_b ] for i in range(self.Num_in)] 
			#self.Num.sort()
			self.Num.sort(key = lambda x:x[0])
			self.Nuum = [[self.Num[i][0],self.Num[i+1][0],self.Num[i][1],self.Num[i+1][1],(self.Num[i+1][1] - self.Num[i][1]) / self.is_zero_(self.Num[i+1][0] - self.Num[i][0])] for i in range(len(self.Num) - 1)]
			if (len(self.Num) == 1):
				self.Nuum .append([cin , cin , out , out , 1])
			#for i in range(self.new_num - 1): self.Nuum[i] += [(self.Nuum[i][3] - self.Nuum[i][2]) / (self.Nuum[i][1] - self.Nuum[i][0])]
			return 0
		#elif (cin == 'T' and [self.Num[i][0] for i in range(len(self.Num))] != [i[0] for i in out[1]]):
		#	well = 1

		#	Ee_a = out[0][0]
		#	Ee_b = out[0][1]
		#	if (Ee_a > Ee_b): return 0
		#	else:
		#		if (r.random() >= Ee_a):
		#			self.Num = Ee_a
		#	self.Num.sort(key = lambda x:x[0])
		#	self.Nuum = [[self.Num[i][0],self.Num[i+1][0],self.Num[i][1],self.Num[i+1][1],(self.Num[i+1][1] - self.Num[i][1]) / self.is_zero_(self.Num[i+1][0] - self.Num[i][0])] for i in range(len(self.Num) - 1)]
		#	if (len(self.Num) == 1):
		#		self.Nuum .append([cin , cin , out , out , 1])
		#	#for i in range(self.new_num - 1): self.Nuum[i] += [(self.Nuum[i][3] - self.Nuum[i][2]) / (self.Nuum[i][1] - self.Nuum[i][0])]
		#	return 0

		if (len(self.Num) == 0):
			self.Num = [[cin , out]]
		#print(cin)
		self.I = self.ddf_find_in(cin) 
		self.Num_in = len(self.Nuum)
		self.Num.append([cin , out])
		#if (self.I == max(self.Num_in - 1 , -1)): self.Num.append([ cin , out ])
		#elif (self.Num[self.I][0] == cin): self.Num[self.I] , self.Num_in = [  cin  ,  out  ] , self.Num_in - 1 
		#elif (cin < self.Num[0][0]): self.Num = [[cin , out]] + self.Num
		#else: self.Num = self.Num[0:self.I + 1] + [[ cin,   out     ]] + self.Num[self.I : self.Num_in]
		self.Num.sort(key=lambda x: x[0])
		self.Nuum = [[self.Num[i][0],self.Num[i+1][0],self.Num[i][1],self.Num[i+1][1],(self.Num[i+1][1] - self.Num[i][1]) / self.is_zero_(self.Num[i+1][0] - self.Num[i][0])] for i in range(len(self.Num) - 1)]
		if (len(self.Num) == 1):
			self.Nuum .append([cin , cin , out , out , 1])
		return 0

	def kiil_list(self, num , lnum):
		XS = 0
		for ii in range(1,lnum):
			if (num[ii] == num[ii-1]):
				num.pop(ii-XS) 
				XS += 1
		return num
				
	def cz(self,nu=0):
		self.Nuum = [[self.Num[i][0],self.Num[i+1][0],self.Num[i][1],self.Num[i+1][1],(self.Num[i+1][1] - self.Num[i][1]) / self.is_zero_(self.Num[i+1][0] - self.Num[i][0])] for i in range(len(self.Num) - 1)]
		return 0

	def backward_(self,cin,out):
		if (cin == 'T'):
			well = 1
			print(f"GeR_New T is treu:{well}") 
			self.Nuum = [[(self.Nuum[i][j] + out[i][j]) / 2 for j in range(4)] for i in range(self.new_num)]
			for i in range(self.new_mdn): self.Nuum[i] += [(self.Nuum[i][3] - self.Nuum[i][2]) / (self.Nuum[i][1] - self.Nuum[i][0])]
			return 0

		self.Num_in = len(self.Nuum)
		IN = self.find(cin)
		if (cin == self.Nuum[IN][0]):
			self.Nuum[IN] = [cin , self.Nuum[IN][1] , out , self.Nuum[IN][3] , (self.Nuum[IN][3] - out) / self.is_zero_(self.Nuum[IN][1] - cin)]
		if (self.Num_in == 0):
			self.Nuum = [[cin,cin,out,out,1]]
		elif (IN == 0):
			self.Nuum[IN] = [cin,self.Nuum[IN][0],out,self.Nuum[IN][2],(self.Nuum[IN][2] - out) / self.is_zero_(self.Nuum[IN][0] - cin)]
		elif (self.Num_in - 1 == IN):
			self.Nuum[IN][1],self.Nuum[IN][3],self.Nuum[IN][4] = cin,out,(out - self.Numm[IN][2]) / self.is_zero_(cin - self.Nuum[IN][0])
		else:
			self.Nuum[IN][1],self.Nuum[IN][3],self.Nuum[IN][4] = cin,out,(out - self.Nuum[IN][2]) / self.is_zero_(cin - self.Nuum[IN][0])
			self.Nuum = self.Nuum[0:IN + 1] + [cin,self.Nuum[IN+1][0],out,self.Nuum[IN+1],(self.Nuum[IN+1][2] - out) / self.is_zero_(self.Nuum[IN+1][0] - cin)]
		return None

	def find(self,num):
		self.Num_in = len(self.Nuum)
		self.ams , self.bma = 0, self.Num_in - 1
		self.The  = None 
		while (self.ams <= self.bma):
			self.INfd = (self.ams + self.bma) // 2
			if (self.Nuum[self.INfd][0] <= num and self.Nuum[self.INfd][1] > num): return self.INfd
			elif (self.Nuum[self.INfd][0] > num): self.bma = self.INfd - 1
			else: self.ams = self.INfd + 1
			self.The = self.INfd

		return max(self.INfd,0)

	def find2_(self,num):
		self.Num_in = len(self.Nuum)
		self.ams , self.bma = 0, self.Num_in - 1
		self.INfd = -1
		while (self.ams < self.bma):
			self.INfd = (self.ams + self.bma) // 2
			if (self.Nuum[self.INfd][0] <= num): return self.INfd
			elif (self.Nuum[self.INfd][0] > num): self.bma = self.INfd - 1
			else: self.ams = self.INfd + 1


		return self.INfd - 1

	def find_one(self,num):
		self.Num_in = len(self.Nuum)
		for i in range(self.Num_in):
			if (self.Nuum[i][0] <= num and self.Nuum[i][1] > num): 
				return i
		return self.Num_in - 1

	def ddf_find_in(self,num_in):
		for i in range(len(self.Num)):
			if (self.Num[i][0] == num_in): return i
			elif (num_in < self.Num[i][0]):  return i-1
			#else :print("""f">{?}" """)
		return 0

	def new_model_be(self,N=1):
		if (N == False):
			self.Nuum = []
		self.Num_in = len(self.Nuum)
		return N

	def is_zero_(self,NUM):
		if (NUM == 0):
			return 1
		return NUM

	def sort(self,N):
		self.Len_n = len(N)
		if (self.Len_n == 2):
			if (N[1] < N[0]):
				return [N[1] , N[0]]
		if (self.Len_n < 3):
			return N
		self.mean = sum(N) /  self.Len_n
		self.a , self.b = [i for i in N if i < self.mean],[j for j in N if j >= self.mean]

		return self.sort(self.a) + self.sort(self.b)

class modelkl:
	def __init__(self,N,rand_=2,atrs = r.randint(0 , 3)):
		self.N_len,self.N,self.K =  len(N),N,1
		self.modelw = [[[model(AN,rand_,atrs) for c in range(N[a+1])] for b in range(N[a])] for a in range(self.N_len-1)]
		self.out_modelwto = [[[0 for c in range(N[a+1])] for b in range(N[a])] for a in range(self.N_len - 1)]
		#self.modelw_f = [[model(AN,rand_) for j in range(N[i])] for i in range(self.N_len)]
		self.modelw_f = [[self.porelr for j in range(N[i])] for i in range(self.N_len)]
		#self.modelw_f = [[self.porelr for j in range(N[i])] for i in range(self.N_len - 1)]
		#self.modelw_f += [[model(AN , rand_ , 4) for i in range(N[self.N_len - 1])]]
		
		self.OUT_f = [model(AN,0) for i in range(self.N[self.N_len - 1])]
		self.out_ = [[0 for j in range(N[i])] for i in range(self.N_len)]

	def forward(self,cin_num):
		outS = [i.copy() for i in self.out_]
		outS[0] = cin_num
		for Ia in range(self.N_len-1):
			for b in range(self.N[Ia]):
				#outS[Ia][b] = self.modelw_f[Ia][b].forward(outS[Ia][b])
				outS[Ia][b] = self.modelw_f[Ia][b](outS[Ia][b])
			for b in range(self.N[Ia]):
				for v in range(self.N[Ia+1]):
					#self.modelw[a_cin]
					outS[Ia+1][v] += self.modelw[Ia][b][v].forward(outS[Ia][b])
		#for bvs in range(self.N[self.N_len - 1]):
		#	outS[self.N_len - 1][bvs] = self.modelw_f[self.N_len - 1][bvs].forward(outS[self.N_len - 1][bvs])
		self.out_cin = outS[self.N_len  -     1].copy()
		for Ia in range(self.N[self.N_len - 1]):
			outS[self.N_len - 1][Ia] = self.OUT_f[Ia].forward(outS[self.N_len - 1][Ia])

		self.outS = outS
		self.cin_ = cin_num
		return self.outS[self.N_len - 1]

	def backward(self,numn):
		self.out_cinn = self.out_cin
		#print("OBEK" , self.out_cinn , numn)
		for i in range(self.N[self.N_len - 1]):
			#print(self.OUT_f[i].forward(self.out_cinn[i]) , numn[i])
			self.modelw[i].find_qxds()
			#print(self.OUT_f[i].forward(self.out_cinn[i]))
		return 1


	def backward(self, num , num_need ,  a_cin , b):
		self.be_num = num_need - num
		self.bnto = sum([self.out_modelwto[a_cin][a][b] for a in range(len(self.out_modelwto[a_cin]))])
		self._ = [self.out_modelwto[a_cin][a][b] / self.bnto * self.be_num for a in range(self.N[a_cin]) ]

		for i in range(self.N[a_cin]):
			self.IN_dtea = self.modelw[a_cin][i][b].Nuum[self.modelw[a_cin][i][b].find_one(self._[i])][4]
			self.modelw[a_cin][i][b] += self._ / self.IN_dtea * anumto
			#self.backward(self , self. , num_need)


	def porelr(self,numr):
		return numr




class modeldwo:
	def __init__(self , n , nnum , rn2 = 0):
		self.E = None
		self.N = [n , nnum]
		self.K = 1
		self.numtr = 0.5
		self.rn_1_nad_no1 = r.randint(1 , 1)
		self.model_and = [[model(AN , rn2 , self.rn_1_nad_no1) for j in range(n[-1])] for i in range(n[-2])]
		self.old_model_and = [i.copy() for i in self.model_and]
		self.Retun = [[model(AN , rn2 , self.rn_1_nad_no1) for i in range(n[-1])] for j in range(n[-2])]
		self.N_len = len(n)
		self.Cinceng = [[[model(AN , nnum , self.rn_1_nad_no1) for c in range(n[a+1])] for b in range(n[a])] for a in range(len(n) - 1)]
		if (qpdoto == 1):
			self.Cintods = [[model(AN , nnum , 2) for j in range(n[i])] for i in range(len(n))]
		self.out_byum = [0 for u in range(n[1])]
		self.Cinlist = [0 for i in range(n[0])]
		self.sekf_outs = [[0 for j in range(i)] for i in n]
		#self.modelw_f = [[model(AN,rand_) for j in range(N[i])] for i in range(self.N_len)]
		#self.modelw_f = [[self.porelr for j in range(N[i])] for i in range(self.N_len)]
		#self.modelw_f = [[self.porelr for j in range(N[i])] for i in range(self.N_len - 1)]
		#self.modelw_f += [[model(AN , rand_ , 4) for i in range(N[self.N_len - 1])]]
		
		self.OUT_f = [model(AN,0) for i in range(self.N[1])]

	def forward(self , numin , cnt = 0):
		sekf_outs = [i.copy() for i in self.sekf_outs]
		sekf_outs[0] = numin
		for a in range(self.N_len-2):
			if (qpdoto):
				sekf_outs[a] = [math.tanh(self.Cintods[a][i].forward(sekf_outs[a][i])) for i in range(self.N[0][a])]
			for b in range(self.N[0][a+1]):
				for c in range(self.N[0][a]):
					if (Tanh == 1):
						sekf_outs[a+1][b] += self.Cinceng[a][c][b].forward(sekf_outs[a][c])
					else:
						sekf_outs[a+1][b] += self.Cinceng[a][c][b].forward(sekf_outs[a][c])
		if (cnt == 1):
			print(sekf_outs)
		self.Cinlist = sekf_outs[self.N_len - 2]
		self.gxz = [[self.model_and[j][i].forward(sekf_outs[self.N_len - 2][j]) for j in range(self.N[0][self.N_len - 2])] for i in range(self.N[0][self.N_len - 1])]
		self.out_byum = [sum(self.gxz[i]) for i in range(self.N[0][self.N_len - 1])]
		self.by = self.out_byum
		#if (qpdoto):
		#	self.out_byum = [self.Cintods[-1][i].forward(self.out_byum[i]) for i in range(self.N[0][self.N_len - 1])]
		return self.out_byum

	def backward(self ,  b_need):
		self.eEtrs = [b_need[i] - self.out_byum[i] for i in range(self.N[0][self.N_len - 1])]
		self.Re = [[self.Retun[j][i].forward(self.eEtrs[i]) for j in range(self.N[0][self.N_len - 2])] for i in range(self.N[0][self.N_len - 1])]
		self.Re_Sum = [sum(self.Re[i]) for i in range(self.N[0][-1])]
		self.Re = [[self.Re[i][j] / self.error_to_1_and_90(self.Re_Sum[i])  for j in range(self.N[0][self.N_len - 2])] for i in range(self.N[0][self.N_len - 1])]
		self.gxz_ = [[self.gxz[i][j] + self.Re[i][j] for j in range(self.N[0][self.N_len - 2])] for i in range(self.N[0][self.N_len - 1])]
		self.out_byum = [sum(self.gxz_[i]) for i in range(self.N[0][self.N_len - 1])]
		self.gxz_needbe = [[self.gxz_[i][j] / self.error_to_1_and_90(self.out_byum[i]) * self.eEtrs[i] for j in range(self.N[0][self.N_len - 2])] for i in range(self.N[0][self.N_len - 1])]
		for i in range(self.N[0][self.N_len - 1]):
			for j in range(self.N[0][self.N_len - 2]):
				self.INds = self.model_and[j][i].find_one(self.Cinlist[j])
				self.model_and[j][i].backward(self.Cinlist[j]  , self.gxz[i][j] + self.gxz_needbe[i][j])

	#def backward(self , b_need):
	#	self.Eetrs = [b_need[i] - self.out_byum[i] for i in range(self.N[0][self.N_len - 1])]
	#	for i in range(len(self.N[0])):
	#		self.Cintods[0].backwarad(self.Eetrs)

	#def backward(self , b_need):
	#	self.eEs = [self.out_byum[i] - b_need[i] for i in range(self.N[0][self.N_len - 1])]
	#	for i in range(self.N[0][-1]):
	#		self.Cintods[-1][i].backward(self.by[i] , self.eEs[i])
	#	return 0

	def error_to_1_and_90(self , numers):
		if (numers == 0):
			return 1          
		return numers

#MODELS = models(100,[2 , 2 ,1] , rand_ = 1000)

#Train_model = train_tio_models(10 , [2 , 5, 1] , 10 , rand_ = 1000)
#Train_model.train([[1 , 2]] , [[1]] , 10  , 100 , 114)



#MODELK = modelkl([5,5,5,5,5] , 100)
#print(MODELK.forward([1,1,1,1,1]))
#MODELK.backward([1,2,3,4,5])

def f(n):
	return n**2

class train__:
	def __init__(self,bun=1000,f=f , a=0 ,   vb=1):
		self.vf , self.modek = f , modelkl([1,50,50,50,6,1])
		self.bun ,self.a , self.vb = bun , a , vb

	def go_bund(self):
		for i in range(self.bun):
			rn = r.random()*r.randint(self.a,self.vb)
			self.modek.forward([rn])
			print('E_?r:' ,  abs(self.modek.forward([rn])[0] - self.vf(rn)))
			self.modek.backward([self.vf(rn)])
			print('E:' ,  abs(self.modek.forward([rn])[0] - self.vf(rn)))

			#print(self.modek.OUT_f[0].Nuum , self.modek.OUT_f[0].Num)
			import time
			time.sleep(0.01)

#train__ = train__(50)
#train__.go_bund()

def trains01():

	cin  , out = [[1]] , [[2]]
	MODELOW = modeldwo([1 , 2 , 2 , 2 , 2 , 2 , 1] , 3)
	MODELOW.forward([1])
	MODELOW.backward([2])
	print(MODELOW.forward([1.5]))
	MODELOW.backward([2])
	print(MODELOW.forward([1]))
	def f(n):
		return n**2


class models:
	def __init__(self,N1,N2,N3 = r.randint(0 , 3),meax= 5 , rand_ = 10):
		print(N2)
		self.mdn = [modeldwo(N2[0] , N2[1]) for i in range(N1)]
		self.rand_ = rand_
		self.a_e = max(int(meax * N1/ 100),1)
		self.N1,self.N2 , self.pan , self.bc = N1 , N2 , N1 // 2 , N1 // 3

	def forward_one(self,cinsn,outsn,trainp,maxn , rand_num = 10 , prints = 1):
		self.to_gosnum = rand_num
		self.bxc = self.bc
		cin_,out_ = [i.copy() for i in cinsn] , [outsn[j].copy() for j in range(len(outsn))]
		self.ind = [cin__.copy() for cin__ in cinsn] , [out__.copy() for out__ in outsn]
		rand_in = [[r.random() , i] for i in range(len(cin_))]
		rand_in = self.sort_in(rand_in)
		rand_get_in  = r.randint(len(cin_) // 2,len(cin_)) - 1
		cinsn_ = [cin_[i].copy() for i in rand_in][0:rand_get_in]
		outsn_ = [out_[j].copy() for j in rand_in][0:rand_get_in]
		list_ = [i for i in range(self.N2[len(self.N2) - 1])]
		a_e = int(self.a_e * trainp)
		begetnum =  2 * a_e * (a_e + 0.5)
		self.jow_num = [(2*a_e - i) / begetnum for i in range(0 , 2*a_e)]

		threads = []
		self.Cnt = 0
		self.Need = self.N1 * len(cinsn_)
		for i in range(self.N1):
			#print(f'\r处理事件:{round(i/len(outsn_) * 100 , 2)}%   ', end='', flush=True) 
			self.f_to_train(cinsn_ , outsn_ , i)
			#threads.append(threading.Thread(target = self.f_to_train , args = (cinsn_ , outsn_ , i)))
			#threads[-1].start()

		#for i in threads:
		#	i.join()

		#print('')	

		for pi in range(self.N1):
			self.mdn[pi].E = 0
			for j in range(len(cin_)):
				outs_in_i = self.mdn[pi].forward(cin_[j])
				self.mdn[pi].E += sum([abs(outs_in_i[i] - out_[j][i])  for i in range(self.mdn[pi].N[0][self.mdn[pi].N_len - 1]) ] )
			self.mdn[pi].E /= len(cin_)
			#self.mdn[pi].OTO = self.mdn[pi].E * 1/(1+sum([sumD([abs(c[4]) for c in b.Nuum]) for a in self.mdn[pi].model_and for b in a]))
		self.new_mdn = self.a_sort(self.mdn)
		#print('Data is over')

		if ( print_ ):
			if (prints == 1): print([f"E and Ks:({self.new_mdn[i].E} , {self.new_mdn[i].K})" for i in range(len(self.new_mdn))][0:50])
		for i in range(self.bxc-1,  len(self.new_mdn)):

			self.a,self.b = max (0,i - a_e)  ,   min(self.N1-1,i + a_e)
			self.gl_toab = self.jow_num[self.a:self.b + 1].copy()
			if (self.b - self.a !=0):
				rn = int(self.rand_to_getnum(self.gl_toab , 2))
			else:
				rn = self.a
			self.Ee_a , self.Ee_b = -1, -1
			if (self.new_mdn[i].E == 0):
				print("Over of one")
				return 0
			if (self.new_mdn[rn].E == 0):
				self.Ee_b = 0
			if (self.Ee_a == None or self.Ee_a == self.Ee_b and self.Ee_b != 0):
				#try:
				self.Ee_a , self.Ee_b = -math.log10(self.new_mdn[i].E) / max(math.sqrt(self.new_mdn[i].K) / (1 + self.new_mdn[i].K **2 / 26) , 1) , -math.log10(self.new_mdn[rn].E)/ self.new_mdn[rn].K  / max(math.sqrt(self.new_mdn[rn].K) / (1 + self.new_mdn[rn].K **2 / 26) , 1)
				#except (OverflowError, ValueError) as e:
				#	cront = 1
			sum_Er = self.Ee_a + self.Ee_b
			if (sum_Er == 0):
				print("OVER")
				return 1
			self.jz_a,self.jz_b = self.Ee_a/ sum_Er  , self.Ee_b / sum_Er 
			if (self.jz_a > self.jz_b or (self.jz_a == self.jz_b and self.new_mdn[i].K > self.new_mdn[i].K)):
				Ir = i
			else:
				Ir = rn
			for a in range(self.mdn[i].N_len-2):
				for b in range(self.mdn[i].N[0][a]):
					#if (qpdoto):
					#	self.mdn[i].Cintods[a][b].backward('T',([self.jz_a , self.jz_b] , self.new_mdn[Ir].Cintods[a][b].Num))
					if (qpdoto):
						self.new_mdn[i].Cintods[a][b].backward('T',([self.jz_a , self.jz_b] , self.new_mdn[Ir].Cintods[a][b].Num))
					for c in range(self.mdn[i].N[0][a+1]):

						#print(self.new_mdn[Ir].modelw[a][b][c])
						#try:
						self.new_mdn[i].Cinceng[a][b][c].backward('T' , ([self.jz_a , self.jz_b] , self.new_mdn[Ir].Cinceng[a][b][c].Num ))
						#except:
						#print("No num")

			for a in range(self.mdn[i].N[0][-2]):
				for b in range(self.mdn[i].N[0][-1]):
					self.mdn[i].Retun[a][b].backward("T" , ([self.jz_a , self.jz_b] , self.new_mdn[Ir].Retun[a][b].Num))

			self.new_mdn[i].K =   max(int(self.new_mdn[i].K * self.jz_a + self.new_mdn[rn].K * self.jz_b) , 1)
			for i in range(0 , self.bxc ): self.mdn[i].K += 1
			self.new_mdn[i].old_model_and = self.new_mdn[i].model_and.copy()
			self.new_mdn[i].model_and = [[model(AN , rn2 , None) for j in range(self.N2[0]	[self.mdn[i].N_len - 1])] for i in range(self.N2[0][self.mdn[i].N_len - 2])]
		self.mdn = self.new_mdn[0:self.bxc] + self.new_mdn[self.bc:self.pan] + [modeldwo(self.N2[0] , self.N2[1]) for i in range(self.N1 - self.pan)]
		return 0

	def f_to_train(self , Cinsd , Outsn , j_in):
		for i in range(len(Cinsd)):
			#self.Cnt += 1 #Cnt
			#print(f"""\r{int(self.Cnt/self.Need*100)}%   """ , end = '' , flush = True)
			self.mdn[j_in].forward(Cinsd[i].copy())
			self.mdn[j_in].backward(Outsn[i].copy())
	def forward_0(self , cin):
		return self.mdn[0].forward(cin)
		#
		#
		#
		#
		#

	def train(self,cinns,outnts,trainp,Epochs1,Epochs2,meaxn , prints = 1):
		for epoch1 in range(Epochs1):
			for epoch2 in range(Epochs2):
				self.forward_one(cinns,outnts,trainp,meaxn , prints = prints)
				if ( prints ):
					print(f"Epochs:{epoch1} , {epoch2}")

	def a_sort(self,num):
		self.Lne_n = len(num)
		self.lenn = self.Lne_n
		for i in range(self.lenn-1):
			for j in range(self.lenn-i-1):
				if (num[j].E > num[j+1].E):
					num[j] , num[j+1] = num[j+1] , num[j]
		return num

	def sort_in(self,num):
		self.Lne_n = len(num)
		self.lenn = self.Lne_n
		for i in range(self.lenn-1):
			for j in range(self.lenn-i-1):
				if (num[j][0] > num[j+1][0]):
					num[j] , num[j+1] = num[j+1] , num[j]
		return [i[1] for i in num]

	def rand_to_getnum(self, numberS , numgointo):
		minS_in = 0
		for i in range(1 , len(numberS)):
			for j in range(self.to_gosnum):
				numberS[i] -= r.random() * (10 ** (numgointo - 1))
				if (numberS[i] < numberS[i - 1]):
					minS_in = i
		return minS_in

	def get_new_model_and(self , Cin , Out):
		for i in self.mdn:
			for j in range(len(Cin)):
				i.backward(Cin[j] , Out[j])
		return 0

	def sort(self,N):
		self.Len_n = len(N)
		if (self.Len_n == 2):
			if (N[1].E > N[0].E):
				return [N[1] , N[0]]
		if (self.Len_n < 3):
			return N
		self.mean = sum([i.E for i in N]) /  self.Len_n
		self.a , self.b = [i for i in N if i.E < self.mean],[j for j in N if j.E >= self.mean]
		return self.sort(self.a) + self.sort(self.b)

	def pore(self,n):
		return n

class Train2:
	def __init__(self , Nlist , Nto = 2):
		self.model = [models(*Nlist) for i in range(Nto)]
		self.Nlist , self.Nto =  Nlist , Nto

	def forward_train_one(self , Cin , Out , num , peint):
		printto = peint

		for j in self.model:
			j.train(Cin , Out , 1 , 1 , num , 19 , prints = printto)
	def train_one(self , Cin , Out , num1 , num2):
		for i in range(num1):
			self.forward_train_one(Cin , Out , num2 , 0)
			self.model[0].mdn[-2] = self.model[1].mdn[0]
			for j in range(1 , self.Nto - 1):
				asint = min(len(self.model[j+1].mdn)//2 - 1 , 0)
				self.model[j].mdn[self.Nlist[0] - asint - 1:self.Nlist[0] - 1] = self.model[j+1].mdn[0:min(len(self.model[j+1].mdn)//2 - 1 , 0)]
			for i in range(self.Nto-1):
				for j in range(self.Nto-i-1):
					if (self.model[j].mdn[0].E > self.model[j+1].mdn[0].E):
						self.model[j] , self.model[j+1] = self.model[j+1] , self.model[j]
			#for j in range(1 , self.Nto - 1):
			#	asint = min(len(self.model[j+1].mdn)//2 - 1 , 0)
			#	self.model[j].mdn[self.Nlist[0] - asint - 1:self.Nlist[0] - 1] = self.model[j+1].mdn[0:min(len(self.model[j+1].mdn)//2 - 1 , 0)]
			print([f'E:(error):{i.E}' for i in self.model[0].mdn[0:50]])
			print([i.mdn[0].E for i in self.model])

		self.model[-1] = models(*self.Nlist)


def sumD(Num):
	if (Num == []):
		return 0
	return sum(Num)		

#Cinmax = 10
#Outtot = 10
#trains = Train2([10 , [[Cinmax ,100,1, Outtot] , 10]] , 10)
#Cinssd = [[r.random() for j in range(Cinmax)] for i in range(10)]
#Outto = [[r.random() for j in range(Outtot)] for i in range(len(Cinssd))]
#trains.train_one(Cinssd , Outto , 10 , 10 )
#trinsd = trains.model[0].mdn[0].forward(Cinssd[0])


#def trains2():
	#print(MODELOW.forward([2]))
	#MODELS = models(50 , [[1 , 1] , 2] , 1)
	#cin = [[1/(i+1) , 1/(j+1)] for j in range(10) for i in range(100) ] 
	#out = [[cin[i][0] * cin[i][0]] for i in range(1000)]
	#MODELS.train(cin , out , 2 , 1 , 10 , "NOE?YES!")

	#MODELK.forward([1,1,1,1,1])
	#MODELK.backward([1,2,3,4,5,6])
	#MODELK.forward([1,2,3,4,5,6])
	#MODELK.backward([5,4,3,2,1])
	#MODELK.forward([1,6,3,7,8])
	#MODELK.backward([5,4,8,2,9])

	#print(MODELK.forward([1,6,3,7,8]))
	#print(MODELK.forward([1,2,3,4,5,6]))
	#print(MODELK.forward([5,1,2,3,2]))
	#print(MODELK.OUT_f[0].Nuum)

	#cin = [	]
	#cin_  = [[[r.random() , r.random()] for i in range(10)] for j in range(10)]
	#for i in cin_:
	#	cin += i
	#out = [[cin[i][0] * cin[i][1]] for i in range(len(cin))]
	#Ns_f1 = 1
	#timre_num = 100
	#for i in range(timre_num): 
	#	for j in range(Ns_f1):
	#		MODELS.mdn[0].forward(cin[i])
	#		MODELS.mdn[0].backward(out[i])

	#def f(ab,b):
	#	return [ab*b]
	#Numbrrd = 0
	#for i in range(1 , 10 ** 2+1):

	#	for j in range(1 , 100):
	#		asin , bsin = MODELS.mdn[0].forward([1/i , 1/j]) , f(1/i , 1/j)
	#		Numbrrd += sum([abs(asin[isn] - bsin[isn]) for isn in range(len(asin))])

	#Numbrrd /= 10**3
	#print(Numbrrd)

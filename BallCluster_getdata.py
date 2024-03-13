from __future__ import print_function 

import numpy as np
import os
import sys
import re
import subprocess

class ProgressBar(object):
	DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
	FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

	def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
				 output=sys.stderr):
		assert len(symbol) == 1

		self.total = total
		self.width = width
		self.symbol = symbol
		self.output = output
		self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
			r'\g<name>%dd' % len(str(total)), fmt)

		self.current = 0

	def __call__(self):
		percent = self.current / float(self.total)
		size = int(self.width * percent)
		remaining = self.total - self.current
		bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

		args = {
			'total': self.total,
			'bar': bar,
			'current': self.current,
			'percent': percent * 100,
			'remaining': remaining
		}
		print('\r' + self.fmt % args, file=self.output, end='')

	def done(self):
		self.current = self.total
		self()
		print('\n', file=self.output)




class InputError(Exception):
		pass

def getpara(File_Path:str,
			Para_List:list,
			Fq_num:int,
			File_Type:str,
			encoding='UTF-8',
			normal=False,
			save=True,
			conv2txt=True):
	
	"""
	This is a preprocessing scrip of tensorflow for the files that export from 'CST Studio'

	---------
	## Args:
	---------
		File_Path: the folder containing the S parameter file, it should be ensured that there is only the S parameter file
		Para_List: the name of the structure variable that needs to be extracted
		Fq_num   : the number of frequency points in the exported file
		encoding : Encoding format. Default to 'UTF-8'
		normal   : the return value is normalized. Default to "True"
		save	 : Store the return value as an npz file instead of directly returning it as a tuple. Default to "False".The stored NPZ file contains in order: structure parameter matrix, S parameter matrix, frequency point vector.
		conv2txt : Convert the file to a txt file. Default to "True"

	## Raises:
	---------
		InputError : When the S-para is out range of [-1,2],this erros will be raised ,please check your raw files

	## Returns:
	---------
		P_matrix : A three-dimensional matrix containing all structural parameters, the 0-axis is the file serial number; the 1-axis is the structural parameter row; the 2-axis is the normalized structural parameter;
		S_matrix : A three-dimensional matrix containing all S-parameters, the 0-axis is the file number; the 1-axis is the frequency point, and the 2-axis is the normalized(optional) Mag and phase;
		Fq		 : Frequency point matrix, reserved for the return value for graphing. 
	"""


	if conv2txt == True:
		Fn_conv2txt(File_Path)
	
	if File_Type == 's4p':
		S_matrix,P_matrix,Fq = Fn_s4p(File_Path,Para_List,Fq_num,encoding)
	elif File_Type == 's2p':
		S_matrix,P_matrix,Fq = Fn_s2p(File_Path,Para_List,Fq_num,encoding)
	else:
		raise InputError('Your File_Tpye is wrong, or not support')
	
	if normal == True:
		S_matrix,P_matrix = Fn_normal(S_matrix,P_matrix,File_Type)

	if save == True:
		Fn_save(S_matrix,P_matrix,Fq,File_Path)
	
	return S_matrix,P_matrix,Fq






def Fn_conv2txt(File_Path:str) -> None:
	__,file_type = os.path.splitext(os.listdir(File_Path)[0])
	subprocess.Popen('ren *'+file_type+' *.txt',cwd='./'+File_Path,shell=True)

def Fn_s4p(File_Path: str,
		   Para_List: list,
		   Fq_num: int,
		   encoding='UTF-8',)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
	
	spara_files = os.listdir(File_Path)
	P_matrix = np.zeros((len(spara_files),len(Para_List)),dtype='float')			#定义xdata为三维矩阵
	S_pass = np.zeros((Fq_num * 4,4),dtype= 'float32')
	S_matrix = np.zeros((len(spara_files),Fq_num,16),dtype='float32') # type: ignore
	Fq = np.zeros((Fq_num,1))
	probar = ProgressBar(len(spara_files),fmt=ProgressBar.FULL)

	for index,file in enumerate(spara_files):
		with open(File_Path + '/' + file,"r",encoding=encoding) as r:
			lines = r.readlines()
			Para_line = lines[3]									#提取参数行
			for i in range(0,len(Para_List)):
				re_exp = re.compile(r"(%s)\=(.+?)(\;|\})" %Para_List[i])
				P_matrix[index,i] =  np.asarray(re_exp.findall(Para_line)[0][1])
			for i in range(9,len(lines)):
				S_extract = np.fromstring(lines[i],dtype=float,sep=' ')
				if (i-9)%4 == 0:
					S_pass[i-9] = S_extract[5:9].T
				else:
					S_pass[i-9] = S_extract[4:8].T

		S_matrix[index] = np.reshape(S_pass,(Fq_num,16))
		Fq = np.loadtxt(File_Path + '/' + file , skiprows=9,usecols=(0),encoding=encoding)	#提取频点
		Fq = Fq[0:Fq.shape[0]:4]
		probar.current += 1
		probar()
	probar.done

	return S_matrix,P_matrix,Fq

def Fn_s2p(File_Path: str,
		   Para_List: list,
		   Fq_num: int,
		   encoding='UTF-8',)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
	
	spara_files = os.listdir(File_Path)
	P_matrix = np.zeros((len(spara_files),len(Para_List)),dtype='float')			#定义xdata为三维矩阵
	S_matrix = np.zeros((len(spara_files),Fq_num,4),dtype='float32') # type: ignore
	Fq = np.zeros((Fq_num,1))
	probar = ProgressBar(len(spara_files),fmt=ProgressBar.FULL)

	for index,file in enumerate(spara_files):
		with open(File_Path + '/' + file,"r",encoding=encoding) as r:
			lines = r.readlines()
			Para_line = lines[3]									#提取参数行
			for i in range(0,len(Para_List)):
				re_exp = re.compile(r"(%s)\=(.+?)(\;|\})" %Para_List[i])
				P_matrix[index,i] =  np.asarray(re_exp.findall(Para_line)[0][1])
		S_matrix[index] = np.loadtxt(File_Path + '/' + file ,skiprows=11,usecols=[1,2,3,4],encoding=encoding)
		Fq = np.loadtxt(File_Path + '/' + file , skiprows=11, usecols=0, encoding=encoding)	#提取频点
		probar.current += 1
		probar()
	probar.done

	return S_matrix,P_matrix,Fq


def Fn_normal(S_matrix:np.ndarray,
			  P_matrix:np.ndarray,
			  File_Type:str):
	
	for i in range(P_matrix.shape[1]):
		x_data_max = np.amax(P_matrix[:,i],0)
		x_data_min = np.amin(P_matrix[:,i],0)
		for j in range(P_matrix.shape[0]):				#Normalize the parameters
			P_matrix[j,i] = (P_matrix[j,i]-x_data_min)/(x_data_max-x_data_min)	
	if File_Type == 's4p':
		for i in range(S_matrix.shape[0]):					#Normalize the phase
			S_matrix[i,:,(1,3,5,7,9,11,13,15)] = S_matrix[i,:,(1,3,5,7,9,11,13,15)] / 180

			for j in range(S_matrix.shape[1]):							#Normalize the Mag
				for k in (0,2,4,6,8,10,12,14):
					if (S_matrix[i,j,k] > 1) | (S_matrix[i,j,k] < 0):
						S_matrix[i,j,k] = abs(np.fix(S_matrix[i,j,k]))
					if (S_matrix[i,j,k] > 2) | (S_matrix[i,j,k] < -1):
						raise InputError('An error occurred at No.{} data at the No.{} frequency point of the No.{} file:The mag is not EVEN in [-1,2],please check your files'.format(k,j,i))
	elif File_Type == 's2p':
		for i in range(S_matrix.shape[0]):					#Normalize the phase
			S_matrix[i,:,(1,3)] = S_matrix[i,:,(1,3)] / 180

			for j in range(S_matrix.shape[1]):							#Normalize the Mag
				for k in (0,2):
					if (S_matrix[i,j,k] > 1) | (S_matrix[i,j,k] < 0):
						S_matrix[i,j,k] = abs(np.fix(S_matrix[i,j,k]))
					if (S_matrix[i,j,k] > 2) | (S_matrix[i,j,k] < -1):
						raise InputError('An error occurred at No.{} data at the No.{} frequency point of the No.{} file:The mag is not EVEN in [-1,2],please check your files'.format(k,j,i))
	return S_matrix,P_matrix



def Fn_save(S_matrix:np.ndarray,
			P_matrix:np.ndarray,
			Fq:np.ndarray,
			File_Path:str):
	np.savez(File_Path + '.npz',para = P_matrix,S = S_matrix,F = Fq)
	print('\n{}.npz has been saved'.format(File_Path))
	return(print('Saved!'))
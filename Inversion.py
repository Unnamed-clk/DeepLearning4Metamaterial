from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

def InverFunc(S_11:np.ndarray[int, np.dtype[np.float64]],
			  S_21:np.ndarray[int, np.dtype[np.float64]],
			  dp1:float,
			  dp2:float,
			  d:float,
			  dltz = 0.1,
			  dlty = 1e-5,)	-> np.ndarray[int, np.dtype[np.float64]]:
	"""The function of S-parameter inversion Theory,based on the content of Chapter 6 of this book (ISBN:978-7-03-047793-4)\n\n

	## Parameters:\n
	----------\n
		S_11 : S-parameters of reflection, a 2D matrix with three columns, the three columns are: frequency, amplitude, phase.\n
		S_21 : S-parameters of transmission, data structure same as S_11.\n
		dp1 : The depth from Port1 plane to metamaterial. If you already setup the Reference plane in CST Studio(or other similar function), it should be the depth from the ref. plane to metamaterial.\n
		dp2 : The depth from Port2 plane to metamaterial. same to dp1.\n
		d : The effective thickness of the metamaterial in the wave direction.\n
		dltz : Judgment conditions for real part of impedance. Not recommended to change.\n
		dlty : Judgment conditions for the absolute value of the real part of e^(iknd). Not recommended to change.

	## Returns:\n
	----------\n
		M : The matrix of output data, including:\n
			Frequency pionts,					save in No.0 column,\n
			Real and Imag part of Permeability, save in No.1 and 2 column,\n
			Real and Imag part of Permittivity, save in No.3 and 4 column,\n
			Real and Imag part of Index,		save in No.5 and 6 column,\n
			Real and Imag part of Impedance,	save in No.7 and 8 column,\n
			Real and Imag part of FOM,			save in No.9 column.\n
	"""


	n = len(S_11)
	r ,t ,index ,impedance ,mue ,epsilon ,y ,num = np.zeros((8,n),dtype=complex)
	M = np.zeros((n,10))


	num = S_11[:,0] * 1e+9 *(dp1 + dp2) / 3e+8
	S_11[:,2] += 360 * (num - np.floor(num))

	num = S_21[:,0] * 1e+9 *(dp1 + dp2) / 3e+8
	S_21[:,2] += 360 * (num - np.floor(num))
	for i in range(n):
		if S_11[i,2] > 180:
			S_11[i,2] -= 360
		if S_21[i,2] > 180:
			S_21[i,2] -= 360
		

	r = S_11[:,1] * np.exp(-1j * S_11[:,2] * np.pi / 180)
	t = S_21[:,1] * np.exp(-1j * S_21[:,2] * np.pi / 180)

	impedance = np.sqrt(
		((1 + r) ** 2 - t ** 2) / ((1 - r) ** 2 - t ** 2)
	)

	y = t / (1 - r * (impedance - 1) / (impedance + 1))

	for i in range(len(S_11)):
		if abs(impedance[i].real) >= dltz:
			if impedance[i].real < 0:
				impedance[i] = - impedance[i]
				y[i] = t[i] / (1 - r[i] * (impedance[i] - 1) / (impedance[i] + 1))
		else:
			if abs(y[i]) > 1:
				impedance[i] = - impedance[i]
				y[i] = t[i] / (1 - r[i] * (impedance[i] - 1) / (impedance[i] + 1))
				# if abs(y[i]) > 1:
				# 	pass 
		if abs(y[i].imag) < dlty:
			y[i] = (y[i] + np.conj(y[i])) / 2


	k = 1e+9 * 2 * np.pi * S_11[:,0] / 3e+8
	index = 1 / (1j*k*d) * np.log(y)

	
	mue = index * impedance
	epsilon = index / impedance
	M[:,0] = S_11[:,0]
	M[:,1] = np.real(mue)
	M[:,2] = np.imag(mue)
	M[:,3] = np.real(epsilon)
	M[:,4] = np.imag(epsilon)
	M[:,5] = np.real(index)
	M[:,6] = np.imag(index)
	M[:,7] = np.real(impedance)
	M[:,8] = np.imag(impedance)
	for i in range(n):
		if np.real(index[i]) < 0:
			M[i,9] = abs(np.real(index[i]) / np.imag(index[i]))
	return M


def MakeFig_Full(M:np.ndarray) -> Figure: # type: ignore
	"""The Function to create a object of matplotlib.figure.Figure, which include a set of pictures containing Permeability, Permittivity, Index, Impedance, FOM and MEN (a coupe of real part of Permeability, Permittivity and Index)

	## Parameters:
	----------\n
		M (np.ndarray): The martrix contains Electromagnetic parameters, with 9 column, and it should be:\n
			Frequency pionts,					save in No.0 column,\n
			Real and Imag part of Permeability, save in No.1 and 2 column,\n
			Real and Imag part of Permittivity, save in No.3 and 4 column,\n
			Real and Imag part of Index,		save in No.5 and 6 column,\n
			Real and Imag part of Impedance,	save in No.7 and 8 column,\n
			Real and Imag part of FOM,			save in No.9 column.

	## Returns:
	----------\n
		Figure: a Figure object contains Permeability, Permittivity, Index, Impedance, FOM and MEN
	"""
	M_dict = {
	'fq'    : M[:,0],
	'mue_r' : M[:,1],
	'mue_i' : M[:,2],
	'eps_r' : M[:,3],
	'eps_i' : M[:,4],
	'ind_r' : M[:,5],
	'ind_i' : M[:,6],
	'imp_r' : M[:,7],
	'imp_i' : M[:,8],
	'fom'   : M[:,9]
}
	ref_line = np.zeros((len(M)))

	fig_full = plt.figure(figsize=(12,6),dpi=200)
	ax_mue = fig_full.add_subplot(231)
	ax_mue.plot(M_dict['fq'],M_dict['mue_r'],'red' ,label = 'Real_mue')
	ax_mue.plot(M_dict['fq'],M_dict['mue_i'],'blue',label = 'Imag_mue')
	ax_mue.plot(M_dict['fq'],ref_line,'--k')
	ax_mue.legend()

	ax_eps = fig_full.add_subplot(232)
	ax_eps.plot(M_dict['fq'],M_dict['eps_r'],'red' ,label = 'Real_eps')
	ax_eps.plot(M_dict['fq'],M_dict['eps_i'],'blue',label = 'Imag_eps')
	ax_eps.plot(M_dict['fq'],ref_line,'--k')
	ax_eps.legend()

	ax_ind = fig_full.add_subplot(233)
	ax_ind.plot(M_dict['fq'],M_dict['ind_r'],'red' ,label = 'Real_n')
	ax_ind.plot(M_dict['fq'],M_dict['ind_i'],'blue',label = 'Imag_n')
	ax_ind.plot(M_dict['fq'],ref_line,'--k')
	ax_ind.legend()

	ax_imp = fig_full.add_subplot(234)
	ax_imp.plot(M_dict['fq'],M_dict['imp_r'],'red' ,label = 'Real_imp')
	ax_imp.plot(M_dict['fq'],M_dict['imp_i'],'blue',label = 'Imag_imp')
	ax_imp.plot(M_dict['fq'],ref_line,'--k')
	ax_imp.legend()

	ax_fom = fig_full.add_subplot(235)
	ax_fom.plot(M_dict['fq'],M_dict['fom'],'red',label = 'FOM')
	ax_fom.legend()

	ax_men = fig_full.add_subplot(236)
	ax_men.plot(M_dict['fq'],M_dict['mue_r'],'red'  ,label = 'mue')
	ax_men.plot(M_dict['fq'],M_dict['eps_r'],'green',label = 'eps')
	ax_men.plot(M_dict['fq'],M_dict['ind_r'],'blue' ,label = 'n')
	ax_men.plot(M_dict['fq'],ref_line,'--k')
	ax_men.legend()

	return fig_full

def MakeFig_MEN(M:np.ndarray) -> Figure: # type: ignore
	"""The Function to create a object of matplotlib.figure.Figure, which contains MEN (a coupe of real part of Permeability, Permittivity and Index)

	## Parameters:
	----------\n
		M (np.ndarray): The martrix contains Electromagnetic parameters, with 9 column, and it should be:\n
			Frequency pionts,					save in No.0 column,\n
			Real and Imag part of Permeability, save in No.1 and 2 column,\n
			Real and Imag part of Permittivity, save in No.3 and 4 column,\n
			Real and Imag part of Index,		save in No.5 and 6 column,\n
			Real and Imag part of Impedance,	save in No.7 and 8 column,\n
			Real and Imag part of FOM,			save in No.9 column.

	## Returns:
	----------\n
		Figure: a Figure object contains MEN
	"""
	MEN = {
		'fq' :M[:,0],
		'mue':M[:,1],
		'eps':M[:,3],
		'ind':M[:,5]
	}

	ref_line = np.zeros((len(M)))

	fig_men = plt.figure(figsize=(4,3),dpi=100)
	ax = fig_men.add_axes([0.1,0.1,0.8,0.8])
	ax.plot(MEN['fq'],MEN['mue'],'r',label = 'mue')
	ax.plot(MEN['fq'],MEN['eps'],'g',label = 'eps')
	ax.plot(MEN['fq'],MEN['ind'],'b',label = 'ind')

	ax.plot(MEN['fq'],ref_line,'--k')
	ax.legend()

	return fig_men

import numpy as np
from numpy import linalg as la
import math
import os
import sys

trainlabel = []
testlabel = []
icurr = 0
icurrTest = 0

#arrTest = np.zeros((xaxis,yaxis,zmaxtesting))
#defines dimension of 3d array
abnormalTraining = 72
abnormalTesting = 48
normalTraining = 500
normalTesting = 336


i = 572   #No of training samples
j = 104
k = 6
t = 384   #No of testing samples
r = 24
s = 6

def calc_eigen(J,e):
    U = []
    a = []
    w,v = la.eig(J)

    #print v
    b = w
    b.sort()
    #print "1"

    for x in range (len(b)):
        a.append(b[len(b)-x-1])
    #print "2"
    for x in range(0,e):
        for j1 in range(0,len(a)):
            if (a[x]==w[j1]):
                U.append(v[:,j1])

    U = np.array(U)
    U = U.transpose()   # U = U.transpose()
    #print type(U)
    #print U
    return U

def svd_2d(T,M,num):


    T_mean = np.zeros((j,k), dtype=np.float)
    New_T = []
    New_T_trans = []
    F = np.zeros((j,j),dtype = np.float)
    G = np.zeros((k,k),dtype = np.float)

    for y in range(0,j):
        for z in range(0,k):
            for x in range(0,num):
                T_mean[y][z] += T[x][y][z]

    T_mean /= num

    for x in range(0,num):
        New_T.append(T[x] - T_mean)
        New_T_trans.append(New_T[x].transpose())

    for x in range(0,num):
        F += np.dot(New_T[x],New_T_trans[x])

    for x in range(0,num):
        G += np.dot(New_T_trans[x],New_T[x])

    F /= num
    G /= num

    U = calc_eigen(F,r)
    V = calc_eigen(G,s)

    U_trans = U.transpose()
    V_trans = V.transpose()

    for x in range(0,num):
        M[x] = np.dot(np.dot(U_trans,T[x]),V)

    return (M)


def calc_distance(X,Y):

    sq_rt = 0.0
    for x in range(0,s):
        term = 0.0
        square = 0.0

        for y in range(0,r):

            term = X[y][x] - Y[y][x]
            square += term**2

        sq_rt += math.sqrt(square)

    return (sq_rt)

def KNN_1(M,N):

    nearest = []
    pos = 0
    
    for x in range(0,t):
        value = 15000.0
        result = 0.0
        for y in range(0,i):

            result = calc_distance(N[x],M[y])
            if value>result :
            	value = result
            	pos = y
                
        nearest.append(pos)

    return (nearest)



def extractSeriesTesting(T,src_dir ,label , noOfSeries):
	global icurrTest,testlabel

	count = 0
	j1=0
	k1=0
	i1=icurrTest

	if not os.path.exists(src_dir):
		print ("Source directory does not exist... ")
		sys.exit()
	files_in_directory = os.listdir(src_dir)

	files_in_directory.sort()

	for file in files_in_directory:

		if (count >= (noOfSeries-1)):
			i1 += 1
			icurrTest = i1
			break

		if (file[8] == 'b' or file[8] == 'a'):
			continue
		if (file[8:10] == '11'):
			j1=0
			k1=0
			i1=icurrTest

		#open file and do other parts
		src_path = os.path.join(src_dir, file)
		if not os.path.exists(src_path):
			print ("File does not exist")
			continue


		if (file[8] == '6'):
			k1 = 0

		elif (file[8] == '7'):
			k1=1
		elif (file[8] == '8'):
			k1=2
			count += 1
			i1 += 1
			icurrTest = i1
			testlabel.append(label)
			#count is increased to notify that all attribute values have been stored in a matrix

		elif (file[8:10] == "11"):
			k1=3
		elif (file[8:10] == "12"):
			k1=4
		elif (file[8:10] == "15"):
			k1=5
		else:
			k1=0
			continue


		temp = 0
		fp = open (src_path , 'r')

		while (temp < j):
			text = fp.readline().split()[1]
			#arrTest[temp][y][z]=int(text)
			T[i1][temp][k1]=int(text)
			#print (str(temp)+" "+str(y)+" "+str(z))
			temp += 1

		fp.close()
		k1 = 0
		icurrTest = i1

		#reintitializing values

	icurrTest = i1
	return T

def function():
	global testlabel,trainlabel
	trainlabel.append(1)
	trainlabel.append(1)
	testlabel.append(0)
	testlabel.append(1)

def extractSeries(T,src_dir , label , noOfSeries):
	global icurr,trainlabel


	count = 0
	j1=0
	k1=0
	i1=icurr

	if not os.path.exists(src_dir):
		print ("Source directory does not exist... ")
		sys.exit()
	files_in_directory = os.listdir(src_dir)

	files_in_directory.sort()

	for file in files_in_directory:

		if (count >= (noOfSeries-1)):
			i1 += 1
			icurr = i1
			break

		if (file[8] == 'b' or file[8] == 'a'):
			continue
		if (file[8:10] == '11'):
			j1=0
			k1=0
			i1=icurr

		#open file and do other parts
		src_path = os.path.join(src_dir, file)
		if not os.path.exists(src_path):
			print ("File does not exist")
			continue


		if (file[8] == '6'):
			k1=0

		elif (file[8] == '7'):
			k1=1
		elif (file[8] == '8'):
			k1=2
			count += 1
			i1 += 1
			icurr = i1
			trainlabel.append(label)
			#count is increased to notify that all attribute values have been stored in a matrix
		elif (file[8:10] == "11"):
			k1=3
		elif (file[8:10] == "12"):
			k1=4
		elif (file[8:10] == "15"):
			k1=5
		else:
			k1=0
			continue


		temp = 0
		fp = open (src_path , 'r')

		while (temp < j):
			text = fp.readline().split()[1]
			T[i1][temp][k1]=int(text)
			#print (str(i1)+" "+str(temp)+" "+str(k1))
			temp += 1

		fp.close()
		k1 = 0
		icurr = i1

		#reintitializing values

	icurr = i1
	return T
  

def main():
	global testlabel,trainlabel
	T = np.zeros((i,j,k))
	arrTest = np.zeros((i,j,k))
	M = np.zeros((i,r,s),dtype = np.float)
	N = np.zeros((t,r,s),dtype = np.float)
	count_correct =0
	count_wrong = 0
	print ("Machine Learning Project on Multivariate Time Series Analysis Classification")

	print ("Input Source Directory for Abnormal Dataset:")
	src_dir = raw_input()
	T = extractSeries(T,src_dir , 0, abnormalTraining)
	#print zcurr
	print ("Input Source Directory for Normal Dataset :")
	src_dir = raw_input()
	T = extractSeries(T,src_dir , 1 , normalTraining)
	##################################################################################
	#print "The i current value is ",
	#print icurr
	
	print ("\n\nTraining over...\n\n")
	#####################################################################################
	M = svd_2d (T,M,i)
	print ("Input Source Directory for Testing Abnormal Dataset:")
	src_dir = raw_input()
	arrTest = extractSeriesTesting(arrTest,src_dir , 0, abnormalTesting)			#0 is label for abnormal dataset

	print ("Input Source Directory for Normal Dataset :")
	src_dir = raw_input()
	#print icurrTest

	
	arrTest = extractSeriesTesting(arrTest,src_dir , 1 , normalTesting)
	function()
	N = svd_2d (arrTest,N,t)
	nearest = KNN_1(M,N)
	#print nearest
	
	for x in range(t):
		#if (nearest[x] >= 382)
		if (int(testlabel[x]) == int(trainlabel[nearest[x]])):
			count_correct += 1
		else:
			count_wrong += 1

	print "Accuracy:"
	print ((count_correct+0.0)/(count_correct+count_wrong))
    






if __name__ == '__main__':
 	main()

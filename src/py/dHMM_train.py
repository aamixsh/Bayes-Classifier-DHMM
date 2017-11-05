#	CS669 - Assignment 3 (Group-2) 
#	Last edit: 6/11/17
#	About: 
#		This program is used to train sequential data to form a Discrete Hidden Markov Model.

import numpy as np
import math
import os
import sys
import random
			
dimension=2									#	Dimension of data vectors.
hmmType=0									#	Type of HMM, ergodic or left-right.
N=3											#	Number of states in HMM.
M=5											#	Number of observation symbols, considered equal for all states.
A=[]										#	State Transition Probability Distribution.
B=[]										#	Observation Symbol Probability Distribution.
Pi=[]										#	Initial State Distribution.

classesM=[]
classObservations=[]

Alpha=[]									#	Forward Variable storing values for current iteration.
Beta=[]										#	Backward Variable storing values for current iteration.
scaleCoeff=[]								#	Scaling coefficient for current iteration.
logP=0										#	log(P(O|lambda)) for current iteration.
Del=[]										#	Best state sequence probability for current iteration.
Psi=[]										#	Previous best state for current iteration.
Q=[]										#	Optimal state sequence for current iteration.
Xi=[]										#	Probability of changing state from i to j at time t.
Gamma=[]									#	Probability of being at state i at time t.

#	Calculates Alpha and Beta for next iteration.
def calcAlphaBeta(O):
	global Alpha,Beta,scaleCoeff
	Alpha=[[0 for i in range(N)] for j in range(len(O))]
	Beta=[[0 for i in range(N)] for j in range(len(O))]
	scaleCoeff=[0 for i in range(len(O))]
	
	#	Initializing base values.
	for i in range(N):
		Alpha[0][i]=Pi[i]*B[i][O[0]]
		Beta[len(O)-1][i]=1
		scaleCoeff[0]+=Alpha[0][i]
	for i in range(N):
		Alpha[0][i]/=scaleCoeff[0]
	
	#	Calculating Alpha(t).
	for t in range(len(O)-1):
		for j in range(N):
			Alpha[t+1][j]=B[j][O[t+1]]
			x=0
			for i in range(N):
				x+=Alpha[t][i]*A[i][j]
			Alpha[t+1][j]*=x
			scaleCoeff[t+1]+=Alpha[t+1][j]
		for j in range(N):
			Alpha[t+1][j]/=scaleCoeff[t+1]
	
	#	Calculating Beta(t)
	for t in range(len(O)-1):
		for j in range(N):
			for i in range(N):
				Beta[len(O)-2-t][j]+=A[j][i]*B[i][O[len(O)-1-t]]*Beta[len(O)-1-t][i]
			Beta[len(O)-2-t][j]/=scaleCoeff[len(O)-2-t]
	
	evaluation()

#	Evaluation function to find P(O|lambda).
def evaluation():
	global logP
	logP=0
	for i in range(len(scaleCoeff)):
		logP-=math.log(scaleCoeff[i])

#	Viterbi Algorithm to find the optimal state sequence for next iteration.
def viterbi(O):
	global Del,Psi,Q
	Del=[[0 for i in range(N)] for j in range(len(O))]
	Psi=[[0 for i in range(N)] for j in range(len(O))]
	for i in range(N):
		Del[i]=Pi[i]*B[i][O[0]]
	for t in range(len(O)-1):
		for j in range(N):
			val=[]
			for i in range(N):
				val.append(Del[t][i]*A[i][j])
			Psi[t][j]=np.argmax(val)
			Del[t+1][j]=max(val)*B[j][O[t+1]]
	Q[len(O)-1]=np.argmax(Del[len(O)-1])
	for t in range(len(O)-2,-1,-1):
		Q[t]=Psi[t+1][Q[t+1]]

#	Calculates Xi & Gamma for next iteration.
def calcXiGamma(O):
	global Xi,Gamma
	Xi=[[[0 for i in range(N)] for j in range(N)] for k in range(len(O)-1)]
	Gamma=[[0 for i in range(N)] for j in range(len(O))]
	for t in range(len(O)-1):	
		denom=0
		for i in range(N):
			for j in range(N):
				denom+=Alpha[t][i]*A[i][j]*B[j][O[t+1]]*Beta[t+1][j]
		for i in range(N):
			for j in range(N):
				Xi[t][i][j]=Alpha[t][i]*A[i][j]*B[j][O[t+1]]*Beta[t+1][j]/denom
				Gamma[t][i]+=Xi[t][i][j]
	for i in range(N):
		Gamma[len(O)-1][i]=Alpha[len(O)-1][i]

#	Calculates distance between two points in 'dimension' dimensional space.
def dist(x,y):
	distance=0
	for i in range(dimension):
		distance+=(x[i]-y[i])**2
	distance=math.sqrt(distance)
	return (distance)

#	Creates subdirectories if not present in a path.
def createPath(output):
	if not os.path.exists(os.path.dirname(output)):
		try:
			os.makedirs(os.path.dirname(output))
		except OSError as exc:
			if exc.errorno!=errorno.EEXIST:
				raise

#	Calculates the confusion matrix of all classes together.
def calcConfusion():
	confusionMatrix=[[0 for i in range(len(classes))] for i in range(len(classes))]
	for i in range(len(classes)):
		file=open(os.path.join(directoryO,"k"+str(k+1),classes[i]+"_values.txt"),"r")
		x=0
		for line in file:
			data=line.split()
			confusionMatrix[i][x]=int(data[1])
			x+=1
	return confusionMatrix

#	Returns first element as key for 'elem'.
def takeFirst(elem):
	return elem[0]

#	Calculates M observation symbols for each class' data using k-means clustering.
def calcMSymbols(filename):
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	N=len(tempClass)
	file.close()
	K=M

	#	K-means clustering...

	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
	randomKMeans=random.sample(range(0,N-1),K)
	for i in range(K):
		for j in range(dimension):
			tempClusterMean[i][j]=tempClass[randomKMeans[i]][j]

	#	Dividing the data of this class to K clusters...
	tempClusters=[[] for i in range(K)]
	totDistance=0
	energy=np.inf
	for i in range(N):
		minDist=np.inf
		minDistInd=0
		for j in range(K):
			Dist=dist(tempClass[i],tempClusterMean[j])
			if Dist<minDist:
				minDist=Dist
				minDistInd=j
		tempClusters[minDistInd].append(tempClass[i])
		totDistance+=minDist
	
	iteration=1
	#	Re-evaluating centres until the energy of changes becomes insignificant (convergence)...
	while energy>0.000001:
		tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimension):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimension):
				tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(K)]
		newTotDistance=0
		for i in range(N):
			minDist=np.inf
			minDistInd=0
			for j in range(K):
				Dist=dist(tempClass[i],tempClusterMean[j])
				if Dist<minDist:
					minDist=Dist
					minDistInd=j
			tempClusters[minDistInd].append(tempClass[i])
			newTotDistance+=minDist
		energy=math.fabs(totDistance-newTotDistance)
		totDistance=newTotDistance
		print "Energy in iteration",iteration,"-",energy
		iteration+=1

	print "Done."
	classesM.append(tempClusterMean)

#	Returns the index of the observation symbol which has maximum similarity with 'x'.
def classifySymbol(x,ind):
	val=[0 for i in range(M)]
	for k in range(M):
		val[k]=dist(x,classesM[ind][k])
	return np.argmin(val)

#	Program starts here...
print ("\nThis program is used to train sequential data to form a Discrete Hidden Markov Model.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for features training input and output or default (o/d): ")

direct=""
directO=""
choiceIn='A'

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training feature data directory: ")
	dimension=input("Enter the number of dimensions in the data (for input format, refer README): ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results of the training: ")
else:
	choiceIn=raw_input("Dataset (A/B): ")
	if choiceIn=='A' or choiceIn=='a':
		direct="../../data/Output/Dataset A/featureVectorsCH/train/"
		directO="../../data/Output/Dataset A/test_results/"
		dimension=24
	elif choiceIn=='B' or choiceIn=='b':
		direct="../../data/Output/Dataset B/featureVectorsSpeech/train/"
		directO="../../data/Output/Dataset B/test_results/"
		dimension=39
		hmmType=1
	else:
		print "Wrong input!. Exiting."
		sys.exit()

classes=[]

print "Making M observation symbols out of each classes' data by k-means clustering..."
for contentsTrain in os.listdir(direct):
	contentTrainName=os.path.join(direct,contentsTrain)
	if os.path.isdir(contentTrainName) and contentsTrain=="use":
		for trainFilename in os.listdir(contentTrainName):
			classes.append(os.path.splitext(trainFilename)[0])
			print "Class - "+os.path.splitext(trainFilename)[0]+"..."
			calcMSymbols(os.path.join(contentTrainName,trainFilename))

for c in range(len(classes)):
	classObservations=[]
	print "Reading observation sequences of class '"+classes[c]+"'..."
	for contentsTrain in os.listdir(direct):
		contentTrainName=os.path.join(direct,contentsTrain)
		if os.path.isdir(contentTrainName) and contentsTrain==classes[c]:
			for trainFilename in os.listdir(contentTrainName):
				file=open(os.path.join(contentTrainName,trainFilename))
				tempSequence=[]
				for line in file:
					number_strings=line.split()
					numbers=[float(n) for n in number_strings]
					symbol=classifySymbol(numbers,c)
					tempSequence.append(symbol)
				classObservations.append(tempSequence)
				file.close()
	
	print "Now applying Baum-Welch algorithm to estimate paramters of the HMM for this class..."

	print "Initializing parameters of the Hidden Markov Model for class '"+classes[c]+"'..."
	#	Exectation step.
	#	Initializing A & Pi
	tempA=[[0 for i in range(N)] for j in range(N)]
	tempPi=[0 for i in range(N)]
	if hmmType:							#	Left-Right HMMM
		for x in range(N):
			for y in range(N):
				if x<=y:
					tempA[x][y]=1.0/(N-x)
				else:
					tempA[x][y]=0
			if not x:
				tempPi[x]=1
			else:
				tempPi[x]=0
	else:								#	Ergodic HMM
		for x in range(N):
			for y in range(N):
				tempA[x][y]=1.0/N
			tempPi[x]=1.0/N

	#	Initializing B
	tempB=[[0 for i in range(M)] for j in range(N)]
	for x in range(N):
		for y in range(M):
			tempB[x][y]=1.0/M
	
	A=tempA
	B=tempB
	Pi=tempPi

	print "Re-estimating parameters until convergence..."
	#	Maximization step.
	energy=np.inf
	previousP=0
	iteration=1

	while energy>0.001:
		
		thisP=0
		newA=[[0 for i in range(N)] for j in range(N)]
		newPi=[0 for i in range(N)]
		newB=[[0 for i in range(M)] for j in range(N)]
		
		newAnum=[[0 for i in range(N)] for j in range(N)]
		newAden=[0 for i in range(N)]
		newBnum=[[0 for i in range(M)] for j in range(N)]
		newPinum=[0 for i in range(N)]
		newPiden=0

		for x in range(len(classObservations)):
			calcAlphaBeta(classObservations[x])
			thisP+=logP
			newPiden+=1.0/logP
			calcXiGamma(classObservations[x])
			for t in range(len(classObservations[x])-1):
				for i in range(N):
					for j in range(N):
						newAnum[i][j]+=1.0/logP*Xi[t][i][j]
					newAden[i]+=1.0/logP*Gamma[t][i]
			for t in range(len(classObservations[x])):
				for i in range(N):
					for k in range(len(classObservations[x])):
						if classObservations[x][t]==k:
							newBnum[i][k]+=1.0/logP*Gamma[t][i]
			for i in range(N):
				newPinum[i]+=1.0/logP*Gamma[0][i]
		
		for i in range(N):
			newPi[i]=newPinum[i]/newPiden
			for j in range(N):
				newA[i][j]=newAnum[i][j]/newAden[i]	
			for j in range(M):
				newB[i][j]=newBnum[i][j]/newAden[i]

		A=newA
		B=newB
		Pi=newPi

		if not previousP:
			previousP=thisP
			continue
		else:
			energy=math.fabs(thisP-previousP)
			previousP=thisP

		print "Energy in iteration",iteration,"-",energy
		iteration+=1
	
	print "Done."

	print "Modelling of this class complete. Writing results in files..."
	createPath(os.path.join(directO,"model_first_attempt",classes[c],"hmm.txt"))
	outFileTotal=open(os.path.join(directO,"model_first_attempt",classes[c],"hmm.txt"),"w")
	outFileTotal.write(str(hmmType)+" "+str(N)+" "+str(M)+" "+str(dimension)+"\n")
	for i in range(len(classesM[c])):
		for j in range(dimension):
			outFileTotal.write(str(classesM[c][i][j])+" ")
		outFileTotal.write("\n")
	for i in range(N):
		outFileTotal.write(str(Pi[i])+" ")
	outFileTotal.write("\n")
	for i in range(N):
		for j in range(N):
			outFileTotal.write(str(A[i][j])+" ")
		outFileTotal.write("\n")
	for i in range(N):
		for j in range(M):
			outFileTotal.write(str(B[i][j])+" ")
		outFileTotal.write("\n")
	outFile.close()

print "Done. Modelling ready for testing."

#	End.
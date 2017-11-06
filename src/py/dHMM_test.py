#	CS669 - Assignment 3 (Group-2) 
#	Last edit: 6/11/17
#	About: 
#		This program is a Bayes Classifier using Discrete Hidden Markov Model.

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

classMeanVectors=[]

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
def classifySymbol(x):
	val=[0 for i in range(M)]
	for k in range(M):
		val[k]=dist(x,classMeanVectors[k])
	return np.argmin(val)

#	Program starts here...
print ("\nThis program is a Bayes Classifier using Discrete Hidden Markov Model.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for features test input and output or default (o/d): ")

direct=""
directO=""
directT=""
choiceIn='A'

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the directory that contains the HMM: ")
	directT=raw_input("Enter the path (relative or complete) of the test feature data directory: ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results of the classification: ")
else:
	choiceIn=raw_input("Dataset (A/B): ")
	if choiceIn=='A' or choiceIn=='a':
		direct="../../data/Output/Dataset A/test_results/model_first_attempt/"
		directT="../../data/Output/Dataset A/featureVectorsCH/test/"
		directO="../../data/Output/Dataset A/test_results/"
	elif choiceIn=='B' or choiceIn=='b':
		direct="../../data/Output/Dataset B/test_results/model_first_attempt/"
		directT="../../data/Output/Dataset B/featureVectorsSpeech/test/"
		directO="../../data/Output/Dataset B/test_results/"
	else:
		print "Wrong input!. Exiting."
		sys.exit()

for contentsTrain in os.listdir(direct):
	contentTrainName=os.path.join(direct,contentsTrain)
	if os.path.isdir(contentTrainName):
		file=open(os.path.join(contentTrainName,"hmm.txt"))
		line=file.readline()
		parameters_string=line.split()
		parameters=[int(n) for n in parameters_string]
		hmmType=parameters[0]
		N=parameters[1]
		M=parameters[2]
		dimension=parameters[3]
		classMeanVectors=[]
		for i in range(M):
			line=file.readline()
			tempMean_string=line.split()
			tempMean=[float(n) for n in tempMean_string]
			classMeanVectors.append(tempMean)
		line=file.readline()
		Pi_string=line.split()
		Pi=[float(n) for n in Pi_string]
		A=[[0 for i in range(N)] for j in range(N)]
		for i in range(N):
			line=file.readline()
			tempA=line.split()
			tempAnum=[float(n) for n in tempA]
			for j in range(N):
				A[i][j]=tempAnum[j]
		B=[[0 for i in range(M)] for j in range(N)]
		for i in range(N):
			line=file.readline()
			tempB=line.split()
			tempBnum=[float(n) for n in tempB]
			for j in range(M):
				B[i][j]=tempBnum[j]
		file.close()
		for contentsTest in os.listdir(directT):
			contentTestName=os.path.join(directT,contentsTest)
			if os.path.isdir(contentTestName):
				for filename in os.listdir(contentTestName):
					file=open(os.path.join(contentTestName,filename))
					testSequence=[]
					for line in file:
						number_strings=line.split()
						numbers=[float(n) for n in number_strings]
						symbol=classifySymbol(numbers)
						testSequence.append(symbol)
					calcAlphaBeta(testSequence)
					outPath=os.path.join(directO,"results_first_attempt",contentsTest,"temp.txt")
					createPath(outPath)
					outFile=open(os.path.join(directO,"results_first_attempt",contentsTest,os.path.splitext(filename)[0]+".txt"),"a")
					outFile.write(str(logP)+" "+contentsTrain+"\n")
					outFile.close()
					file.close()

classes=[]
directory=os.path.join(directO,"results_first_attempt")

for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName) and contents!="use":
		classes.append(contents)

confusionMatrix=[[0 for i in range(len(classes))] for j in range(len(classes))]

for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName) and contents!="use":
		classInd=0
		for i in range(len(classes)):
			if classes[i]==contents:
				classInd=i
		for filename in os.listdir(contentName):
			file=open(os.path.join(contentName,filename))
			data=[]
			for line in file:
				dataLine=line.split()
				dataVector=[]
				dataVector.append(float(dataLine[0]))
				dataVector.append(dataLine[1])
				data.append(dataVector)
			data.sort(key=takeFirst)
			classifyInd=0
			for i in range(len(classes)):
				if data[len(data)-1][1]==classes[i]:
					classifyInd=i
			confusionMatrix[classifyInd][classInd]+=1

Sumtot=0
for i in range(len(classes)):
	for j in range(len(classes)):
		Sumtot+=confusionMatrix[i][j]

confusionMatClass=[]
for i in range(len(classes)):
	tempConfusionMatClass=[[0 for j in range(2)] for l in range(2)]
	sumin=0
	tempConfusionMatClass[0][0]=confusionMatrix[i][i]
	sumin+=tempConfusionMatClass[0][0]
	Sum=0
	for j in range(len(classes)):
		Sum+=confusionMatrix[i][j]
	tempConfusionMatClass[0][1]=Sum-tempConfusionMatClass[0][0]
	sumin+=tempConfusionMatClass[0][1]
	Sum=0
	for j in range(len(classes)):
		Sum+=confusionMatrix[j][i]
	tempConfusionMatClass[1][0]=Sum-tempConfusionMatClass[0][0]
	sumin+=tempConfusionMatClass[1][0]
	tempConfusionMatClass[1][1]=Sumtot-sumin
	confusionMatClass.append(tempConfusionMatClass)

print "Data testing complete. Writing results in files for future reference..."
filer=open(os.path.join(directory,"results.txt"),"w")

filer.write("The Confusion Matrix of all classes together is: \n")
for i in range(len(classes)):
	for j in range(len(classes)):
		filer.write(str(confusionMatrix[i][j])+" ")
	filer.write("\n")

filer.write("\nThe Confusion Matrices for different classes are: \n")
for i in range(len(confusionMatClass)):
	filer.write("\nClass "+str(i+1)+": \n")
	for x in range(2):
		for y in range(2):
			filer.write(str(confusionMatClass[i][x][y])+" ")
		filer.write("\n")

Accuracy=[]
Precision=[]
Recall=[]
FMeasure=[]

filer.write("\nDifferent quantitative values are listed below.\n")
for i in range(len(classes)):
	tp=confusionMatClass[i][0][0]
	fp=confusionMatClass[i][0][1]
	fn=confusionMatClass[i][1][0]
	tn=confusionMatClass[i][1][1]
	accuracy=float(tp+tn)/(tp+tn+fp+fn)
	if tp+fp:
		precision=float(tp)/(tp+fp)
	else:
		precision=-1.0
	if tp+fn:
		recall=float(tp)/(tp+fn)
	else:
		recall=-1.0
	if precision+recall:
		fMeasure=2*precision*recall/(precision+recall)
	else:
		fMeasure=-1.0
	filer.write("\nClassification Accuracy for class "+str(i+1)+" is "+str(accuracy)+"\n")
	if precision!=-1.0:
		filer.write("Precision for class "+str(i+1)+" is "+str(precision)+"\n")
	else:
		filer.write("Precision for class "+str(i+1)+" is -\n")
	if recall!=-1.0:
		filer.write("Recall for class "+str(i+1)+" is "+str(recall)+"\n")
	else:
		filer.write("Recall for class "+str(i+1)+" is -\n")
	if fMeasure!=-1.0:
		filer.write("F-measure for class "+str(i+1)+" is "+str(fMeasure)+"\n")
	else:
		filer.write("F-measure for class "+str(i+1)+" is -\n")
	Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
flagP,flagR,flagF=True,True,True
for i in range (len(classes)):
	avgAccuracy+=Accuracy[i]
	if Precision[i]!=-1.0:
		avgPrecision+=Precision[i]
	else:
		flagP=False
	if Recall[i]!=-1.0:
		avgRecall+=Recall[i]
	else:
		flagR=False
	if FMeasure[i]!=-1.0:
		avgFMeasure+=FMeasure[i]
	else:
		flagF=False
avgAccuracy/=len(classes)
avgPrecision/=len(classes)
avgRecall/=len(classes)
avgFMeasure/=len(classes)

filer.write("\nAverage classification Accuracy is "+str(avgAccuracy)+"\n")
if flagP:
	filer.write("Average precision is "+str(avgPrecision)+"\n")
else:
	filer.write("Average precision is -\n")
if flagR:
	filer.write("Average recall is "+str(avgRecall)+"\n")
else:
	filer.write("Average recall is -\n")
if flagF:
	filer.write("Average F-measure is "+str(avgFMeasure)+"\n")
else:
	filer.write("Average F-Measure is -\n")
filer.write("\n**End of results**")
filer.close()

#	End.
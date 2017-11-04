#	CS669 - Assignment 3 (Group-2) 
#	Last edit: 5/11/17
#	About: 
#		This program is a Bayes Classifier using Discrete Hidden Markov Model.

import numpy as np
import math
import os
import sys
			
dimension=2									#	Dimension of data vectors.
N=3											#	Number of states in HMM.
M=5											#	Number of observation symbols, considered equal for all states.
A=[]										#	State Transition Probability Distribution, ergodic.
Alr=[]										#	State Transition Probability Distribution, left-right.
B=[]										#	Observation Symbol Probability Distribution.
Pi=[]										#	Initial State Distribution. π(i) for all states.
numClasses=3								#	Number of classes.
Alpha=[]									#	Forward Variable storing values for current iteration. α(i) at time t.
Beta=[]										#	Backward Variable storing values for current iteration. β(i) at time t.
P=0											#	P(O|λ) for current iteration.
Del=[]										#	Best state sequence probability for current iteration. δ(i) at time t.
Psi=[]										#	Previous best state for current iteration. ψ(i) at time t.
Q=[]										#	Optimal state sequence for current iteration.
Xi=[]										#	ξ(i,j) at time t.
Gamma=[]									#	γ(i) at time t.

#	Calculates Alpha and Beta for next iteration.
def calcAlphaBeta(O,ind):
	global Alpha,Beta
	Alpha=[[0 for i in range(N)] for j in range(len(O))]
	Beta=[[0 for i in range(N)] for j in range(len(O))]
	for i in range(N):
		Alpha[0][i]=Pi[ind][i]*B[ind][i][O[0]]
		Beta[len(O)-1][i]=1
	for t in range(len(O)-1):
		for j in range(N):
			Alpha[t+1][j]=B[ind][j][O[t+1]]
			x=0
			for i in range(N):
				x+=Alpha[t][i]*A[ind][i][j]
				Beta[len(O)-2-t][j]+=A[ind][j][i]*B[ind][i][O[len(O)-1-t]]*Beta[len(O)-1-t][i]
			Alpha[t+1][j]*=x
	evaluation(len(O))

#	Evaluation function to find P(O|λ).
def evaluation(T):
	global P
	P=0
	for i in range(N):
		P+=Alpha[T-1][i]

#	Viterbi Algorithm to find the optimal state sequence for next iteration.
def viterbi(O,ind):
	global Del,Psi,Q
	Del=[[0 for i in range(N)] for j in range(len(O))]
	Psi=[[0 for i in range(N)] for j in range(len(O))]
	for i in range(N):
		Del[i]=Pi[ind][i]*B[ind][i][O[0]]
	for t in range(len(O)-1):
		for j in range(N):
			val=[]
			for i in range(N):
				val.append(Del[t][i]*A[ind][i][j])
			Psi[t][j]=np.argmax(val)
			Del[t+1][j]=max(val)*B[ind][j][O[t+1]]
	Q[len(O)-1]=np.argmax(Del[len(O)-1])
	for t in range(len(O)-2,-1,-1):
		Q[t]=Psi[t+1][Q[t+1]]

#	Calculates Xi & Gamma for next iteration.
def calcXiGamma(O,ind):
	global Xi,Gamma
	Xi=[[[0 for i in range(N)] for j in range(N)] for k in range(len(O)-1)]
	Gamma=[[0 for i in range(N)] for j in range(len(O)-1)]
	for t in range(len(O)-1):	
		for i in range(N):
			x=0
			for j in range(N):
				Xi[t][i][j]=Alpha[t][i]*A[ind][i][j]*B[ind][j][O[t+1]]*Beta[t+1][j]/P
				x+=Xi[t][i][j]
			Gamma[t][i]=x

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

#	Program starts here...
print ("\nThis program is a Bayes Classifier using Discrete Hidden Markov Model.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for features training/test input and output or default (o/d): ")

direct=""
directO=""
directT=""
choiceIn='A'

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training feature data directory: ")
	directT=raw_input("Enter the path (relative or complete) of the test feature data directory: ")
	dimension=input("Enter the number of dimensions in the data (for input format, refer README): ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results of the classification: ")
else:
	choiceIn=raw_input("Dataset (A/B): ")
	if choiceIn=='A' or choiceIn=='a':
		direct="../../data/Output/Dataset A/featureVectorsCH/train/"
		directT="../../data/Output/Dataset A/featureVectorsCH/test/"
		directO="../../data/Output/Dataset A/test_results/"
		dimension=24
	elif choiceIn=='B' or choiceIn=='b':
		direct="../../data/Output/Dataset B/featureVectorsSpeech/train/"
		directT="../../data/Output/Dataset B/featureVectorsSpeech/test/"
		directO="../../data/Output/Dataset B/test_results/"
		dimension=39
	else:
		print "Wrong input!. Exiting,"
		sys.exit()

if direct[len(direct)-1]!='/':
	direct+="/"
if directO[len(directO)-1]!='/':
	directO+="/"
if directT[len(directT)-1]!='/':
	directT+="/"

maxK=input("Enter the value of K upto which you want to test the data: ")

classes=[]

print "Calculating DTW distances. This will take a while..."
for contents in os.listdir(directT):
	contentName=os.path.join(directT,contents)
	if os.path.isdir(contentName):
		classes.append(contents)
		print "Inside test class - "+contents+"..."
		for filename in os.listdir(contentName):
			file=open(os.path.join(contentName,filename))
			testSequence=[]
			for line in file:
				number_strings=line.split()
				numbers=[float(num) for num in number_strings]
				testSequence.append(numbers)
			n=len(testSequence)
			print "Calculating DTW distances of all training samples from test sample - "+filename+"..."
			for contentsTrain in os.listdir(direct):
				createPath(os.path.join(directO,"distances_third_attempt",contents,filename,"total.txt"))
				outFileTotal=open(os.path.join(directO,"distances_third_attempt",contents,filename,"total.txt"),"a")
				contentTrainName=os.path.join(direct,contentsTrain)
				if os.path.isdir(contentTrainName) and contentsTrain!="use":
					for trainFilename in os.listdir(contentTrainName):
						trainFile=open(os.path.join(contentTrainName,trainFilename))
						trainSequence=[]
						for line in trainFile:
							number_strings=line.split()
							numbers=[float(num) for num in number_strings]
							trainSequence.append(numbers)
						m=len(trainSequence)
						DTWdistance=DTW(testSequence,n,trainSequence,m)
						outFileTotal.write(str(DTWdistance)+" "+trainFilename+" "+contentsTrain+"\n")
				outFileTotal.close()

directory=os.path.join(directO,"distances_third_attempt")
directoryO=os.path.join(directO,"results_third_attempt")
print "Sorting distances..."
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		for contentsTest in os.listdir(contentName):
			contentTestName=os.path.join(contentName,contentsTest)
			if os.path.isdir(contentTestName):
				file=open(os.path.join(contentTestName,"total.txt"),"r")
				outFile=open(os.path.join(contentTestName,"total_sorted.txt"),"w")
				Array=[]
				for line in file:
					number_strings=line.split()
					numbers=[]
					for i in range(len(number_strings)):
						if i==0:
							numbers.append(float(number_strings[i]))
						else:
							numbers.append(number_strings[i])
					Array.append(numbers)
				Array.sort(key=takeFirst)
				for i in range(len(Array)):
					outFile.write(str(Array[i][0])+" "+Array[i][1]+" "+Array[i][2]+"\n")

print "Classifiying samples according to k-NN method..."
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		for contentsTest in os.listdir(contentName):
			contentTestName=os.path.join(contentName,contentsTest)
			if os.path.isdir(contentTestName):
				file=open(os.path.join(contentTestName,"total_sorted.txt"),"r")
				outFile=open(os.path.join(contentTestName,"classify_labels_for_k.txt"),"w")
				k=0
				freq=[0 for i in range(len(classes))]
				for line in file:
					number_strings=line.split()
					value=number_strings[len(number_strings)-1]
					for x in range(len(classes)):
						if value==classes[x]:
							freq[x]+=1
					maxFreq=0
					maxFreqInd=0
					for x in range(len(classes)):
						if freq[x]>maxFreq:
							maxFreq=freq[x]
							maxFreqInd=x
					outFile.write(classes[maxFreqInd]+"\n")
					k+=1
					if k==maxK:
						break
				file.close()
				outFile.close()

classesData=[]
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		classImages=[]
		for contentsTest in os.listdir(contentName):
			contentTestName=os.path.join(contentName,contentsTest)
			if os.path.isdir(contentTestName):
				imageData=[]
				file=open(os.path.join(contentTestName,"classify_labels_for_k.txt"),"r")
				for line in file:
					number_strings=line.split()
					imageData.append(number_strings[0])
				classImages.append(imageData)
				file.close()
		classesData.append(classImages)

for k in range(maxK):
	print "Calculating results for k = "+str(k+1)+"..."
	for i in range(len(classesData)):
		createPath(os.path.join(directoryO,"k"+str(k+1),classes[i]+"_values.txt"))
		file=open(os.path.join(directoryO,"k"+str(k+1),classes[i]+"_values.txt"),"w")
		count=[]
		for j in range(len(classes)):
			x=[]
			x.append(classes[j])
			x.append(0)
			count.append(x)
			del x
		for x in range(len(classesData[i])):
			for y in range(len(classes)):
				if classesData[i][x][k]==count[y][0]:
					count[y][1]+=1
		for x in range(len(count)):
			file.write(count[x][0]+" "+str(count[x][1])+"\n")
		file.close()

	confusionMatrix=calcConfusion()
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
	filer=open(os.path.join(directoryO,"k"+str(k+1),"results.txt"),"w")

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
	del confusionMatClass

#	End.
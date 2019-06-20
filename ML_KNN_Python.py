#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 09:59:38 2019

@authors: hamza, saad, shaker
"""

import numpy as np
import matplotlib.pyplot as plt

def generateData():

    x = np.random.rand(1000,3)

    a = (np.power(x[:,0]-0.3, 2) + np.power(x[:,1]-0.6,2)) < 0.056
    b = (np.power(x[:,0]-0.5, 2) + np.power(x[:,1]-0.37,2)) < 0.06
    c = (np.power(x[:,0]-0.7, 2) + np.power(x[:,1]-0.6,2)) < 0.056
    x[:,2] = (a + b + c)
    print(np.sum(x[:,2]) / 1000)
    
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:700], indices[700:]
    trainingSet, testingSet = x[training_idx,:], x[test_idx,:]
    
    return trainingSet, testingSet

def addNoise(noise, tLinkFlag):
    
    #Adding noise to the 0% noise data set
    trainingSet, testingSet = loadData(0, tLinkFlag)
    
    #Introducing noise in how much % of the training set? 
    noiseRatio = (len(trainingSet) * noise/100)
    
    for x in range(int(noiseRatio)):
        if (trainingSet[x][-1] == 1):
            trainingSet[x][-1] = 0
        else:
            trainingSet[x][-1] = 1
    
    saveData(trainingSet, testingSet, noise, tLinkFlag)
          

def saveData(trainingSet, testingSet, noise, tLinkFlag):
    
    np.savetxt('trainingSet_' + str(noise) + '_' + str(tLinkFlag) + '.csv' , trainingSet, fmt="%f", delimiter=",")
    np.savetxt('testingSet_%s.csv' % noise, testingSet, fmt="%f", delimiter=",")
    
def loadData(noise, tLinkFlag):
    
    trainingSet = np.loadtxt(open('trainingSet_' + str(noise) + '_' + str(tLinkFlag) + '.csv',), delimiter=",")
    testingSet = np.loadtxt(open("testingSet_%s.csv" % noise,), delimiter=",")
    
    return trainingSet, testingSet
 
def plotGraph(trainingSet):
    
    plt.scatter(trainingSet[trainingSet[:,2]==0][:,0] , trainingSet[trainingSet[:,2]==0][:,1], alpha=0.7, label='Negative Examples', color='blue', )
    plt.scatter(trainingSet[trainingSet[:,2]==1][:,0] , trainingSet[trainingSet[:,2]==1][:,1], alpha=0.7, label='Positive Examples', color='orange')
    #plt.scatter(trainingSet[trainingSet[:,2]==2][:,0] , trainingSet[trainingSet[:,2]==2][:,1], alpha=0.7, color='magenta')    
    plt.show()
    
def plotAccVsNoise(npArr):
    
    kValues = npArr[:, 0]
    
    for k in range(7, 23, 2):
        for m in range(len(np.where(kValues == k**2))):
            plt.xticks(ticks = (0,5,10,15,20,25,30,35,40,45,50))
            plt.yticks(ticks = (40,45,50,55,60,65,70,75,80,85,90,95,100))
            plt.xlabel("Noise (%)")
            plt.ylabel("Classification Accuracy (%)")
            plt.plot(npArr[npArr[:,0]==k**2][:,2] , npArr[npArr[:,0]==k**2][:,1], label = 'K = %s' % k**2)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.plot()


def plotAccVsNoise2(npArr):
    
    kValues = npArr[:, 0]
    
    for k in range (1, 44, 6):
        for m in range(len(np.where(kValues == k))):
            plt.xticks(ticks = (0,5,10,15,20,25,30,35,40,45,50))
            plt.yticks(ticks = (40,45,50,55,60,65,70,75,80,85,90,95,100))
            plt.xlabel("Noise (%)")
            plt.ylabel("Classification Accuracy (%)")
            plt.plot(npArr[npArr[:,0]==k][:,2] , npArr[npArr[:,0]==k][:,1], label = 'K = %s' % k)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.plot()

def getDistancesIndices(data, testExample):

    calculatedDistances = np.empty(shape=[1, 0])

    #Using Euclidean distance metric.
    for i in range(len(data)):
        calculatedDistances = np.append(calculatedDistances, np.sqrt(np.sum(np.power(data[i, 0:2]-testExample[0:2], 2))))
        
    sortedDistancesIndices = np.argsort(calculatedDistances)
    
    #Element of Index 0 of sortedDistancesIndices, corresponds to the index of smallest
    #distance in the calculatedDistances list.
    return sortedDistancesIndices

def getNeighbors(trainingSet, testExample, k):
    
    sortedDistancesIndices = getDistancesIndices(trainingSet, testExample)
    
    #Initializing a numpy array of size kx3.
    kClosestNeighbors = np.empty(shape=[k, 3])
    
    #Retrieve the closest K data points to the test example.
    #Exclude the 0th row, as thats just the data point itself, and include uptil k (inclusive) and k+1 (exclusive)
    kClosestNeighbors = np.array(trainingSet[sortedDistancesIndices[1:k+1], 0:])
   
    #outputs a 2D numpy array
    return kClosestNeighbors

def getPredictions(trainingSet, testingSet, k):  
    
    predictions = np.empty(shape=[0, 3])
    
    for row in range(len(testingSet)):
        neighbors = getNeighbors(trainingSet, testingSet[row], k) 
        
        votes = np.array(neighbors[0:, -1])
        
        num_zeros = (votes == 0).sum()
        num_ones = (votes == 1).sum()
    
        if (num_zeros > num_ones):
            temp = np.append(testingSet[row, 0:2], [0])
            predictions = np.append(predictions, np.array(temp).reshape(1, 3), axis = 0)
        elif (num_ones > num_zeros):
            temp = np.append(testingSet[row, 0:2], [1])
            predictions = np.append(predictions, np.array(temp).reshape(1, 3), axis = 0)
            
    return predictions   #numpy array with feature values & predicted labels
        

def getAccuracy(testingSet, predictionsArr):
    
    predictedCorrectly = 0.0
    
    #3rd column, -1 for correct prediction & -2 for wrong prediction.
    for x in range(len(testingSet)):
        if testingSet[x][-1] == predictionsArr[x][-1]:
            predictedCorrectly += 1.0
            predictionsArr[x][-1] = -1
        else:
            predictionsArr[x][-1] = -2

    #returns a numpy array & accuracy percentage.
    return predictionsArr, round(((predictedCorrectly/float(len(testingSet))) * 100.0), 1) 
 
 
def plotPredictions(trainingSet, predicted_to_plot, k, noise, percentageAccuracy, num_tLinks_removed):
    
    plt.title('KNNClassifier Test Accuracy: ' + str(percentageAccuracy) + '%' + '; K = ' + str(k) + '; Noise: ' + str(noise) + '%' 
              '\n' 'Tomek Links Removed: ' + str(num_tLinks_removed) + '; Noise Removed: ' + 
              "{0:.1f}".format((num_tLinks_removed/(len(trainingSet) + num_tLinks_removed))*100) + '%')
    plt.scatter(trainingSet[trainingSet[:,2]==0][:,0] , trainingSet[trainingSet[:,2]==0][:,1], alpha=0.7, color='blue')
    plt.scatter(trainingSet[trainingSet[:,2]==1][:,0] , trainingSet[trainingSet[:,2]==1][:,1], alpha=0.7, color='orange')
    plt.scatter(predicted_to_plot[predicted_to_plot[:,2]==-1][:,0] , predicted_to_plot[predicted_to_plot[:,2]==-1][:,1], label='Correctly Predicted', color='green')
    plt.scatter(predicted_to_plot[predicted_to_plot[:,2]==-2][:,0] , predicted_to_plot[predicted_to_plot[:,2]==-2][:,1], label='Wrongly Predicted', color='red')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    

def detectTomekLinks(trainingSet):
    
    #tomek_links = np.empty(shape=[0, 3])
    tomek_links = np.full((len(trainingSet), 1), False, dtype=bool)
    
    num_tlinks_found = 0
    
    for i in range(len(trainingSet)):
        #Assume y is the 1-neighbor of x
        x = trainingSet[i, 0:3]
        y = (getNeighbors(trainingSet, x, 1)).flatten()
        
        #if the x, y labels/class are different
        if (x[2] != y[2] and tomek_links[i, 0] == False):
            #if x is also a 1-neighbor of y then, tomek link identified.
            if np.all((getNeighbors(trainingSet, y, 1)).flatten() == x): 
                #tomek_links = np.append(tomek_links, x.reshape(1,3), axis = 0)
                #tomek_links = np.append(tomek_links, y.reshape(1,3), axis = 0)
                tomek_links[i,0] = True
                y_index = np.where(trainingSet[0:, 0:2] == y[0:2])[0]
                
                if (y_index[0] == y_index[1]):
                    tomek_links[y_index[0], 0] = True
                    num_tlinks_found += 1
         
    #outputs a boolean (mask) Nx1 array corresponding to the indexes in the trainingSet.
    return tomek_links, num_tlinks_found

def removeTomekLinks(trainingSet):
    
    clean_trainingSet = np.copy(trainingSet)
    
    tomek_links, t_links_removed = detectTomekLinks(trainingSet)
        
    #Removes the corresponding indexes in training set, at which dataset row - tomek_links is true.
    clean_trainingSet = trainingSet[np.where(tomek_links == False)[0], 0:3]
    
    temp = np.setdiff1d(trainingSet, clean_trainingSet, True)
    #for x in range(len(trainingSet)):
    #    if (tomek_links[x][0] == True):
    #        clean_trainingSet[x][-1] = 2

    return clean_trainingSet, t_links_removed

def saveTrainingSetsTomek(n):
    
    for t in range(0, 51, 10):
        
        trainingSet, testingSet = loadData(t, n)
        print(len(trainingSet))
        clean_trainingSet, t_links_removed = removeTomekLinks(trainingSet)
        print(len(clean_trainingSet))
        saveData(clean_trainingSet, testingSet, t, n+1)

def testing_Ks(tLinkFlag): #K: 49 onwards
    tempPlot = np.zeros([8,3])
    kPlot = np.empty(shape=[0,3])
    
    for t in range(0, 51, 10): #the noise 
        training, testing = loadData(t, tLinkFlag) #retrieving .csv files
        print('Noise%: ' + str(t))
        for k in range (7, 23, 2):
            tempPlot[int((k-7)/2),2] = t
            predictions = getPredictions(training, testing, k**2)
        #      print('2')
            tempPlot[int((k-7)/2),0] = k**2
            x2, tempPlot[int((k-7)/2),1] = getAccuracy(testing, predictions)
            print(k**2)
        kPlot = np.append(kPlot, tempPlot, axis = 0) 
            
#    plt.plot(kplot[:,0],kplot[:,1])

    return kPlot

def testing_Ks2(tLinkFlag): #K: 1 to 43
    tempPlot = np.zeros([8,3])
    kPlot = np.empty(shape=[0,3])
    
    for t in range(0, 51, 10): #the noise 
        training, testing = loadData(t, tLinkFlag) #retrieving .csv files
        print('Noise%: ' + str(t))
        for k in range (1, 44, 6):
            tempPlot[int((k-1)/6),2] = t
            predictions = getPredictions(training, testing, k)
        #      print('2')
            tempPlot[int((k-1)/6),0] = k
            x2, tempPlot[int((k-1)/6),1] = getAccuracy(testing, predictions)
            print(k)
        kPlot = np.append(kPlot, tempPlot, axis = 0) 
            
#    plt.plot(kplot[:,0],kplot[:,1])

    return kPlot


    
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np,random, operator, pandas as pd
import scipy as sc
from skimage import io
import math
from statistics import mode
from scipy import stats
from PIL import Image
from skimage.transform import rescale, resize
import random, operator, pandas as pd

def createChromosome(k,maxSol):
    solution = np.random.randint(0,maxSol,k)
    return solution

def initialPopulation(popSize, k,constant):
    population = []
    for i in range(0, popSize):
        population.append(createChromosome(k,constant))
    return population

def calculateError(chromosome,problem):
    size = len(chromosome)
    result = 0
    for i in range(0,size):
        if chromosome[i] != problem[i]:
            result = result + 1
    return result
    
def rankRoutes(population,problem):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = calculateError(population[i],problem)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    #df['cum_sum'] = df.Fitness.cumsum()
    #df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
    	selectionResults.append(popRanked[i][0])
    #for i in range(0, len(popRanked) - eliteSize):
    #	pick = 100*random.random()
    #	for i in range(0, len(popRanked)):
    #	    if pick <= df.iat[i,3]:
    #	        selectionResults.append(popRanked[i][0])
    #	        break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
    	index = selectionResults[i]
    	matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    mutationIndex1 = random.randrange(0,len(patent1));
    mutationIndex2 = random.randrange(mutationIndex1,len(patent1));

    for i in range(0, mutationIndex1):
    	child.append(parent1[i]);

    for i in range(mutationIndex1,mutationIndex2):
    	child.append(parent2[i]);	
        
    for i in range(mutationIndex2,len(patent2)):
    	child.append(parent2[i]);	
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
    	children.append(matingpool[i])
    	
    for i in range(0, length):
    	child = breed(pool[i], pool[len(matingpool)-i-1])
    	children.append(child)
    return children	

def mutate(individual, mutationRate):

    index = int(random.random() * len(individual))
    individual[index] = (individual[index] + 1)%2        
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
    	mutatedInd = mutate(population[ind], mutationRate)
    	mutatedPop.append(mutatedInd)
    return mutatedPop
    
def getmax(a,b):
    size = len(a)
    if size == 1:
        return a[0]
    else:
        if b[0] < b[1]:
            return a[0]
        else:
            return a[1]

#array = getTargetArray()
def createarray(array,rows,cols):
    #aa = np.ones((2,3))
    print(len(array))
    i = 0
    myarr = []
    for x in range(0,rows):
        #tarr = []
        for y in range(0,cols):
            if array[i] == 0:
                b = np.zeros((5,5))
                if y == 0:
                    tarr = b
                else:    
                    tarr = np.hstack((tarr,b)) 
            if array[i] == 1:
                b = np.ones((5,5))
                if y == 0:
                    tarr = b
                else:    
                    tarr = np.hstack((tarr,b))
            i = i+1
        if x == 0:
            myarr = tarr
        else:     
            myarr = np.vstack((myarr,tarr))
    return myarr
    

# Reading imge 
# Convert it to binary image
def getTargetArray(array,rows,cols):
    print(rows)
    print(cols) 
    my_list = []
    for x in range(0,rows):
        for y in range(0,cols):
                subarray = array[(x*5):((x*5)+5),(y*5):((y*5)+5)]
                unique,counts = np.unique(subarray,return_counts=True) 
                k = getmax(unique,counts)
                my_list.append(k)
    return my_list

def geneticAlgorithm(constant, popSize, k,eliteSize, mutationRate, generations,orgImage):
    currentGen = initialPopulation(popSize, k,constant)
    #print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        popRanked = rankRoutes(currentGen,orgImage)
        print(popRanked[0])
        if popRanked[0][1] == 0:
            return currentGen[popRanked[0][0]]
        selectionResults = selection(popRanked, eliteSize)
        matingpool = matingPool(currentGen, selectionResults)
        children = breedPopulation(matingpool, eliteSize)
        children = mutatePopulation(children, mutationRate)
        
        currentGen = np.vstack((currentGen,children))
        popRanked = rankRoutes(currentGen,orgImage)
        selectionResults = selection(popRanked, popSize)
        currentGen = matingPool(currentGen, selectionResults)
        #currentGen = nextGeneration(currentGen, eliteSize, mutationRate,problem)

    
    #print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    #bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = currentGen[0]

    return bestRoute
    
    img = io.imread('test1.png',as_gray=True) 
    img = resize(img,(215,215))
    array = np.array(img)
    array[array < 1] = 0;
    array[array >= 1] = 1;
    
    #array = getDummy()
    trows,tcols = array.shape
    rows = math.floor(trows/5)
    cols = math.floor(tcols/5)

    arr = getTargetArray(array,rows,cols);
    #print(arr);
    #myarr1 = createarray(aa,rows,cols)
    size = len(arr)
    #mproblem = Equation(arr)
    myarr = geneticAlgorithm(constant = 2,popSize = 100,k=size,eliteSize = 100,mutationRate = 0.1,
       generations = 2000,orgImage = arr )
    print(myarr)
#   myarr = mproblem.getParams()
#   currentGen = initialPopulation(10, size,2)
#   for i in range(0,9):    
#         myarr = createarray(currentGen[i],rows,cols)
#   print()
    resarr = createarray(myarr,rows,cols)
    resarr[resarr < 1] = 0
    resarr[resarr > 0] = 255 
    img = Image.fromarray(resarr)
    img.show()
    

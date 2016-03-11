
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
n = 2
t = 4
noise = 0.1 #expansion coefficient
np.seterr(divide='raise', invalid='raise')

eta = .0001
Randy = noise*np.random.normal(size = (n, n) )
Webster = noise*np.random.normal(size = (n, n) )
np.fill_diagonal(Randy, 0)
print noise
A = np.ones((n,n))+Randy+Randy.transpose()

Ad = A-np.identity(n)
B = 4*A+Webster
#B[0,0]=8
#A[0,3]=.4
dist = np.bmat([[Ad, B], [B.transpose(), Ad]])
dist = np.asarray(dist)
print dist
print type(dist)
print type(A)

def computeLaplaceMatrix(dMatrix, t):
    try :K = np.exp(-dMatrix/(2*t))
    except: print dMatrix,'error here'
    density = K.sum(axis = 1)
    try: K = np.diag(1/(density-1)).dot(K)
    except:
		print 'density------------------->', density
		exit(0)
    try:

        dd_err = density/(density-1)
        I = np.identity(len(dMatrix))
        I = np.diag((density)/(density-1)).dot(I)

    except: print density, K, dMatrix, "density"
    L = K -I
    L = (2/t)*L
    return L
    
L = computeLaplaceMatrix(dist,t)

def CDC1(L,f,g):
    u=f*g
    cdc = L.dot(u)-f*L.dot(g)-g*L.dot(f)
    return cdc/2




def coarseRicci(L,dMatrix):
    Ric = np.zeros((len(dMatrix),len(dMatrix)))
    for i in range(len(L)):
        for j in range(len(L)):
            f_ij = dMatrix[:,i]-dMatrix[:,j]#pay close attention to this if it becomes asymmetric
            cdc1 = CDC1(L, f_ij,f_ij)
            cdc2 = L.dot(cdc1)-2*CDC1(L,f_ij,L.dot(f_ij))
            Ric[i,j]=cdc2[i]/2
    return Ric

Ricci = coarseRicci(L,dist)
            
total_distance0 = dist.sum()


print 'initial distance'
print dist
dist = dist -eta*Ricci
print 'new dist'
print dist
c = 1
for i in range(20000):
    L = computeLaplaceMatrix(dist,t)
    Ricci = coarseRicci(L,dist)
    dist = dist -eta*Ricci
    dist = dist + dist.transpose()
    dist = dist/2
	
    #total_distance = dist.sum()
    #dist = (total_distance0/total_distance)*dist
    nonzero = dist[np.nonzero(dist)]
    mindist = np.amin(nonzero)
    t = mindist
    #print t
    dist = dist/t
    if i%900==2:
		#print Ricci
		print dist
		#print Ricci/dist
		print '---------'

#plt.show()



quit()


for i in range(50):
    L = computeLaplaceMatrix(dist,t)
    Ricci = coarseRicci(L,dist)
    dist = dist -eta*Ricci+7*eta*dist
    dist = dist + dist.transpose()
    dist = dist/2
    Ii =[1,2,3, len(dist)-4, len(dist)-3, len(dist)-3]
    window1 = dist[0:3,0:3]
    window2 = dist[0:3,len(dist)-4:len(dist)-1]
    window3 = dist[len(dist)-4:len(dist)-1,len(dist)-4:len(dist)-1]
    c = c+1
    if c%10==2:
        print 'block1' , window1
        print 'block mixed', window2
        print 'block 2', window3

#plt.show()
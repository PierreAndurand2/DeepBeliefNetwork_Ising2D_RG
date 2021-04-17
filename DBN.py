## This program simulates 25,000 40x40 Ising2D model on a square lattice at critical temperature. 
## Then a deep belief network with 1600-400-100-25 layers is trained on that data, and we compare reconstructed images to input images
## and compare the receptive fields of each layer in order to compare qualitatively to Kananoff's variational block spin renormalization
## It takes about 24 hours to simulate 25,000 Ising configurations
## The DBM takes about 8h to be trained with the chosen parameters 

import numpy as np
from random import random, randint
import math
from matplotlib import pyplot as plt

#Initializing square matrix of side L with -1 and 1s
def init(L):
        state = 2 * np.random.randint(2, size=(L,L)) - 1
        return state

#Calculate energy of a spin configuration        
def E_dimensionless(config,L):
    total_energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%L, j] + config[i, (j+1)%L] + config[(i-1)%L, j] + config[i, (j-1)%L]
            total_energy += -nb * S
    return (total_energy/4)

#Calculate magnetization of a configuration (we do not use it here)
def magnetization(config):
    Mag = np.sum(config)
    
    return Mag

#One MC Metropolis step
def MC_step(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    L = len(config)
    for i in range(L):
        for j in range(L):
            a = np.random.randint(0, L) # looping over i & j therefore use a & b
            b = np.random.randint(0, L)
            sigma =  config[a, b]
            neighbors = config[(a+1)%L, b] + config[a, (b+1)%L] + config[(a-1)%L, b] + config[a, (b-1)%L]
            del_E = 2*sigma*neighbors
            if del_E < 0:
                sigma *= -1
            elif random() < np.exp(-del_E*beta):
                sigma *= -1
            config[a, b] = sigma
    return config

#This function is to plot how observables vary with temperature (we do not use it here)
def calcul_energy_mag_C_X(config, L, eqSteps, err_runs):
        
    # L is the length of the lattice
        
    nt      = 100         #  number of temperature points
    mcSteps = 1000
    
    T_c = 2/math.log(1 + math.sqrt(2))
        
    # the number of MC sweeps for equilibrium should be at least equal to the number of MC sweeps for equilibrium

    # initialization of all variables
    T = np.linspace(1., 7., nt) 
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    C_theoric, M_theoric = np.zeros(nt), np.zeros(nt)
    delta_E,delta_M, delta_C, delta_X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    n1 = 1.0/(mcSteps*L*L)
    n2 = 1.0/(mcSteps*mcSteps*L*L)    # n1 and n2 will be use to compute the mean value and the # by sites
        # of E and E^2
        
    Energies = []
    Magnetizations = []
    SpecificHeats = []
    Susceptibilities = []
    delEnergies = []
    delMagnetizations = []
    delSpecificHeats = []
    delSusceptibilities = []
    
    for t in range(nt):
        # initialize total energy and mag
        beta = 1./T[t]
        # evolve the system to equilibrium
        for i in range(eqSteps):
            MC_step(config, beta)
        # list of ten macroscopic properties
        Ez = [], Cz = [], Mz = [], Xz = [] 

        for j in range(err_runs):
            E = E_squared = M = M_squared = 0
            for i in range(mcSteps):
                MC_step(config, beta)           
                energy = E_dimensionless(config,L) # calculate the energy at time stamp
                mag = abs(magnetization(config)) # calculate the abs total mag. at time stamp

                # sum up total energy and mag after each time steps

                E += energy
                E_squared += energy**2
                M += mag
                M_squared += mag**2


            # mean (divide by total time steps)

            E_mean = E/mcSteps
            E_squared_mean = E_squared/mcSteps
            M_mean = M/mcSteps
            M_squared_mean = M_squared/mcSteps

            # calculate macroscopic properties (divide by # sites) and append

            Energy = E_mean/(L**2)
            SpecificHeat = beta**2 * (E_squared_mean - E_mean**2)/L**2
            Magnetization = M_mean/L**2
            Susceptibility = beta * (M_squared_mean - M_mean**2)/(L**2)

            Ez.append(Energy); Cz.append(SpecificHeat); Mz.append(Magnetization); Xz.append(Susceptibility);

        Energy = np.mean(Ez)
        Energies.append(Energy)
        delEnergy = np.std(Ez)
        delEnergies.append(float(delEnergy))
        
        Magnetization = np.mean(Mz)
        Magnetizations.append(Magnetization)
        delMagnetization = np.std(Mz)
        delMagnetizations.append(delMagnetization)

        
        SpecificHeat = np.mean(Cz)
        SpecificHeats.append(SpecificHeat)
        delSpecificHeat = np.std(Cz)
        delSpecificHeats.append(delSpecificHeat)

        Susceptibility = np.mean(Xz)
        delSusceptibility = np.std(Xz)        
        Susceptibilities.append(Susceptibility)
        delSusceptibilities.append(delSusceptibility)
        
            
        
        if T[t] - T_c >= 0:
            C_theoric[t] = 0
        else:
            M_theoric[t] = pow(1 - pow(np.sinh(2*beta), -4),1/8)
        
        coeff = math.log(1 + math.sqrt(2))
        if T[t] - T_c >= 0:
            C_theoric[t] = 0
        else: 
            C_theoric[t] = (2.0/np.pi) * (coeff**2) * (-math.log(1-T[t]/T_c) + math.log(1.0/coeff) - (1 + np.pi/4)) 
        
    return (T,Energies,Magnetizations,SpecificHeats,Susceptibilities, delEnergies, delMagnetizations,M_theoric, 
            C_theoric, delSpecificHeats, delSusceptibilities)

# Pruning weights of W matrix for regularization. We do not use it here, as we chose the L1 regularization
def pruneW(W, portion=0.5):
    #prunes the lowest portion of weights of W
    m = W.shape[0] #number of rows
    n = W.shape[1] #number of columns
    WA=np.absolute(W)
    WAf=WA.flatten()
    WAf = np.array(WAf)
    threshold = np.sort(WAf)[int(m*n*portion)]
    for i in range(m):
        for j in range(n):
            if W[i,j]>=0. and W[i,j]<threshold:
                W[i,j]=0.
            elif W[i,j]<0. and W[i,j]>-threshold:
                W[i,j]=0.
    return W
#function pruning vector weights        
def prune_vec(V, portion=0.5):
    n=len(V)
    VA=np.absolute(V)
    VA=np.array(VA)
    threshold = np.sort(VA)[int(n*portion)]        
    for i in range(n):
        if V[i]>=0. and V[i]<threshold:
            V[i]=0.
        elif V[i]<0 and V[i]<-threshold:
            V[i]=0.
    return V

#Sampling from binomial distribution with proba p
def sample(p):
    return np.random.binomial(1,p)     

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#contrastive divergence 
def CD(M, k, W, b, c, eta, del_W1):
    #contrastive divergence
    #k steps
    #M list of matrices (visible weights). Matrices to be flatten into vectors of length m. batch
    #H hidden layer. vector of length n
    #W, b, c weights and biases
    #eta: learning rate
    #del_W1 for momentum
    #print("M=",M)
    lamb=.0002 #for L1 regularization
    batch_n = len(M) #s is the number of samples in M. batch number
    n = len(c)
    #print("n=",n)
    S=[M[i].flatten() for i in range(batch_n)]
    #print("S=",S)
    m=len(S[0])
    #print("m=",m)
    del_W=np.zeros((n, m))
    del_b=np.zeros(m)
    del_c=np.zeros(n)
    pv_h=np.zeros((k+1,n))
    #print("pv_h[0] size=",np.shape(pv_h[0]))
    ph_v=np.zeros((k,m))
    #print("ph_v[0] size=",np.shape(ph_v[0]))
    for l in range(batch_n):
        v_0=S[l]
        v_k=v_0
        for t in range(k):
            #print("W=",W)
            #print("v=",v[t])
            #print("t=",t)
            #print("W_size=",np.shape(W))
            #print("v_size=",np.shape(v_k))
            pv_h[t]=sigmoid(np.matmul(W,v_k)+c)
            h=sample(pv_h[t])
            ph_v[t]=sigmoid(np.matmul(np.transpose(W),h)+b)
            v_k=sample(ph_v[t])
        pv_h[k]=sigmoid(np.matmul(W,v_k)+c)    
        #print()
        del_W+=np.outer(pv_h[0],v_0)-np.outer(pv_h[k],v_k)
        
        del_b+=v_0-v_k
        #print("del_b=",del_b)
        del_c+=pv_h[0]-pv_h[k]
        #print("del_c=",del_c)
    
    
    new_W = W+eta*del_W+0.5*del_W1
    #L1 regularization
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if new_W[i,j]>0:
                new_W[i,j]-=lamb
            else:
                new_W[i,j]+=lamb
    new_b = b+eta*del_b
    new_c = c+eta*del_c
    del_W = eta*del_W+0.5*del_W1

    return new_W, new_b, new_c, del_W

     
#initializing weights for first RBM    
def initialize():
    b=np.zeros(1600)
    c=np.random.uniform(0,.001,400)       
    W=np.random.normal(0,0.1,(400,1600))    
    return W, b, c    

#initializing weights for second RBM 
def initialize2():
    b=np.zeros(400)
    c=np.random.uniform(0,.001,100)       
    W=np.random.normal(0,0.1,(100,400))    
    return W, b, c

#initializing weights for third RBM 
def initialize3():
    b=np.zeros(100)
    c=np.random.uniform(0,.001,25)       
    W=np.random.normal(0,0.1,(25,100))    
    return W, b, c
    
def train(M, i):
    # i is the RBM number in the DBN 
    total_training_examples = len(M)
    if i==1:
        W, b, c = initialize()
    elif i==2:
        W, b, c = initialize2()
    elif i==3:
        W, b, c = initialize3()

    epochs=200
    batch=100
    eta=0.001
    number_batches = int(total_training_examples/batch)
    del_W=np.zeros(W.shape)
    for r in range(epochs):
        print("training, epoch number:", r)
        for q in range(number_batches):
            M_batch = M[q*batch:(q+1)*batch]
            W, b, c, del_W=CD(M_batch, k=1, W=W, b=b, c=c, eta=eta, del_W1=del_W)
    return W, b, c

#sampling the next layer after the weights are trained
def generate_next_layer(M, W, b, c):
    n_training = len(M)
    HL=[]
    for i in range(n_training):
        S = M[i].flatten()
        p = sigmoid(np.matmul(W,S)+c)
        h = sample(p)
        HL.append(h)
    return HL

def generate_training_data(n):
    #temperatures in J/k units
    #energy in J units  
    #n number of training data to generate (2D Ising at critical temperature)
    
    T=2.3
    beta = 1/T
    eqSteps=100
    M=[]
    for j in range(n):
        print("generating data, instance number: ", j)
        config=init(40)
        for i in range(eqSteps):
            config = MC_step(config,beta)
        for k in range(40):
            for l in range(40):
                if config[k,l]==-1: #replacing -1 by 0 for the contrastive divergence
                    config[k,l]=0
        M.append(config)
    return M
        
def main():
    number_instances = 25000
    M=generate_training_data(number_instances)
    W, b, c = train(M,1)
    #print("pruning")
    #W=pruneW(W,0.5)
    #b=prune_vec(b,0.5)
    #c=prune_vec(c,0.5)
    #above we prune 50% of lowest absolute weights 
    H=generate_next_layer(M, W, b, c)
    print("generated 2nd layer")
    W2, b2, c2 = train(H,2)
    #W2=pruneW(W2,0.5)
    H2=generate_next_layer(H, W2, b2, c2)
    print("generated 3rd layer")
    W3, b3, c3 = train(H2,3)
    #W3=pruneW(W3,0.5)
    H3=generate_next_layer(H2, W3, b3, c3)
    print("generated 4th layer")
    
    
    
    #last hidden layer
    HRS3=[]
    for i in range(len(H3)):
        h = H3[i].reshape((5,5))
        HRS3.append(np.transpose(h))
        
    
    
    #now reconstruct M from H3:
    MR3=[]
    MR2=[]
    MR=[]
    for i in range(number_instances):
        h3 = H3[i]
        p3 = sigmoid(np.matmul(np.transpose(W3),h3)+b3)
        vr3 = sample(p3)
        MR3.append(vr3)
        p2 = sigmoid(np.matmul(np.transpose(W2),vr3)+b2)
        vr2 = sample(p2)
        MR2.append(vr2)
        p1 = sigmoid(np.matmul(np.transpose(W),vr2)+b)
        vr1 = sample(p1)
        MR.append(vr1)
    
    VR=[]
    for i in range(number_instances):
        v = MR[i].reshape((40,40))
        VR.append(np.transpose(v))
        
    for i in range(10):
    
        plt.imshow(M[i],interpolation='nearest')
        plt.title('Visible 40x40')
        plt.show()
    
        plt.imshow(VR[i],interpolation='nearest')     
        plt.title('1600-400-100-25 Visible reconstructed 40x40')
        plt.show()
    
        plt.imshow(HRS3[i],interpolation='nearest')    
        plt.title('Hidden 5x5')
        plt.show()
        
        
    RF2=np.matmul(W2, W)
    RF3=np.matmul(W3, RF2)    
        
    #plot receptive fields
    for i in range(10):
        plt.imshow(W[i].reshape((40,40)), interpolation='nearest')
        plt.title('Receptive field 1st layer ')
        plt.show()
        plt.imshow(RF2[i].reshape((40,40)), interpolation='nearest')
        plt.title('Receptive field 2nd layer ')
        plt.show()
        plt.imshow(RF3[i].reshape((40,40)), interpolation='nearest')
        plt.title('Receptive field 3rd layer ')
        plt.show()
    


    #savinf the weights in files
    file1 = open("W matrix","w")
    for row in W:
        np.savetxt(file1, row)
    file1.close()
    
    file2 = open("W2 matrix","w")
    for row in W2:
        np.savetxt(file2, row)
    file2.close()
    
    file3 = open("W3 matrix","w")
    for row in W3:
        np.savetxt(file3, row)
    file3.close()
    
    
        
    
main()    

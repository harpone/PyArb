""" PyArb: Statistical Arbitrage with Python
Copyright (c) Heikki Arponen <heikki.a.arponen@gmail.com>

Distributed under the terms of the GNU GENERAL PUBLIC LICENSE Version 2.

The full license is in the file LICENSE.md, distributed with this software.
""" 



#-----------------------------------------------------------------------------
#IMPORTS:
#-----------------------------------------------------------------------------
from  __future__  import  division
from  __future__  import  print_function
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

#-----------------------------------------------------------------------------
# Some helper functions
#-----------------------------------------------------------------------------

def diffdata(dataframe):
    """Acts on a dataframe (time*space). Returns the
    1 min time difference.
    """
    dim = len(dataframe.values[0])
    temp_df = dataframe.shift(-1)-dataframe
    temp_df.iloc[-1] = temp_df.iloc[-2] # Otherwise uses the first value!
    return temp_df

#-----------------------------------------------------------------------------
# Data import for The Bonnot Gang
#(thebonnotgang.com) 1 min bar data
#-----------------------------------------------------------------------------

class PrepareData_TBG:
    """Prepare data downloaded from 
    The Bonnot Gang.
    """
    def __init__(self):
        self.pricedata = []
        self.numberofdays = None
        self.dim = None
        self.baddayslist = []
        self.normalized = None
        self.read_data()
    
    def read_file(datafile):
        """Uses Pandas read_csv. Change the parameters 
        according to data used! Uses only the closing 
        prices for the time being...
        """
        return pd.read_csv(datafile,sep=';',decimal=',',\
            usecols=['timestamp','close'],index_col = [0], parse_dates=True)
    
    def read_data(self):  
        """Reads a csv file from the current directory, 
        resamples and adjusts dates and times.
        
        NOTE: assumes 1 min data!!
        
        """
        prices = self.pricedata
        #read data:
        for datafile in glob.glob("*.csv"):  
            prices.append(pd.read_csv(datafile,sep=';',decimal=',',\
            usecols=['timestamp','close'],index_col = [0], parse_dates=True))
            #use filename as label:
            prices[-1] = prices[-1].rename(columns={'close' : \
                        os.path.split(datafile)[1].split('.')[0]})
        #concatenate together, sort, resample to 1 min:
        prices = pd.concat(prices, axis = 1, join = 'outer').sort_index(axis=0)\
        .resample('min',how='mean')
        #remove weekends and nights:
        prices = prices[prices.index.dayofweek < 5]
        prices = prices.between_time('09:30:00', '16:00:00')
        
        prices = prices.ix['2011-09-22':]# REMOVE THIS!!!
        
        self.pricedata = prices
        self.numberofdays = int(np.floor(len(self.pricedata)/391))
        self.dim = len(self.pricedata.iloc[0])
        
    def normalize(self):
        """Normalizes the stock prices to start with 1.
        """
        self.normalized = self.pricedata/self.pricedata.iloc[0]
        
    def clean(self, tolerance = .5):
        """Finds days with more than tolerance*datapoints NaNs
        and returns new price data with those days removed. 
        Also interpolates rest of missing data.
        """
        baddayslist = []
        for n in xrange(0,self.numberofdays):
            if np.isnan(self.pricedata.iloc[n*391:(n+1)*391].values).sum() \
            > tolerance*391*self.dim:
                baddayslist.append(n)
        self.baddayslist = baddayslist
        #Remove bad days and get rid of NaNs by interpolation:
        indexlist = []
        cleandata = self.pricedata
        for n in xrange(0,len(baddayslist)):
            ran = [i for i in xrange(baddayslist[n]*391,(baddayslist[n]+1)*391)]
            indexlist.append(ran)
        indexlist = np.array(indexlist)
        indexlist = indexlist.flatten()
        cleandata = self.pricedata.drop(self.pricedata.index[indexlist])
        cleandata = cleandata.apply(pd.Series.interpolate)
        #Update pricedata and number of days:
        self.pricedata = cleandata
        self.numberofdays = int(np.floor(len(self.pricedata)/391))
        
    def get_X(self):
        """Gets the price data (array X). Just restricts with
        between_time to force the array to be same size
        as dX.
        """
        return self.pricedata.between_time('09:30:00', '15:59:00')
    
    def get_dX(self):
        """Gets the price time difference data (array dX).
        """
        return diffdata(self.pricedata).between_time('09:30:00', '15:59:00')
        



#-----------------------------------------------------------------------------
# Child class for data import from eoddata.com
# NOTE!! The data is assumed to be in the form
# "3F VIP Trading"
#-----------------------------------------------------------------------------

class PrepareData_eoddata(PrepareData_TBG):
    """Class for the data from eoddata.com 
    in "3F VIP Trading" format.
    """
    
    def __init__(self, symbols =['AAPL','GOOG','FB']):
        self.symbollist = symbols
        self.pricedata = []
        self.numberofdays = None
        self.dim = None
        self.baddayslist = []
        self.normalized = None
        self.read_data()
    
    def read_data(self):  
        """        
        NOTE: assumes 1 min data!!
        
        """
        prices = self.pricedata
        #read data:
        counter = 0
        for datafile in glob.glob("stock/*.txt"): # Note that the files are .txt instead of .csv!  
            daydata = pd.read_csv(datafile,usecols=['<TICKER>','<DTYYYMMDD>','<CLOSE>'],dtype = {'<DTYYYMMDD>':str})
            daydata = daydata.pivot_table(values='<CLOSE>', rows='<DTYYYMMDD>', cols='<TICKER>')
            daydata.index = pd.to_datetime(daydata.index)
            #daydata.index = pd.date_range(start = str(min(daydata.index)), end = str(max(daydata.index)), freq='T')
            daydata = daydata[self.symbollist]
            daydata = daydata.between_time('09:30:00', '16:00:00')
            daydata = daydata.fillna(method='pad') 
            daydata = daydata.fillna(method='bfill') # takes care of possible first NaNs
            prices.append(daydata)
            counter += 1
            print(counter, end=',')
        #concatenate:
        prices = pd.concat(prices, axis = 0)        
        self.pricedata = prices
        self.numberofdays = int(np.floor(len(self.pricedata)/391))
        self.dim = len(self.pricedata.iloc[0])



#-----------------------------------------------------------------------------
# Vector Geometric Brownian Motion (VGBM) simulation:
#-----------------------------------------------------------------------------

A_default = .1*np.array([[-2,1,1],[1,-2,1],[1,1,-2]])
rho_default = .0001*np.array([[1,0,0],[0,1,0],[0,0,1]])
def VGBM(A = A_default, rho = rho_default, N=390, T=1, init = np.array([1,1,1])):
    """ Simulates VGBM for number of days T and 
    minute level time steps 1/N.
    -------------------------------------------
    - A is the drift/interaction matrix
    - rho is the noise correlation matrix
    - init is the initial condition
    """
    dt = 1/N
    dim = len(init)
    W_indep = np.random.randn(T*N,dim)
    L_W = np.linalg.cholesky(rho)
    W = np.einsum('ij,kj', L_W, W_indep).T
    X = np.ones((T*N,dim))
    X[0,:] = init
    for t in xrange(1,T*N):
        for d in xrange(0,dim):
           X[t,d]=X[t-1,d]+np.einsum('ij,j',A,X[t-1,:])[d]*dt+\
            X[t-1,d]*W[t-1,d]*np.sqrt(dt)
    return X





#-----------------------------------------------------------------------------
# VGBM parameter estimation
#-----------------------------------------------------------------------------

def VGBM_moment(X_data,dX_data):#data is DataFrame i.e. time*space
    ''' This is a simple and extremely efficient "moment
    method" estimation method, which is actually 
    equivalent to the "multivariate least squares" method 
    (look it up on Wikipedia).    
    ------------------------------------------------------
    X_data = price DataFrame
    dX_data = dprice DataFrame
    Assumes between_time Pandas method for 09:00..15:59
    Outs rho, A of the VGBM model.
    '''
    dt = 1/390
    NT=len(X_data)
    #Take values & switch to space*time:
    X = X_data.values.T 
    dim = len(X[:,0])
    dX = dX_data.values.T # time differential, first index is space, second time
    R = np.zeros((dim,dim))
    C = np.zeros((dim,dim))
    Q = np.zeros((dim,dim))
    for i in xrange(0,dim):
        for j in xrange(0,dim):
            R[i,j] = sum(dX[i]*dX[j])/NT/dt
            C[i,j] = sum(X[i]*X[j])/NT
            Q[i,j] = sum(dX[i]*X[j])/NT/dt
    return R/C, einsum('ij,jk',Q,linalg.inv(C))
    
    
    
    
#-----------------------------------------------------------------------------   
# Functions to extract the eigenvalues of matrix A:
#-----------------------------------------------------------------------------

def min_eigvec(Amat):
    """Sorts the eigenvalues from smallest to largest
    and sorts the eigenvectors in same order. Returns
    the smallest eigenvector if the corresponding
    eigenvalue is real, or the real part if it is
    complex valued (could take the Im part too).
    """
    
    eigvals, eigvecs = np.linalg.eig(Amat.T) #LEFT eigenvectors since phi = xi.S
    ind = argsort(eigvals)
    eigvals = eigvals[ind]
    eigvecs = eigvecs[:, ind]   # second axis !!
    
    #Get rid of sign flips:
    neutralmode = np.real(eigvecs[:,0])*sign(np.real(eigvecs[0,0])) 
    #Note that the real part of an eigenvector is NOT an eigenvector!
    
    return neutralmode

def eigvecs_sorted(Amat):
    """Sorts the eigenvalues from smallest to largest
    and sorts the eigenvectors in same order. Returns 
    the real parts if complex valued (could take
    the Im part too). Although the real parts of eigs
    are NOT eigs, the corresponding portfolios can
    be shown to still be mean reverting.
    """
    eigvals, eigvecs = np.linalg.eig(Amat.T) # LEFT eigenvectors, since phi = xi.X
    ind = np.argsort(eigvals)
    eigvals = eigvals[ind]
    eigvecs = np.real(eigvecs[:, ind])   # second axis !!
    
    # Normalize s.t. 1st component is positive:    
    first_components_sign = np.sign(eigvecs[0])
    eigvecs = np.multiply(eigvecs, first_components_sign)
        
    return eigvecs



#-----------------------------------------------------------------------------
# Compute the dynamical eigenvectors etc. for all of the data
#-----------------------------------------------------------------------------

def compute_modes(X_data, dX_data, day1, day2, caldays):
    """Computes & returns the time dependent eigen-
    modes, the portfolio's standard deviation, the 
    eigenvector and the portfolio value as a function
    of t.
    ----------------------------------------------
    Returns phi = array((time,dimension)), 
            pf_std = array((time,dimension)), 
            xi = array((time,dimension,dimension)), 
            phi_val array((time,dimension))= 
    """
    
    # Initialize data
    X = X_data.iloc[(day1-caldays)*390:390*(day2)].values # NOTE: time*space!
    dX = dX_data.iloc[(day1-caldays)*390:390*(day2)].values
    X_init = X_data.iloc[(day1-caldays)*390:390*(day1)].values
    dX_init = dX_data.iloc[(day1-caldays)*390:390*(day1)].values
    
    dim = len(X[0])
    NT = 390*caldays
    C = np.zeros((dim,dim))
    Q = np.zeros((dim,dim))
    phi = np.zeros(((day2-day1)*390,dim))
    phi_val = np.zeros(((day2-day1)*390,dim))
    pf_std = np.zeros(((day2-day1)*390,dim))
    xi = np.zeros(((day2-day1)*390,dim,dim)) # Last dim is the eigenvalue index!
    for i in xrange(0,dim):
        for j in xrange(0,dim): # Note time*space below!!
            C[i,j] = sum(X_init[:,i]*X_init[:,j])/NT
            Q[i,j] = sum(dX_init[:,i]*X_init[:,j])/caldays
    for t in xrange(0,(day2-day1)*390): 
        for i in xrange(0,dim):
            for j in xrange(0,dim):
                C[i,j] += (X[390*caldays+t,i]*X[390*caldays+t,j] -X[t,i]*X[t,j])/NT
                Q[i,j] += (dX[390*caldays+t,i]*X[390*caldays+t,j] -dX[t,i]*X[t,j])/caldays
        A = np.einsum('ij,jk',Q,np.linalg.inv(C))
        xi[t]  = eigvecs_sorted(A)
        phi[t] = np.dot(X[390*caldays+t],xi[t])
        phi_val[t] = np.dot(X[390*caldays+t],abs(xi[t]))
        pf_std[t] = np.sqrt(np.diagonal(np.dot(xi[t].T,np.dot(C,xi[t]))))
    return phi, pf_std, xi, phi_val
    
    
    
    
    
#-----------------------------------------------------------------------------  
# The main backtest class
#-----------------------------------------------------------------------------

class VGBM_backtest:
    def __init__(self, X_data, dX_data, day1=50, day2=51, caldays=4):
        self.day1, self.day2, self.caldays = day1, day2, caldays
        self.Xdata, self.dXdata = X_data, dX_data
        
        #All data to be used:
        self.Xvec = X_data.iloc[(day1-caldays)*390:390*(day2)]
        self.dXvec = dX_data.iloc[(day1-caldays)*390:390*(day2)]
        
        #Initial values for rho, A and the neutral mode:
        #self.rho_init, self.A_init = VGBM_moment(self.Xvec[0:caldays*390],self.dXvec[0:caldays*390])
        #self.nmode_init = min_eigvec(self.A_init)
        self.dim = len(self.Xvec.T)
        self.oos_mins = len(self.Xvec[0:(day2-day1)*390])
        
        #Initialize modes(t), pf_std(t):
        self.modes = None
        self.modes_val = None
        self.pf_std = None
        self.xi = None
        
    def compute(self):
        """ Computes all the modes and stds, calls outside
        compute_modes.
        """
        self.modes = \
         compute_modes(self.Xdata,self.dXdata,self.day1,self.day2,self.caldays)[0]
        self.pf_std\
        = compute_modes(self.Xdata,self.dXdata,self.day1,self.day2,self.caldays)[1]
        self.xi\
        = compute_modes(self.Xdata,self.dXdata,self.day1,self.day2,self.caldays)[2]
        self.modes_val\
        = compute_modes(self.Xdata,self.dXdata,self.day1,self.day2,self.caldays)[3]


# This pnl function closes all positions at the end of day.
#
    def pnl(self, mode = 0, cls = .15, exit = 50., \
            opn = 1., wait = 0, tr_percent = 0, tr_pershare = 0):
        """Calculates the cumulative p&l for given
        *mode* over the
        entire range of dates. The portfolio will
        be continuously readjusted until the pf 
        crosses the trade open band at 
        opn*standard deviation. The pf will then 
        be locked & tracked until it hits the 
        cls*std band (winning trade) or the 
        exit*std band (losing trade) or at the
        end of day.
        ------------------------------------------
        - cls: Close position/ cash out at +- cls*std
        - exit: Close position/ cut losses
        - opn: Open long/short position
        - wait: Wait n minutes at the beginning of
        each trading day before starting to trade
        - tr_percent: Transaction costs in percent
        - tr_pershare: Transaction costs in price
        per share
        """
        self.cls, self.exit, self.opn = cls, exit, opn
        X = self.Xdata.iloc[self.day1*390:390*self.day2].values
        phi = self.modes[:,mode]
        phi_val = self.modes_val[:,mode]
        xi = self.xi[:,:,mode]
        tr_fac = 1-tr_percent/100
        eps = tr_pershare
        
        opn = self.opn*self.pf_std[:,mode]
        cls = self.cls*self.pf_std[:,mode]
        exit = self.exit*self.pf_std[:,mode]
        
        pf_pos = None
        #lims = np.zeros((self.oos_mins,7))
        openval = 1E-10
        openprice = 0
        openxi = np.ones(self.dim)
        self.capital = np.ones(self.oos_mins)
        #poslist =  [] # for debugging
        #open_at = [] # for debugging
        #close_at = [] # for debugging
        numberofdays = int(np.floor(self.oos_mins/390))
        tradetime = np.concatenate([np.arange(wait+n*390,390+n*390)\
                    for n in xrange(0,numberofdays)])
        self.numberoftrades = 0
        
        for t in xrange(0,self.oos_mins):
            self.capital[t] = self.capital[t-1]
            if t in tradetime:
                if pf_pos == None and exit[t]>phi[t]>opn[t]: # short
                    pf_pos = -1
                    openprice = phi[t]
                    openval = phi_val[t]
                    openxi = xi[t]
                    #poslist.append(pf_pos)
                    #open_at.append(t)
                    self.numberoftrades +=1
                if pf_pos == None and -exit[t]<phi[t]<-opn[t]: # long
                    pf_pos = 1
                    openprice = phi[t]
                    openval = phi_val[t]
                    openxi = xi[t]
                    #poslist.append(pf_pos)
                    #open_at.append(t)
                    self.numberoftrades +=1
                if pf_pos == 1 and np.dot(X[t],openxi)>-cls[t]: # close long
                    self.capital[t] =\
tr_fac*self.capital[t - 1]*(1+pf_pos*(np.dot(X[t],openxi)-openprice)/openval\
                            -2*eps*np.sum(abs(openxi))/openval)
                    pf_pos = None
                    #close_at.append(t)
                    self.numberoftrades +=1
                if pf_pos == -1 and np.dot(X[t],openxi)<cls[t]: # close short
                    self.capital[t] =\
tr_fac*self.capital[t - 1]*(1+pf_pos*(np.dot(X[t],openxi)-openprice)/openval\
                            -2*eps*np.sum(abs(openxi))/openval)
                    pf_pos = None
                    #close_at.append(t)
                    self.numberoftrades +=1
                if t in 390*np.arange(1,self.day2-self.day1+1)-1 and pf_pos is not\
                None: # close at EoD
                    self.capital[t] =\
tr_fac*self.capital[t - 1]*(1+pf_pos*(np.dot(X[t],openxi)-openprice)/openval\
                            -2*eps*np.sum(abs(openxi))/openval)
                    pf_pos = None
                    #close_at.append(t)
                    self.numberoftrades +=1
        self.averagereturn = \
                    (self.capital[-1]/self.capital[0])**(1/self.numberoftrades)
        return self.capital#, open_at, close_at
    #, openprice, openval, openxi, poslist, np.dot(X,openxi)
                
                
    def show(self, t, mode = 0, day=0, past = 1, future = 1):
        """Show the process X(s).xi(t) for s.
        """
        plt.figure(figsize=(12, 9))
        #set xticks.. TAKES TIME, GET RID OF AFTER DEBUG
        plt.xticks(np.arange(0,(self.caldays+self.day2-self.day1)*390,10),\
                   np.arange(-390*self.caldays,future*390,10))
        plt.tick_params(labelsize = 'small')
        plt.grid()
        #Plot limits:
        plt.xlim([390*(self.caldays-past+day),(self.caldays+future+day)*390])
        plt.ylim([-6*self.pf_std[t,mode],6*self.pf_std[t,mode]])
        #Calibration begin and end, i.e. tracking window:
        plt.axvspan(t, 390*(self.caldays)+t, facecolor='g', alpha=0.05)
        #plt.axvline(x=t, color='g')
        #plt.axvline(x=390*(self.caldays)+t, color='g')
        #First dawn and other day closes
        plt.axvline(x=390*(self.caldays), color='black')
        for n in xrange(1,future+past):
            plt.axvline(x=390*(self.caldays+n), \
                color='black', linestyle='dashed', alpha=0.5)
        
        #Open bands (at * 1 std)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             self.pf_std[:,mode])), color='r', alpha=0.7)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             -self.pf_std[:,mode])), color='r', alpha=0.7)
        
        #Close bands (at * 1 std)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             self.pf_std[:,mode])), color='r', alpha=0.7)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             -self.pf_std[:,mode])), color='r', alpha=0.7)
        
        plt.plot(np.dot(self.Xvec.values,self.xi[t,:,mode]))
        plt.plot(np.concatenate((np.zeros((self.caldays*390,))\
                ,self.modes[:,mode])), color='g', alpha=0.3)
        plt.show()    
        
        
        
    def show_dynamic(self, mode= 0, day=0, past = 1, future = 1):
        """Show the process X(t).xi(t) and std(t) for all t.
        """
        plt.figure(figsize=(12, 9))
        #Plot limits:
        plt.xlim([390*(self.caldays-past+day),(self.caldays+future+day)*390])
        plt.ylim([-6*self.pf_std[0,mode],6*self.pf_std[0,mode]])
        #First dawn and other day closes
        plt.axvline(x=390*(self.caldays), color='black')
        for n in xrange(1,future+past):
            plt.axvline(x=390*(self.caldays+n),\
                color='black', linestyle='dashed', alpha=0.5)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             self.modes[:,mode])), color='g', alpha=0.7)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             self.pf_std[:,mode])), color='r', alpha=0.7)
        plt.plot(np.concatenate((np.zeros((self.caldays*390,)),\
                             -self.pf_std[:,mode])), color='r', alpha=0.7)
        plt.show()
        
    def read_compute_data(self, name):
        readarray = np.load(name + '.npy')
        self.modes = readarray[:,:,0]
        self.pf_std = readarray[:,:,1]
        self.modes_val = readarray[:,:,2]
        self.xi = readarray[:,:,3:]
        
        
    def write_compute_data(self, name):
        writearray = np.dstack((self.modes,self.pf_std,self.modes_val,self.xi))
        np.save(name, writearray)
        
        
        
        




#-----------------------------------------------------------------------------
# This is a child class for arbitrary holding periods
# (by default the positions are closed at the end of day)
#-----------------------------------------------------------------------------

class VGBM_backtest_overnight(VGBM_backtest):
    
    def pnl(self, mode = 0, cls = .15, exit = 50., opn = 1., \
            tr_percent = 0, tr_pershare = 0, close_after = 390):
        """Calculates the cumulative p&l over the
        entire range of dates. The portfolio will
        be continuously readjusted until the pf 
        crosses the trade open band at 
        opn*(standard deviation). The pf will then 
        be locked & tracked until it hits the 
        cls*std band (winning trade) or the 
        exit*std band (losing trade) or after a
        time close_after (win or lose).
        ------------------------------------------
        - cls: Close position/ cash out at +- cls*std
        - exit: Close position/ cut losses
        - opn: Open long/short position
        - wait: Wait n minutes at the beginning of
        each trading day before starting to trade
        - tr_percent: Transaction costs in percent
        - tr_pershare: Transaction costs in price
        per share
        - close_after: Close the position after n
        minutes
        """
        self.cls, self.exit, self.opn = cls, exit, opn
        X = self.Xdata.iloc[self.day1*390:390*self.day2].values
        phi = self.modes[:,mode]
        phi_val = self.modes_val[:,mode]
        xi = self.xi[:,:,mode]
        tr_fac = 1-tr_percent/100
        eps = tr_pershare
        
        opn = self.opn*self.pf_std[:,mode]
        cls = self.cls*self.pf_std[:,mode]
        exit = self.exit*self.pf_std[:,mode]
        
        pf_pos = None
        #lims = np.zeros((self.oos_mins,7))
        openval = 1E-10
        openprice = 0
        openxi = np.ones(self.dim)
        self.capital = np.ones(self.oos_mins)
        #poslist =  [] # for debugging
        open_at = [0]
        #close_at = [] # for debugging
        numberofdays = int(np.floor(self.oos_mins/390))
        #self.numberoftrades = 0
    

        for t in xrange(0,self.oos_mins):
            self.capital[t] = self.capital[t-1]
            if pf_pos == None and exit[t]>phi[t]>opn[t]: # short
                pf_pos = -1
                openprice = phi[t]
                openval = phi_val[t]
                openxi = xi[t]
                #poslist.append(pf_pos)
                open_at.append(t)
                #self.numberoftrades +=1
            if pf_pos == None and -exit[t]<phi[t]<-opn[t]: # long
                pf_pos = 1
                openprice = phi[t]
                openval = phi_val[t]
                openxi = xi[t]
                #poslist.append(pf_pos)
                open_at.append(t)
                #self.numberoftrades +=1
            if pf_pos == 1 and np.dot(X[t],openxi)>-cls[t]: # close long
                self.capital[t] =\
tr_fac*self.capital[t - 1]*(1+pf_pos*(np.dot(X[t],openxi)-openprice)/openval\
                            -2*eps*np.sum(abs(openxi))/openval)
                pf_pos = None
                #close_at.append(t)
                #self.numberoftrades +=1
            if pf_pos == -1 and np.dot(X[t],openxi)<cls[t]: # close short
                self.capital[t] =\
tr_fac*self.capital[t - 1]*(1+pf_pos*(np.dot(X[t],openxi)-openprice)/openval\
                            -2*eps*np.sum(abs(openxi))/openval)
                pf_pos = None
                #close_at.append(t)
                #self.numberoftrades +=1
            if t - open_at[-1] > close_after and (pf_pos == 1 or pf_pos == -1):
                self.capital[t] =\
tr_fac*self.capital[t - 1]*(1+pf_pos*(np.dot(X[t],openxi)-openprice)/openval\
                            -2*eps*np.sum(abs(openxi))/openval)
                pf_pos = None
                #close_at.append(t)
                #self.numberoftrades +=1
        #self.averagereturn = (self.capital[-1]/self.capital[0])**(1/self.numberoftrades)
        return self.capital#, open_at, close_at
    #, openprice, openval, openxi, poslist, np.dot(X,openxi)
    

#-----------------------------------------------------------------------------
# Sharpe ratio:
#-----------------------------------------------------------------------------

def sharpe(data,bm = 1.02):#data is array, 1 min data. return is annualized
    """ Data is an array of 1 min data. 
    benchmark is an effective interest rate. 
    Return is annualized. 
    """
    rets = np.delete((np.roll(data,-1)-data),-1)
    bm_rets = np.ones(len(rets))*(bm**(1/97500)-1)
    diff = rets-bm_rets
    return np.sqrt(97500)*np.mean(diff)/np.std(diff)


#-----------------------------------------------------------------------------
# Max drawdown:
#-----------------------------------------------------------------------------

def max_drawdown(pnl):
    peak = -9999
    MDD = 0
    for value in pnl:
        if value > peak:
            peak = value
        DD = 100.*(peak-value)/peak
        if DD > MDD:
            MDD = DD

    return MDD









########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np 

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)] # Why do we have M+1 states instead of just M? 
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]  # Why is this a set of strings...?
        # If there are M observations and each observation connects 2 steps, shouldn't we have M+1 steps? why are there only M?
        
        
        # Initialize the hidden state probabilities
        for i in range(self.L): 
            probs[0][i] = self.A_start[i]*self.O[i][x[0]] # Given the zeroth observation what's the prob that you moved from each state
            seqs[0][i] =''
    
        
        for obs in range(1,M):   # For each observation in the sequence (+1?)
            for stTo in range(self.L):# For each hidden state to
                maxval = probs[obs-1][0]*self.A[0][stTo]*self.O[stTo][x[obs]] 
                argmaxk = 0 
                # Calculate max: 
                for stFrom in range(self.L): # For each hidden state from
                    val = probs[obs-1][stFrom]*self.A[stFrom][stTo]*self.O[stTo][x[obs]]
                    if val > maxval: 
                        maxval = val 
                        argmaxk = stFrom
                # Update Next Probabilities
                probs[obs][stTo]= maxval
                seqs[obs][ stTo]= argmaxk

        max_seq = []

        probsEnd = max(probs[M-1]) # Final Probability
        max_seq.append(probs[M-1].index(probsEnd)) # Final Hidden State
        


        for stRev in range(M,1,-1) :
            max_seq.append(seqs[stRev-1][max_seq[M-stRev]])
        
        out =''
        for m in max_seq:
            out += str(m)

        out = out[len(out)::-1]

        
        return out


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M)] # I Changed the length of this from M+1 !#$!%!@%!($#%*#%#%)

                
        # Initialize the hidden state probabilities
        for i in range(self.L): 
            alphas [0][i] = self.A_start[i]*self.O[i][x[0]] # Given the zeroth observation what's the prob that you moved from each state
                    
        for obs in range(1,M):   
            for stTo in range(self.L):# For each hidden state from 
                val = 0
                for stFrom in range(self.L): 
                    val += alphas[obs-1][stFrom]*self.O[stTo][x[obs]]*self.A[stFrom][stTo ]
                    #print(alphas[obs-1][stFrom], '\t',self.O[stTo][x[obs]], '\t', self.A[stFrom][stTo ] )
                
                # Update Next Probabilities
                alphas[obs][stTo]= val

        # Normalization Term

            if normalize: 
            #for obs in range(len(alphas)):
            #    if sum(alphas[obs])!=0:
                alphas[obs] = [stProb/sum(alphas[obs]) for stProb in alphas[obs]]

        


 
        


        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M+1)] # Also changed this from M+1 like in forward 

        # Initialize 
        for i in range(self.L): 
            betas[M][i] = 1 # Given the zeroth observation what's the prob that you moved from each state
    


        for obs in range(M-1,-1,-1):   
            for stFrom in range(self.L):# For each hidden state from 
                val = 0
                for stTo in range(self.L): 
                    val += betas[obs+1][stTo]*self.O[stTo][x[obs]]*self.A[stFrom][stTo]
                # Update Next Probabilities
                betas[obs][stFrom]= val

            if normalize: 
                #for obs in range(len(betas)):
                #    if sum(betas[obs])!=0:
                betas[obs] = [stProb/sum(betas[obs]) for stProb in betas[obs]]
                #    else:
                #        print(betas)
        """

        for b in betas: 
            print(b)

        print('\t \t')
        for o in self.A:
            print(o)
        """
            # Normalization Term


        # Termination Step
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        A = []
        for b in range(self.L):
            arow = []
            for a in range(self.L):
                numerator = 0
                denominator = 0 
                for j in range(len(Y)):
                    for i in range(len(Y[j])-1):
                        numerator += (Y[j][i+1] == a)&(Y[j][i]==b) 
                        denominator += Y[j][i]==b 
                arow.append(numerator/denominator) 
            A.append(arow)
        self.A = A 

        # Calculate each element of O using the M-step formulas.
        O = []
        for b in range(self.L):
            orow = []
            for a in range(self.D):
                numerator = 0
                denominator = 0 
                for j in range(len(Y)):
                    for i in range(len(Y[j])):
                        numerator += (X[j][i] == a)&(Y[j][i]==b) 
                        denominator += Y[j][i]==b 
                orow.append(numerator/denominator) 
            O.append(orow)
        self.O = O 




    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''


        for n in range(N_iters):

            ## E-STEP##

            R = len(X)
            #### From their pseudocode
            A_num = np.zeros((self.L, self.L))
            A_den = np.zeros((self.L, self.L))
            O_num = np.zeros((self.L, self.D))
            O_den = np.zeros((self.L, self.D))
            ####

            for r in range(R):
                M = len(X[r]) # Again this is variable
                zi = np.zeros((M-1, self.L, self.L))

                alphas = np.asarray(self.forward( X[r], normalize=True))
                betas  = np.asarray(self.backward(X[r], normalize=True))
                betas=betas[1:]


                for m in range(M-1): 
                    # For all the states
                    for i in range(self.L):
                        # For all the states to, for zi's 3rd dimension
                        for j in range(self.L): 
                            zi[m,i,j] = alphas[m][j]*self.A[j][i]*betas[m+1][i]*self.O[i][X[r][m+1]]
            
                for t in range(M-1):
                    zi[t,:,:] = zi[t,:,:]/(np.sum(zi[t,:,:])+1e-8) # TODO               
            
                ## M-STEP ##



                #Calculate each element of A using the M-step formulas.
                

                for t in range(M):
                    if t<M-1:
                        A_num[:, :] += np.transpose(zi[t][:][:])
                        a_den_num =  np.multiply(alphas[t][:],betas[t][:])
                        a_den_den = np.dot(alphas[t], betas[t])
                        A_den += np.tile(np.expand_dims(a_den_num/a_den_den,1), self.L)
                    
                    num = np.multiply(alphas[t,:],betas[t,:])
                    den = np.dot(alphas[t], betas[t])
                    O_den += np.tile(np.expand_dims(num/den,1), self.D)
                    for vk in range(self.D): 
                        O_num[:,vk] += num*(X[r][t]==vk)/den
                        #O_den[:,vk] += num/den


            
            print(n, end=' ')
            self.A = np.divide(A_num, A_den) 
            self.O = np.divide(O_num,O_den)

        for i in range(10):
            for i in range(len(self.A)): 
                self.A[i,:] = self.A[i,:]/sum(self.A[i,:])
            for i in range(len(self.O)): 
                self.O[i,:] = self.O[i,:]/sum(self.O[i,:])


        pass

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        
        emission = []
        states = []


        stateOptions = []
        for i in range(self.L): 
            stateOptions.append(i)
        emissionOptions = []
        for j in range(self.D): 
            emissionOptions.append(j)

        # Find Initial State
        lastState = random.randint(0,self.L-1)

        for m in range(M): 
            nextStProb = []
            for stTo in range(self.L):
                nextStProb.append(self.A[lastState][stTo])
            states.append(np.random.choice(stateOptions, p=nextStProb))

            nextEmitProb = []
            for emit in range(self.D):
                nextEmitProb.append(self.O[lastState][emit])
            emission.append(np.random.choice(emissionOptions, p=nextEmitProb))

            lastState = states[-1]

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM

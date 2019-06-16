#!/usr/bin/python3

import sys
import random
import math
from math import log as log

import numpy

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 7
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
# <Your feedback goes here>
#####################################################
#####################################################



# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0

class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior = [0.5, 0.5]
        self.transition = [[0.999, 0.001], [0.01, 0.99]]
        self.emission = [{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                         {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}]
                        
        self.possible_transitions = [[0,0], [0,1], [1,0], [1,1]]

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Computes the (natural) log probability of sequence given a sequence of states.
    def logprob(self, sequence, states):
        ###########################################
        # Start your code
        # print("My code here")

        #init state probability
        prob = log(self.prior[states[0]]) + log(self.emission[states[0]][sequence[0]])


        #loop through all the states/sequences and add the log of all the probabilities
        #each iterate has a transition probability and an emisson probability
        for iter in range(1, len(sequence)):

            currentState = states[iter]
            prevState = states[iter - 1]
            currentChar = sequence[iter]

            #probability of emitting this character in this state
            currentState_EmissionProb = log(self.emission[currentState][currentChar])

            #probability of transitioning from prevState to the CurrentState
            transitionProbfromPrevState = log(self.transition[prevState][currentState])

            #add the log of these probabilities to prob to accumulate
            prob += currentState_EmissionProb 
            prob += transitionProbfromPrevState


        #print(prob)

        return prob
        #print(sequence)
        #print(states)
        # End your code
        ###########################################


    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    
    def viterbi(self, sequence):
        ###########################################
        #Start your code
        #print("My code here")
        #init structures
        M = numpy.zeros(( len(sequence), self.num_states))
        Prev = numpy.zeros(( len(sequence), self.num_states), dtype = numpy.int64)
        #print(M)

        #probability of going to state 0 and emitting the first character in the sequence
        M[0][0] = log(self.prior[0]) + log(self.emission[0][sequence[0]])
        #probability of going to state 1 and emitting the first character in the sequence
        M[0][1] = log(self.prior[1]) + log(self.emission[1][sequence[1]])

        for seq_index in range(1, len(sequence)):

            #the current acid in the sequence, A C T or G
            char = sequence[seq_index]

            #probability of emiting this character in state 0 
            Emission_0 = log(self.emission[0][char])
            
            #probability of emiting this character in state 1
            Emission_1 = log(self.emission[1][char])

            #probability of starting in state 0 and staying in state 0 and emitting this character
            prob_0_to_0 = Emission_0 + log(self.transition[0][0]) + M[seq_index - 1][0]

            #probability of starting in state 0 and transitioning to state 1 and emitting this char
            prob_0_to_1 = Emission_1 + log(self.transition[0][1]) + M[seq_index - 1][0]

            #probability of starting in state 1 and transitioning to state 0 and emitting this char
            prob_1_to_0 = Emission_0 + log(self.transition[1][0]) + M[seq_index - 1][1]

            #probability of starting in state 1 and staying in state 1 and emitting this character
            prob_1_to_1 = Emission_1 + log(self.transition[1][1]) + M[seq_index - 1][1]

            #update probability table
            #the more probable event of staying in state 0 vs transitioning from state 1
            if(prob_0_to_0 > prob_1_to_0):
                M[seq_index][0] = prob_0_to_0 # + M[seq_index - 1][0]
            else:
                M[seq_index][0] = prob_1_to_0 # + M[seq_index - 1][1]

            #the more probable event of staying in state 1 vs transitioning from state 0
            if(prob_0_to_1 > prob_1_to_1):
                M[seq_index][1] = prob_0_to_1 # + M[seq_index - 1][0]
            else:
                M[seq_index][1] = prob_1_to_1 # + M[seq_index - 1][1]

            #keep track of which previous state was more likely
            if (prob_0_to_0 > prob_1_to_0):
                Prev[seq_index][0] = 0
            else:
                Prev[seq_index][0] = 1

            if (prob_0_to_1 > prob_1_to_1):
                Prev[seq_index][1] = 0
            else:
                Prev[seq_index][1] = 1

        #print(M)
        #print(Prev)

        return generatesequence(M, Prev)
        # End your code
        ###########################################

#returns list of highest prob sequence of events
def generatesequence(probtable, prev):

    path = []

    if (probtable[0][-1] > probtable[1][-1]):
        col = 0
    else:
        col = 1

    path.append(col)

    #iterating backwards, using what state is in the current index of Prev as the next destination
    for iter in range (len(probtable)- 1, 0, -1):

        path.append(prev[iter][col])

        if prev[iter][col] == 1:
            col = 1
        else:
            col = 0

    #iterated backwards so we need to flip the list of states
    path.reverse()

    #print(path)

    return path


def read_sequence(filename):
    with open(filename, "r") as f:
        return f.read().strip()

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, logprob, states):
    with open(filename, "w") as f:
        f.write(str(logprob))
        f.write("\n")
        for state in range(2):
            f.write(str(states.count(state)))
            f.write("\n")
        f.write("".join(map(str, states)))
        f.write("\n")

hmm = HMM()

file = sys.argv[1]
sequence = read_sequence(file)
viterbi = hmm.viterbi(sequence)
logprob = hmm.logprob(sequence, viterbi)
name = "my_"+file[:-4]+'_output.txt'
write_output(name, logprob, viterbi)



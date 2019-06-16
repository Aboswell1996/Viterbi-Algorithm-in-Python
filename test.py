import random
import math
import copy

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 0
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
    #0.5*0.169*0.01*0.169
    def __init__(self):
        self.num_states = 2
        self.prior = [0.5, 0.5]
        self.transition = [[0.999, 0.001], [0.01, 0.99]]
        self.emission = [{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                         {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}]
        self.path=[]
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
        probability =[]
        current_state = states[0]
        current_obs = sequence[0]
        probability = math.log(self.prior[current_state]) + math.log(self.emission[current_state][current_obs])
        for index in range(1,len(sequence)):

            current_obs = sequence[index]
            current_state = states[index]
            prev_state = states[index -1]
            trans_prob = math.log(self.transition[prev_state][current_state])
            #trans_prob = self.prob_of_path((prev_state,current_state),probability[index-1],current_obs)
            emission_prob = math.log(self.emission[current_state][current_obs])
            probability =  trans_prob + emission_prob + probability

        return probability
        # End your code
        ###########################################


    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def viterbi(self, sequence):
        ###########################################
        # Start your cod
        steps = [[0 for x in range(2)] for y in range(len(sequence))]
        prev_table=  [[0 for x in range(2)] for y in range(len(sequence))]
        prev_table[0][0] = -1
        prev_table[0][1] = -1

        steps[0][0]= math.log(self.prior[0]) + math.log(self.emission[0][sequence[0]])
        steps[0][1] = math.log(self.prior[1]) + math.log(self.emission[1][sequence[0]])
        #print(steps)

        for sym in range(1,len(sequence)):
            # 1= move, 0 = stay
            #probability of emitting T under state 0 or 1
            emission_H = math.log(self.emission[1][sequence[sym]])
            emission_L = math.log(self.emission[0][sequence[sym]])

            #L_to_L = math.log(steps[sym-1][0],2) + math.log(self.transition[0][1],2) + math.log(self.emission[0][sequence[sym]],2)
            #probability of staying in state 0 + log of previous state
            L_to_L = self.prob_of_path((0,0),
                                       steps[sym-1][0],
                                       sequence[sym]
                                       )
           # H_to_L = steps[sym-1][1] * self.transition[1][1] *self.emission[0][sequence[sym]]
           #probability of going to state 0 from 1 + log(previous char)
            H_to_L = self.prob_of_path((1,0),
                                       steps[sym-1][1],
                                       sequence[sym])
           # H_to_H = steps[sym-1][1] * self.transition[1][0] *self.emission[1][sequence[sym]]
            H_to_H = self.prob_of_path((1,1),
                                       steps[sym-1][1],
                                       sequence[sym])
           # L_to_H = steps[sym-1][0] * self.transition[0][1] *self.emission[1][sequence[sym]]
            L_to_H = self.prob_of_path((0,1),
                                       steps[sym-1][0],
                                       sequence[sym])

            steps[sym][0] = emission_L + max(L_to_L , H_to_L)
            steps[sym][1] = emission_H + max(L_to_H , H_to_H)
            #came from L
            prev_table[sym][0] =  0 if  max(L_to_L, H_to_L) == L_to_L else 1
            #came from H

            prev_table[sym][1] =  0 if max(L_to_H , H_to_H) == L_to_H else 1
        print(steps)
        print(prev_table)
        return self.back_track(steps,prev_table)

                # need to consider transition probabilities

    def back_track(self,prob_table,prev_table):
        path = []
        #find max for last element of sewuence
        last_sym = prob_table[len(prob_table)-1]
        #print(last_sym)
        start_index = 0
        if(max(last_sym[0],last_sym[1])) == last_sym[0]:
            start_index = 0
            path.append(0)
        else:
            start_index = 1
            path.append(1)
        #prev_table.reverse()
        for index in range(len(prev_table)-1,0,-1):

            prev_node = prev_table[index][start_index]
            path.append( prev_node)
            start_index = prev_node
        path.reverse()
        #print(path)
        return path

    def prob_of_path(self,path,prev_probablity,sym):
        start = path[0]
        end = path[1]
        return prev_probablity + math.log(self.transition[start][end])

    def getSequence(self,path):
        outputsequence=""
        for tup in path:
            outputsequence = outputsequence + tup[0]
            print(outputsequence)
        return outputsequence
    def obtain_max(self,probL, probR):
        if max(probL, probR) == probL:
            return ('0', probL)
        else:
            return ('1', probR)

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

sequence = read_sequence("small.txt")
viterbi = hmm.viterbi(sequence)
logprob = hmm.logprob(sequence, viterbi)

write_output("test_op.txt",logprob, viterbi)
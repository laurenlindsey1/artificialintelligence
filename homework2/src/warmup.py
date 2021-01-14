'''
warmup.py

Skeleton for answering warmup questions related to the
AdAgent assignment. By the end of this section, you should
be familiar with:
- Importing, selecting, and manipulating data using Pandas
- Creating and Querying a Bayesian Network
- Using Samples from a Bayesian Network for Approximate Inference

@author: Lauren Lindsey, Shanaya Nagendran, Maya Pegler-Gordon
'''
import pandas as pandas
from pomegranate import *

if __name__ == '__main__':
    """
    PROBLEM 2.1
    Using the Pomegranate Interface, determine the answers to the
    queries specified in the instructions.
    
    ANSWER GOES BELOW:
    𝑃(𝑆)
    S = 0: 0.46533620497832795
    S = 1: 0.27472136125259405
    S = 2: 0.2599424337690779

    𝑃(𝑆|𝐺=1)
    S = 0: 0.5417396233612717
    S = 1: 0.24638223081938232
    S = 2: 0.21187814581934603
    
    𝑃(𝑆|𝑇=1,𝐻=1)
    S = 0: 0.4017071051761353
    S = 1: 0.31237085652590424
    S = 2: 0.28592203829796037

    """
    X = pandas.read_csv("../dat/adbot-data.csv") 
    model = BayesianNetwork.from_samples(X, algorithm='exact')

    # 𝑃(𝑆)
    prob1 = model.predict_proba([[None, None, None, None, None, None, None, None, None, None]])
    val1_0 = prob1[0][5].parameters[0][0.0]
    print("P(S=0): ", val1_0)
    val1_1 = prob1[0][5].parameters[0][1.0]
    print("P(S=1): ", val1_1)
    val1_2 = prob1[0][5].parameters[0][2.0]
    print("P(S=2): ", val1_2, "\n")

    # 𝑃(𝑆|𝐺=1)
    prob2 = model.predict_proba([[None, None, None, None, 1, None, None, None, None, None]])
    val2_0 = prob2[0][5].parameters[0][0.0]
    print("P(S=0|G=1): ", val2_0)
    val2_1 = prob2[0][5].parameters[0][1.0]
    print("P(S=1|G=1): ", val2_1)
    val2_2 = prob2[0][5].parameters[0][2.0]
    print("P(S=2|G=1): ", val2_2, "\n")
    
    # 𝑃(𝑆|𝑇=1,𝐻=1)
    prob3 = model.predict_proba([[None, None, None, None, None, None, 1, None, 1, None]])
    val3_0 = prob3[0][5].parameters[0][0.0]
    print("P(S=0|T=1,H=1): ", val3_0)
    val3_1 = prob3[0][5].parameters[0][1.0]
    print("P(S=1|T=1,H=1): ", val3_1)
    val3_2 = prob3[0][5].parameters[0][2.0]
    print("P(S=2|T=1,H=1): ", val3_2, "\n")
    

'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.

@author: Maya Pegler-Gordon, Shanaya Nagendran, Lauren Lindsey
'''
import pandas as pd
from pomegranate import *
import math
import itertools
import unittest
import numpy as np

class AdEngine:

    def __init__(self, data_file, dec_vars, util_map):
        
        """
        Responsible for initializing the Decision Network of the
        AdEngine using the following inputs
        
        :param string data_file: path to csv file containing data on which
        the network's parameters are to be learned
        :param list dec_vars: list of string names of variables to be
        considered decision variables for the agent. Example:
          ["Ad1", "Ad2"]
        :param dict util_map: discrete, tabular, utility map whose keys
        are variables in network that are parents of a utility node, and
        values are dictionaries mapping that variable's values to a utility
        score, for example:
          {
            "X": {0: 20, 1: -10}
          }
        represents a utility node with single parent X whose value of 0
        has a utility score of 20, and value 1 has a utility score of -10
        """
        X = pd.read_csv(data_file) 
        self.dec_vars = dec_vars
        self.dec_vars_values = []
        self.util_map = util_map
        for v in self.dec_vars:
            self.dec_vars_values.append(np.unique(X[v]))
        self.state_names = X.columns.tolist()
        self.model = BayesianNetwork.from_samples(X, algorithm='exact', state_names = self.state_names)
        
    def meu(self, evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, selects the ad content that maximizes expected utility
        and returns a dictionary over any decision variables and their
        best values plus the MEU from making this selection.
        
        :param dict evidence: dict mapping network variables to their
        observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: 2-Tuple of the format (a*, MEU) where:
          - a* = dict of format: {"DecVar1": val1, "DecVar2": val2, ...}
          - MEU = float representing the EU(a* | evidence)
        MEU = max(EU(a | e) = sum(s in S) P(s | a, e) U(s, a))
        """
        best_util = -math.inf
        best_decisions = {}
        dec_dict = {}
        
        combos = itertools.product(*self.dec_vars_values)
        
        for combo in combos:
            if "nan" not in str(combo):
                # add decisions to dict 
                for i in range(len(self.dec_vars)):
                    dec_dict[self.dec_vars[i]] = combo[i]
                new_evidence = {**dec_dict, **evidence}
                
                # get the utility letter from util_map
                util_letter = ""
                for key in self.util_map:
                    util_letter = key
                    
                # util index: where in the probability map the values are located
                util_index = self.state_names.index(util_letter)
                prob = self.model.predict_proba(new_evidence)
                
                # narrow down to just a dictionary of keys and values {0.0: val, 1.0:, val}
                prob_shorter = prob[util_index].parameters[0]  
                curr_util = 0
                
                for key in self.util_map[util_letter]:
                    probability = prob_shorter[key]
                    utility = self.util_map[util_letter][key]
                    curr_util += (probability * utility)
                if curr_util > best_util:
                    best_util = curr_util
                    best_decisions = {}
                    for key in dec_dict:
                        best_decisions[key] = dec_dict[key]

        return (best_decisions, best_util)

    def vpi(self, potential_evidence, observed_evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.
        
        :param string potential_evidence: string representing the variable name
        of the variable under evaluation
        :param dict observed_evidence: dict mapping network variables 
        to their observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: float value indicating the VPI(potential | observed)
        """
           
        # will this be a problem if the evidence has values besides 0,1?
        zero = {potential_evidence: 0}
        one = {potential_evidence: 1}
        evidence_zero = self.meu({**observed_evidence, **zero})[1]
        evidence_one = self.meu({**observed_evidence, **one})[1]   
        
        evidence_index = self.state_names.index(potential_evidence)
        prob = self.model.predict_proba(observed_evidence)
        prob_shorter = prob[evidence_index].parameters[0]
            
        prob_0 = prob_shorter[0.0]
        prob_1 = prob_shorter[1.0]

        evidence_probs = (evidence_zero * prob_0) +  (evidence_one * prob_1)
        no_evidence = self.meu(observed_evidence)[1]
        vpi = evidence_probs - no_evidence
        if vpi < 0:
            vpi = 0
        return vpi

class AdAgentTests(unittest.TestCase):
    
    def test_meu_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 0}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)
    
    def test_meu_lecture_example_with_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {"X": 0}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 1}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)
        
        evidence2 = {"X": 1}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"D": 0}, decision2[0])
        self.assertAlmostEqual(2.4, decision2[1], delta=0.01)

    def test_vpi_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        vpi = ad_engine.vpi("X", evidence)
        self.assertAlmostEqual(0.24, vpi, delta=0.1)
    
    def test_meu_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 0}, decision[0])
        self.assertAlmostEqual(746.72, decision[1], delta=0.01)
    
    def test_meu_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 1}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 1}, decision[0])
        self.assertAlmostEqual(720.73, decision[1], delta=0.01)
        
        evidence2 = {"T": 0, "G": 0}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"Ad1": 0, "Ad2": 0}, decision2[0])
        self.assertAlmostEqual(796.82, decision2[1], delta=0.01)
        
    def test_vpi_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(20.77, vpi, delta=0.1)
        
        vpi2 = ad_engine.vpi("F", evidence)
        self.assertAlmostEqual(0, vpi2, delta=0.1)
        
    def test_vpi_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 0}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(25.49, vpi, delta=0.1)
        
        evidence2 = {"G": 1}
        vpi2 = ad_engine.vpi("P", evidence2)
        self.assertAlmostEqual(0, vpi2, delta=0.1)
        
        evidence3 = {"H": 0, "T": 1, "P": 0}
        vpi3 = ad_engine.vpi("G", evidence3)
        self.assertAlmostEqual(66.76, vpi3, delta=0.1)
        
if __name__ == '__main__':
    unittest.main()

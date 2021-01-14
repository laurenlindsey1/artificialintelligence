'''
maze_clause.py

Specifies a Propositional Logic Clause formatted specifically
for Grid Maze Pathfinding problems. Clauses are a disjunction of
MazePropositions (2-tuples of (symbol, location)) mapped to
their negated status in the sentence.
'''
import unittest

class MazeClause:
    
    def __init__(self, props):
        """
        Constructor parameterized by the propositions within this clause;
        argument props is a LIST of MazePropositions, like:
        [(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)]
        
        :props: a list of tuples formatted as: (MazeProposition, NegatedBoolean)
        """       
        
        self.props = {}
        self.valid = False
        if props != []:
            for clause in props:
                location = clause[0]
                if location in self.props and self.props.get(location) != clause[1]: # inverse
                    self.valid = True
                    self.props.pop(location)
                    break
                else:
                    self.props[location] = clause[1]
    
    def get_prop(self, prop):         
        """
        Returns:
          - None if the requested prop is not in the clause
          - True if the requested prop is positive in the clause
          - False if the requested prop is negated in the clause
          
        :prop: A MazeProposition as a 2-tuple formatted as: (Symbol, Location),
        for example, ("P", (1, 1))
        """
        
        if prop in self.props:
            return self.props.get(prop)
        else:
            return None
    def is_valid(self):
        return self.valid
    
    def is_empty(self):
        if self.props == {}:
            return True
        else:
            return False
    
    def __eq__(self, other):
        """
        Defines equality comparator between MazeClauses: only if they
        have the same props (in any order) or are both valid
        """
        return self.props == other.props and self.valid == other.valid
    
    def __hash__(self):
        """
        Provides a hash for a MazeClause to enable set membership
        """
        # Hashes an immutable set of the stored props for ease of
        # lookup in a set
        return hash(frozenset(self.props.items()))
    
    # Hint: Specify a __str__ method for ease of debugging (this
    # will allow you to "print" a MazeClause directly to inspect
    # its composite literals)
    # def __str__ (self):
    #     return ""
    
    @staticmethod
    def resolve(c1, c2):
        """
        Returns a set of MazeClauses that are the result of resolving
        two input clauses c1, c2 (Hint: result will only ever be a set
        of 0 or 1 MazeClause, but it being a set is convenient for the
        inference engine) (Hint2: returning an empty set of clauses
        is different than returning a set containing the empty clause /
        contradiction)
        
        :c1: A MazeClause to resolve with c2
        :c2: A MazeClause to resolve with c1
        """
        results = set()
        found_inverse = False
        final_clause = []
        
        for clause in c1.props:
            if (clause, c1.props.get(clause)) in final_clause or (clause, not c1.props.get(clause)) in final_clause:
                if (clause, c1.props.get(clause)) not in final_clause and not found_inverse: # inverse
                    final_clause.remove(clause, c1.props.get(clause))
                    found_inverse = True
                elif (clause, c1.props.get(clause)) not in final_clause and found_inverse:
                    new_clause = (clause, c1.props.get(clause))
                    final_clause.append(new_clause)
            elif clause in c2.props:
                if c1.props.get(clause) != c2.props.get(clause) and not found_inverse: # inverse
                    c2.props.pop(clause)
                    found_inverse = True
                elif c1.props.get(clause) != c2.props.get(clause) and  found_inverse: # inverse
                    new_clause = (clause, c1.props.get(clause))
                    final_clause.append(new_clause)
            else:
                new_clause = (clause, c1.props.get(clause))
                final_clause.append(new_clause)
                
        for clause in c2.props: 
            new_clause = (clause, c2.props.get(clause))
            final_clause.append(new_clause)
            
        # no resolution was made, return nothing
        if not found_inverse:
            return results
        
        # only add to results IF it is not valid
        end = MazeClause(final_clause)
        if not end.valid:
            results.add(end)
        return results
    

class MazeClauseTests(unittest.TestCase):
    def test_mazeprops1(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)])
        self.assertTrue(mc.get_prop(("X", (1, 1))))
        self.assertTrue(mc.get_prop(("X", (2, 1))))
        self.assertFalse(mc.get_prop(("Y", (1, 2))))
        self.assertTrue(mc.get_prop(("X", (2, 2))) is None)
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops2(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (1, 1)), True)])
        self.assertTrue(mc.get_prop(("X", (1, 1))))
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops3(self):
        mc = MazeClause([(("X", (1, 1)), True), (("Y", (2, 1)), True), (("X", (1, 1)), False)])
        self.assertTrue(mc.is_valid())
        self.assertTrue(mc.get_prop(("X", (1, 1))) is None)
        self.assertFalse(mc.is_empty()) 
        
    def test_mazeprops4(self):
        mc = MazeClause([])
        self.assertFalse(mc.is_valid())
        self.assertTrue(mc.is_empty())
    
    def test_mazeprops5(self):
        mc1 = MazeClause([(("X", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0) 
        
    def test_mazeprops6(self):
        mc1 = MazeClause([(("X", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([]) in res)
        
    def test_mazeprops7(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (2, 2)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("Y", (1, 1)), True), (("Y", (2, 2)), True)]) in res)
        
    def test_mazeprops8(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops9(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False), (("Z", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True), (("W", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops10(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False), (("Z", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), False), (("W", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True), (("W", (1, 1)), False)]) in res)
if __name__ == "__main__":
    unittest.main()
    
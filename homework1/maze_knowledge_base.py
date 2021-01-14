'''
maze_knowledge_base.py

Specifies a simple, Conjunctive Normal Form Propositional
Logic Knowledge Base for use in Grid Maze pathfinding problems
with side-information.
'''
from maze_clause import MazeClause
from itertools import product 

import unittest


class MazeKnowledgeBase:

    def __init__(self):
        self.clauses = set()

    def tell(self, clause):
        """
        Adds the given clause to the CNF MazeKnowledgeBase
        Note: we expect that no clause added this way will ever
        make the KB inconsistent (you need not check for this)
        """

        self.clauses.add(clause)

        return

    def ask(self, query):
        """
        Given a MazeClause query, returns True if the KB entails
        the query, False otherwise
        """
 
        new_kb = list(self.clauses)
       
        # add ATTC to new_kb
        for keys in query.props:
            query_negated = MazeClause([(keys, not query.props.get(keys))])
            pairs = list(product(new_kb, [query_negated]))
            new_kb.append(query_negated)
        
        while len(pairs) != 0:
            resolvent = MazeClause.resolve(pairs[0][0], pairs[0][1])
            
            if MazeClause([]) in resolvent:
                return True
            
            # something was resolved
            if len(resolvent) != 0:
                for keys in resolvent:
                    pairs = pairs + list(product(new_kb, [keys])) 
                    new_kb.append(keys)
            pairs.pop(0)        
        return False

class MazeKnowledgeBaseTests(unittest.TestCase):
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))
        
    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))
        
    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))


if __name__ == "__main__":
    unittest.main()
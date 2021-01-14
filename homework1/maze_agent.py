'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
'''

import time
from pathfinder import *
from maze_problem import *
from queue import Queue
from maze_knowledge_base import MazeKnowledgeBase
from maze_clause import MazeClause

class MazeAgent:
    
    ##################################################################
    # Constructor
    ##################################################################
    
    def __init__ (self, env):
       
        self.loc  = env.get_player_loc()
        self.goal = env.get_goal_loc()
    
        self.maze = env.get_agent_maze()
        
        self.plan = Queue()
        
        self.kb = MazeKnowledgeBase()

        self.poss_pit = set()

        self.not_pits = set()
    
    ##################################################################
    # Methods
    ##################################################################
    
    def think(self, perception):
        """
        think is parameterized by the agent's perception of the tile type
        on which it is now standing, and is called during the environment's
        action loop. This method is the chief workhorse of your MazeAgent
        such that it must then generate a plan of action from its current
        knowledge about the environment.
        
        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """
        self.kb.tell(MazeClause([(("P",(self.goal[0], self.goal[1])), False)]))
        self.loc = perception.get("loc")
        if perception.get('tile') == "G":
            return
        
        if perception.get('tile') == "1":
            # Tell KB that pit = true at up or down or left or right from current tile
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] + 1)), True), 
            (("P",(perception.get("loc")[0], perception.get("loc")[1] - 1)), True),
            (("P",(perception.get("loc")[0] + 1, perception.get("loc")[1])), True), 
            (("P",(perception.get("loc")[0] - 1, perception.get("loc")[1])), True)]))
            
            self.check_out_of_bounds(perception, 1)            
            
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1])), False)]))            
            self.poss_pit.add((perception.get("loc")[0], perception.get("loc")[1] + 1))
            self.poss_pit.add((perception.get("loc")[0], perception.get("loc")[1] - 1))
            self.poss_pit.add((perception.get("loc")[0] + 1, perception.get("loc")[1]))
            self.poss_pit.add((perception.get("loc")[0] - 1, perception.get("loc")[1]))
            
             # Remove anything from poss pit that is definitely not a pit
            updated_pits = set()
            for pit in self.poss_pit:
                if pit not in self.not_pits:
                    updated_pits.add(pit)
            self.poss_pit = updated_pits
                    
            for possible_pit in self.poss_pit:
                if (self.kb.ask(MazeClause([(("P", (possible_pit[0], possible_pit[1])), True)]))):
                    self.update_tile("P", possible_pit)
                else:
                    self.update_tile("M", possible_pit)
            
        if perception.get('tile') == "2":
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] + 2)), True), 
            (("P", (perception.get("loc")[0], perception.get("loc")[1] - 2)), True), 
            (("P", (perception.get("loc")[0] + 2, perception.get("loc")[1])), True), 
            (("P", perception.get("loc")[0] - 2, perception.get("loc")[1]), True)]))
            
            self.check_out_of_bounds(perception, 2)
           
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1])), False)]))
            
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] + 1)), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] - 1)), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] + 1, perception.get("loc")[1])), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] - 1, perception.get("loc")[1])), False)]))
            
            self.poss_pit.add((perception.get("loc")[0], perception.get("loc")[1] + 2))
            self.poss_pit.add((perception.get("loc")[0], perception.get("loc")[1] - 2))
            self.poss_pit.add((perception.get("loc")[0] + 2, perception.get("loc")[1]))
            self.poss_pit.add((perception.get("loc")[0] - 2, perception.get("loc")[1]))
           
            # Remove anything in poss pit that is definitely not a pit
            updated_pits = set()
            for pit in self.poss_pit:
                updated_pits.add(pit)
            self.poss_pit = updated_pits
                    
            for possible_pit in self.poss_pit:
                if (self.kb.ask(MazeClause([(("P", (possible_pit[0], possible_pit[1])), True)]))):
                    self.update_tile("P", possible_pit)
                else:
                    self.update_tile("M", possible_pit)
    
        if perception.get('tile') == ".":
            # Put C to any +-1 and +-2 adjacent files 
            self.update_tile("C", (perception.get("loc")[0] + 1, perception.get("loc")[1]))
            self.update_tile("C", (perception.get("loc")[0] - 1, perception.get("loc")[1]))
            self.update_tile("C", (perception.get("loc")[0], perception.get("loc")[1] + 1))
            self.update_tile("C", (perception.get("loc")[0], perception.get("loc")[1] - 1))
            self.update_tile("C", (perception.get("loc")[0] + 2, perception.get("loc")[1]))
            self.update_tile("C", (perception.get("loc")[0] - 2, perception.get("loc")[1]))
            self.update_tile("C", (perception.get("loc")[0], perception.get("loc")[1] + 2))
            self.update_tile("C", (perception.get("loc")[0], perception.get("loc")[1] - 2))
            self.not_pits.add((perception.get("loc")[0] + 1, perception.get("loc")[1]))
            self.not_pits.add((perception.get("loc")[0] - 1, perception.get("loc")[1]))
            self.not_pits.add((perception.get("loc")[0], perception.get("loc")[1] + 2))
            self.not_pits.add((perception.get("loc")[0], perception.get("loc")[1] - 2))
            self.not_pits.add((perception.get("loc")[0] + 2, perception.get("loc")[1]))
            self.not_pits.add((perception.get("loc")[0] - 2, perception.get("loc")[1]))
            self.not_pits.add((perception.get("loc")[0], perception.get("loc")[1] + 2))
            self.not_pits.add((perception.get("loc")[0], perception.get("loc")[1] - 2))

            # Add to KB that there are no pits +- from 1 around it
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] + 1, perception.get("loc")[1])), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] - 1, perception.get("loc")[1])), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] + 1)), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] - 1)), False)]))     
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] + 2, perception.get("loc")[1])), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] - 2, perception.get("loc")[1])), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] + 2)), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] - 2)), False)]))
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1])), False)]))            
        if perception.get('tile') == "P":
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1])), True)]))
        
        self.poss_pit.clear()

        maze_problem = MazeProblem(self.maze)
        
        # Pathfind to the best safe spot
        smallest = len(self.maze) * 5
        tile = self.goal
        for tiles in self.not_pits:
            man_dist = abs(self.goal[0] - tiles[0]) + abs(self.goal[1] - tiles[1])
            if man_dist < smallest:
                smallest = man_dist
                tile = tiles    
        
        next_step = pathfind(maze_problem, self.loc, tile)

        new_plan = Queue()
        for i in next_step[1]:
            new_plan.put(i)

        if new_plan.empty():
            next_step = pathfind(maze_problem, self.loc, self.goal)
            for i in next_step[1]:
                new_plan.put(i)
        self.plan = new_plan

    def update_tile(self, tile_type, tile_location):
        if tile_location[0] > 0 and tile_location[1] > 0 and tile_location[1] < len(self.maze) and tile_location[0] < len(self.maze[0]):
            if self.maze[tile_location[1]][tile_location[0]] == "?":
                self.maze[tile_location[1]][tile_location[0]] = tile_type

    def check_out_of_bounds(self, perception, diff):
        if perception.get("loc")[1] - diff <= 0:
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] - diff)), False)]))            
        if perception.get("loc")[0] - diff <= 0:
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] - diff, perception.get("loc")[1])), False)]))            
        if perception.get("loc")[1] + diff >= len(self.maze) - 1:
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0], perception.get("loc")[1] + diff)), False)]))            
        if perception.get("loc")[0] + diff >= len(self.maze[0]) - 1:
            self.kb.tell(MazeClause([(("P",(perception.get("loc")[0] + diff, perception.get("loc")[1])), False)]))
    
    def get_next_move(self):
        """
        Returns the next move in the plan, if there is one, otherwise None
        [!] You should NOT need to modify this method -- contact Dr. Forney
            if you're thinking about it
        """
        return None if self.plan.empty() else self.plan.get()
    
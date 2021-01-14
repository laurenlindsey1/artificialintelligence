from constants import Constants

class NewConstants:
    
    @staticmethod
    def get_potential_pit_penalty ():
        return 10

    @staticmethod
    def get_curious_penalty ():
        return -100
    
    @staticmethod
    def get_goal_penalty ():
        return 0.1
    
    MAYBE_PIT_BLOCK = "M"
    CURIOUS_BLOCK = "C"
    
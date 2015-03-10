#!/usr/bin/env python

class CMTSolution(object):
    
    def __init__(self):
        self.half_duration = 3.805
        self.source_decay_mimic_triangle = 1.6280
        self.alpha = self.source_decay_mimic_triangle / self.half_duration
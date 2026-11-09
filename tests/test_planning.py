import itertools
import math
import random
import unittest
from types import SimpleNamespace
from planning import knapsack_select
from scripts.score_solvability import wilson_interval


class PlanningTests(unittest.TestCase):
    def test_knapsack_recovers_combination_missed_by_density(self):
        tasks = [SimpleNamespace(estimated_cost=c,reward=r) for c,r in [(10,60),(20,100),(30,120)]]
        chosen = knapsack_select(tasks,50,quantum=1,margin=1)
        self.assertEqual(sum(t.reward for t in chosen),220)
        self.assertEqual(sum(t.estimated_cost for t in chosen),50)

    def test_exact_solver_matches_exhaustive_search(self):
        rng=random.Random(17)
        for _ in range(70):
            tasks=[SimpleNamespace(estimated_cost=rng.randint(1,20),reward=rng.randint(1,15)) for _ in range(8)]
            budget=rng.randint(1,60)
            optimum=max(sum(tasks[i].reward for i in range(8) if mask>>i&1)
                for mask in range(256) if sum(tasks[i].estimated_cost for i in range(8) if mask>>i&1)<=budget)
            selected=knapsack_select(tasks,budget,quantum=1,margin=1)
            self.assertEqual(sum(t.reward for t in selected),optimum)

    def test_rounding_is_conservative_and_empty_budget_is_safe(self):
        tasks=[SimpleNamespace(estimated_cost=10.1,reward=3) for _ in range(10)]
        selected=knapsack_select(tasks,57)
        self.assertLessEqual(sum(t.estimated_cost*1.25 for t in selected),57)
        self.assertEqual(knapsack_select(tasks,0),[])
        with self.assertRaises(ValueError): knapsack_select(tasks,float('inf'))

    def test_four_successes_do_not_imply_certainty(self):
        low,high=wilson_interval(4,4)
        self.assertAlmostEqual(low,0.510109,places=5)
        self.assertAlmostEqual(high,1)
        self.assertIsNone(wilson_interval(0,0))

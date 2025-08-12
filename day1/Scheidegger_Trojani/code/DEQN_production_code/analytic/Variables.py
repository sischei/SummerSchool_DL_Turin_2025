# -*- coding: utf-8 -*-

states = [{'name': 'TFP'},
  {'name': 'depr'},
  {'name': 'K2'},
  {'name': 'K3'},
  {'name': 'K4'},
  {'name': 'K5'},
  {'name': 'K6'},
  {'name': 'K_total'},
  {'name': 'r'},
  {'name': 'w'},
  {'name': 'Y'},
  {'name': 'fw2'},
  {'name': 'fw3'},
  {'name': 'fw4'},
  {'name': 'fw5'},
  {'name': 'fw6'}]

policies = [{'name': 'a1', 'bounds': {'lower': 1e-5}},
  {'name': 'a2', 'bounds': {'lower': 1e-5}},
  {'name': 'a3', 'bounds': {'lower': 1e-5}},
  {'name': 'a4', 'bounds': {'lower': 1e-5}},
  {'name': 'a5', 'bounds': {'lower': 1e-5}}]
 
definitions = [{'name': 'c1', 'bounds': {'lower': 1e-4}},
  {'name': 'c2', 'bounds': {'lower': 1e-4}},
  {'name': 'c3', 'bounds': {'lower': 1e-4}},
  {'name': 'c4', 'bounds': {'lower': 1e-4}},
  {'name': 'c5', 'bounds': {'lower': 1e-4}},
  {'name': 'c6', 'bounds': {'lower': 1e-4}},
  {'name': 'K_total', 'bounds': {'lower': 1e-4}},
  {'name': 'K_total_next', 'bounds': {'lower': 1e-4}},
  {'name': 'r'},
  {'name': 'w'},
  {'name': 'Y'}]

constants = {'alpha': 0.3, 'eq_scale': 1e-2}
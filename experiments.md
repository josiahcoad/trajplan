
## Hparam Sets

- HP1: n_steps=64, nminibatches=64, gamma=0.90, learning_rate=2e-5, ent_coef=0.01, cliprange=0.4, noptepochs=25, lam=0.99

## Test Weight Sets

- TW1: {'fr': 0.3, 'fl': 5, 'fk': 1,
               'ft': 1, 'fail': 0, 'success': 0, 'step_bonus': 0}

# Experiments

## M1

### Settings
- HP1
- weights = {'fr': 0.3, 'fl': 5, 'fk': 1,
            'ft': 1, 'success': 10, 'fail': -10, 'step_bonus': 10}
- depth, width, move_dist, plan_dist = 3, 3, 3, 3
- max_steps, obstacle_pct = 1, 0.5

### Results
- TW1
- T1
    - max_steps, obstacle_pct = 1, 0.0
    - R: -3.93, S: 1
    - rule: R: -1.11
- T2
    - max_steps, obstacle_pct = 1, 0.5
    - R: -5.3, S: .80
    - rule: R: -6.55
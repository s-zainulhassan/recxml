# RecXML

This repository is a e2ad framework research
implementation.


## Repository structure

```text
src/
  config.py
  data.py
  metrics.py
  losses.py
  feature_fusion.py         
  models/
    generator.py
    critic.py
    student.py
  trainers/
    wcgan.py
    kd.py
  ga/
    config.py
    chromosome.py
    population.py
    operators.py
    fitness.py
    controller.py
  pipeline.py

scripts/
  train_teacher.py
  run_ga_distillation.py
  train_final_student.py
  evaluate.py
  compare_teacher_student.py
```

## Data

```text
X_train.npz
Y_train.npz
X_test.npz
Y_test.npz
```
(Please set paths accordingly).


## Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Train teacher:

```bash
python scripts/train_teacher.py
```

Run GA optimization:

```bash
python scripts/run_ga_distillation.py
```

Train final student after evolutionary optimization:

```bash
python scripts/train_final_student.py
```

=

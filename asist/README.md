# A simple walk-through of the pipeline

### Basic Structure

```
Attention-ASIST/
├─ asist/
│  ├─ data/                     <- data folder that contains json semantic map
│  ├─ graph                     <- semantic graph class
│  ├─ visulizer                 <- visulizer that can plot semantic graph to png
│  └─ routeGenerator            <- generating routes/path from trained model
├─ problems/
│  ├─ op/
│  ├─ pctsp/
│  ├─ tsp/
│  └─ vrp/
│     ├─ problem_pcvrp (cvrpp)  <- cvrpp problem related (engineer)
│     └─ problem_vrp (cvrp)     <- cvrp problem related (medic)
├─ params.yaml                  <- contains training hyperparameters
└─ run.py                       <- run this to start training
```

For detailed original Transformer+REINFORCE model documentation, please visit [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)

The modification we made for the medic model (cvrp) and the engineer model (cvrpp) mainly locates in the `problems/vrp` folder

### Pipeline Diagram

![Pipline Diagram](ASIST-Pipline.png)
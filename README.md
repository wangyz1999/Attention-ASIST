# Attention-ASIST
Attention-learn-to-route model for ASIST Environment

Original attention model repository:
[attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)

I will continue update this code base. Please let me know if anywhere in the instruction or code is unclear.

### Distance Matrix to 2D Coordinate Instruction
1. There is a pickle files that stores transformed 2d coordinates from falcon hard map. See `asist/falcon_hard_new.pkl`
2. To get your own version, see `asist/get_attention_problem.py`
    * specify the json data file
    * The function `distance_matrix_to_coordinate` allows you to get an array of coordinates transformed from the given distance matrix. However, those coordinates will be in very high dimensions.
    * The function `jl_transform` allows you to reduce coordinate dimension will preserve relative distance. `dataset_in` should be the output of `distance_matrix_to_coordinate`. `objective_dim` is the desired dimension after reduction (2 in our case)
    * (OP problem only): need to specify price/reward for each victim and the max_length
    * Save Falcon data to pickle file. We will need to load them when later employing the TSP and OP model.

### TSP Instruction
1. train a new model
    * specify the graph_size param, see `options.py` for a complete explanation of all avaliable parameters

        ```
        > python run.py --problem tsp --graph_size 34 --baseline rollout --run_name 'tsp34_rollout'
        ```
    * trained models are in the "output" folder where specific runs are grouped by graph size
2. Use model
    * see `tsp_asist.ipynb`
    * load the pickle file for 2d coordinates in Falcon as mentioned in section 1 of `Distance Matrix to 2D Coordinate Instruction`. There is a pickle files that stores transformed 2d coordinates from falcon hard map. See `asist/falcon_hard_new.pkl`
    * The generated path is represented as a list of indeces of nodes
    * I will have to later update the program such that it automatically gives a path consistes of node ids. For now, please copy the list of indeces back to the end of `asist/get_attention_problem.py` and get the path with node ids.
    * The path can be visulized with the visulizer

### OP Instruction
1. train a new model
    * specify the graph_size param, see `options.py` for a complete explanation of all avaliable parameters

        ```
        > python run.py --problem op --data_distribution falcon --graph_size 34 --baseline rollout --run_name 'op34_rollout'
        ```
    * trained models are in the "output" folder where specific runs are grouped by graph size
2. Use model
    * see op_asist.ipynb
    * load the pickle file for 2d coordinates in Falcon as mentioned in section 1 of `Distance Matrix to 2D Coordinate Instruction`. There is a pickle files that stores transformed 2d coordinates from falcon hard map. See `asist/falcon_hard_new.pkl`
    * The generated path is represented as a list of indeces of nodes
    * I will have to later update the program such that it automatically gives a path consistes of node ids. For now, please copy the list of indeces back to the end of `asist/get_attention_problem.py` and get the path with node ids.
    * The path can be visulized with the visulizer

### Visualizer
1. The generated path is a list of node ids. An @ symbol is added directly before the id of victims being triaged. This is because sometimes agent may pass by a victim but not intend to save it. 
2. `visualizer.simulate_run` generate frame information for location, score, and time
3. `visualizer.animate_MIP_graph` creates the actual animaiton. You can save it as mp4.

### Training GPU
1. The default training GPU is parallel on all avaliable GPU. To use any specific GPU, first in `run.py` comment out 
    ```Python
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    ```
    then specify GPU at
    ```Python
    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    ```

### Mix-Integer-Programming
1. See `asist/MixedIntegerDisocuntedCluster.py`
2. Open json map data file
3. use `find_path` function to get path consists of list of node ids. Specify initial location and victims you want to ignore (Already saved or died).
4. Path can be visualized using the visualizer

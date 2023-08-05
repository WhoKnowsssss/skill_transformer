# Skill Transformer: A Monolithic Policy for Mobile Manipulation

This is the official PyTorch code implementation for ["Skill Transformer: A Monolithic Policy for Mobile Manipulation"].

## Installation
Refer to [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) repo for installation. 
* Requires Python >= 3.7: `conda create -y -n py37 python=3.7`

### Important Config Options
Important hyperparameters and options can be found in `habitat_baselines/config/rearrange/hab/transformer.yaml`
* `NUM_ENVIRONMENTS`: Num of environments used in evaluation. 
* `CHECKPOINT_INTERVAL`: Frequency to checkpoint the model during training
* `TEST_INTERVAL`: Frequency to run through validation dataset
* `RL.POLICY`: Options for the policy class
  * `ACtION_DIST`: Options to experiment with discrete actions and continuous actions. We found that continuous actions work better in our case. 
  * `train_planner` and `train_control`: Options to only train skill inference module or action inference module respectively.  
* `RL.TRAJECTORY_DATASET`: Options for the training dataset
  * `trajectory_dir`: Specify the dataset location. 
  * `dataset_size`: Specify how many episodes to use. 
  * `files_per_load`: Specify how many files (episodes) to load in each epoch. More files means a longer loading time. 
  * `queue_size`: How many epochs of data to load in advance and store in memory
* `RL.VALIDATION_DATASET`: Options for including a validation dataset to check for overfitting in training. 
* `TRANSFORMER`: Hyperparameters

### Collect dataset with DDPPO expert (Habitat 2.0)
* run `python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/tp_srl_oracle_plan.yaml --run-type eval`
* Add following arguments: 
`TEST_EPISODE_COUNT num_episodes`
`DATASET_SAVE_PATH path_to_your_dataset` 
`TASK_CONFIG.SEED seed TASK_CONFIG.SIMULATOR.SEED seed`

### Train Skill Transformer
* run `python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/transformer.yaml --run-type train`
* Add following arguments: 
`TASK_CONFIG.SEED seed TASK_CONFIG.SIMULATOR.SEED seed`

### Evaluate Skill Transformer
* run `python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/transformer.yaml --run-type eval`
* Add following arguments: 
`TASK_CONFIG.SEED seed TASK_CONFIG.SIMULATOR.SEED seed`

## Code Structure
Structure of the code under `habitat_baselines/transformer_policy`:
* `dataset_utils`: Helper functions to load in dataset. 
* `dataset`: The rolling dataset class that loads in dataset and provides data to the trainer in parallel. 
* `focal_loss`: The focal loss implementation. 
* `action_distribution`: Used when some actions are categorical and some are normally distributed. 
* `transformer_model`: Contains the action inference module and the skill inference module. 
* `transformer_policy`: The policy class that specifies the visual encoder, the agent, and the loss functions. 
* `transformer_trainer`: The trainer class that specifies 

# Citation
```
pending
```

# License
this code is licensed under the CC-BY-NC license, see LICENSE.md for more details
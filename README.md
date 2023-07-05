# Ghanbari_Pulse2023

This repository contains the input files and python scripts used to conduct the simulations presented in the following study: https://doi.org/tobedetermined
If you benefit from this work, please include a reference to above study.

## Finite-Element Simulations
The input files for running the FE simulations used for training and validating the artificial neural networks are provided in `FE_input_files`.

| Folder name            | Content                                       |
| ---------------------- | --------------------------------------------- |
| 0\_const\_0L\_\*        | With constant properties for base function.   |
| 1\_HXTdep\_0L\_\*       | With T-dep. properties for base function.     |
| 2\_G33L4mm\_LH         | Layer heating model of 4mm cuboid.            |
| 2\_G33L8mm\_LH         | Layer heating model of 8mm cuboid.            |
| 3\_GxxL4mm\_8t05mm\_C\* | Small scan areas on corners of 4mm cuboid.    |
| 3\_GxxL4mm\_8t05mm\_M  | Small scan areas on the middle of 4mm cuboid. |
| 3\_GxxL8mm\_15t8mm     | Full layer scanning of 8mm cuboid.            |

For running the simulations on various layers of the cuboids, it is necessary to run the layer heating models, and place a copy of `1_run_LH.odb` in the corresponding folder to provide the initial layer temperature distributions.

## Pulse training
The scripts and directories for training the neural networks are provided in `pulse_training`.

To prepare the temperature histories for the training process, the ODB files must be first placed into the `FE_ODB` folder. Using `abaqus python`, the `_abqprep_odb2csv.py` reads the temperature histories and writes them as CSV files inside `FE_CSV`. Then, it is optional to transform the CSV files into pickle format by running `_prep_csv2pkl.py` using modern python for faster load times at the start of training. Alternatively, `_prep_odb2pkl.py` can be used to combine these two steps using modern python.
(Modern python refers to v3.8+ as opposed to abaqus python which is only used to access the content of ODB files.)

Once the data is ready, `_train_pulse.py` can be used to start the training process. In the beginning of this code, various dictionaries are defined to assign various settings:

| Abbreviation | Name          | Content                                                 |
| ------------ | ------------- | ------------------------------------------------------- |
| STG          | Settings      | General settings related to file names and directories. |
| SMP          | Sampler       | Various options for tuning the sampling strategy.       |
| ARCH         | Architecture  | Parameters related to network architecture.             |
| LSS          | Loss function | Options for defining the loss function.                 |
| OPT          | Optimizer     | Settings related to the optimization process.           |
| RST          | Restart       | Options for resuming from a previous training.          |
| GEOM         | Geomtery      | Settings for training the geometry corrector function.  |

The same script was used to train all presented networks. For switching between base and corrector training, the `GEOM['geom_training']` parameter should be set to `True`. For corrector training, it is necessary to pre-evaluate the differences between the FE and base solution using `_prep_perror.py`.

Each time the training starts, a folder with current date and time is created which contains: an overview.txt files containing the training settings, time.log containing information about each epoch, and pickle files with relevant network & optimizer parameters at the end of each epoch, and the loss history. The copy of the jupyter notebook in each folder can be used to plot the loss history.

The pickle files containing the network parameters for presented results in the study are provided in `NNs` folder. The settings for training each of them is provided in accompanying `*_overview.txt` files.

### Ensemble training
All parameters in setting dictionaries can be varied during an ensemble training for different cases. For doing so, the parameters of interest should be defined as a tuple containing various cases of the same format. For instance: `str` -> (`str1`, `str2`)
When more than one parameter is varied, the code iterates through all possible combinations.

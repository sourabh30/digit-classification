https://www.machinelearningplus.com/deployment/conda-create-environment-and-everything-you-need-to-know-to-manage-conda-virtual-environment/

## Step1: Create Conda environment
conda create --name {env_name} python==3.9
conda create --name digits python==3.9

## Step2: Activate the environment
conda activate {env_name}
conda activate digits

## Step3: Create requirments.txt file
pip install -r requirments.txt

## Step4: Run the python file
python plot_digits_classification.py 


## Step4: 


## Step5: 


## Step6: 

Two places of randomness
1. Create the split
	- Freezing the data, In this code the dataset is frozen because line 67 shuffel=False
2. Data order (Learning is iterative)
3. Model:
	- weight initialization




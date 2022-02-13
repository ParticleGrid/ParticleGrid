# Predicting Molecular Energy Gaps using ParticleGrid

``Practical Relevance:``

 High-throughput screeing of molecular properties can be an integral part of discovering and exciting molecules. Density Functional Theory (DFT) is the canonical tool to accurately predict various molecular properties. However, DFT, is time-consuming and can take save hours for even small molecules. Approximating DFT results with fast and accurate data-driven models is an important step in designing an inverse design workflow. 

## Contents
---
The files in this example included are: 

1. `download_data.py` : Helper file to download the `.xyz` files. ``Warning: The dataset is 2.7 GB compressed, and may take some time to download`` 
3. `requirements.txt` : Required packages to run this example.  
2. `predict_energies.ipynb` : Jupyter notebook to train and evaluate a 3D-CNN for energy predictions

## Instructions
---
To run the example, run the notebook `predict_energies.ipynb`

We suggest spinning up a virtual environment to install the required packages. 

```
source <env_name>/bin/activate
pip install -r requirements.txt
jupyter lab
```

Once up and running, the notebook will be on: 

```
http(s)://<server:port>/<lab-location>/lab/tree/predict_energies.ipynb
```
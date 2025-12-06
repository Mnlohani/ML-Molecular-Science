# DataScience_Molecular_physics
## Data science in Molecular physics problems  

This repo is related to the works/ assignments/ projects in machine learning & data Science in molecular science course in Freie Universität Berlin. In this work, we will show the use of data science & machine learning concepts with scikit-learn in various problems mainly in molecular physics area.
First, we will implement the general concepts such as linear, non-linear regression, and classifiction  in context of specific problems. Then we will develop neural network potentials for Lennard-Jones clusters. After that, we will develop a neural network (NN) for the classi cation of local structural environments in bulk phases


**1. Regression**:
    - Linear regression
        - Diabetes Dataset evaluation  
   
    - Non-linear regression :
        - Appoximation of a function using
            - Support vector machine + linear kernel
            - Support vector machine + Gaussian kernel
            - Neural network  
        
**2. Classification**:

    - Non-linear classification 
        - Solving problem in 2D space
            - Neural network
            - Effects of regularisation
        
**3. Project: Neural network potentials for Lennard-Jones clusters**: We develop a neural network (NN) potential for small Lennard-Jones (LJ) clusters. There will be three steps that will be worked on :

    - Setting up the LJ clusters and creating several datasets
        - Initial cluster setup
        - Optimising cluster setup
        - Creating datasets using Monte Carlo sampling
            - MC sampling and dataset for 3D LJ cluster
            - MC sampling and dataset for 2D LJ cluster
        
    - Optimizing the hyper-parameters and training the NN
        - Network architecture and weight optimization
        - Learning curve with respect to dataset size
        
    - Application, transferability, and limitations
        - Use the fitted NN to perform Monte Carlo (MC) sampling of the 3D clusters
        - Transferability of the NN potential 
            - Low and high temperature data
            - Fitting with mixed datasets or including all datasets
            - Transferability from 3D to 2D
            

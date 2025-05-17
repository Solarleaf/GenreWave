# GenreWave

Music Genre Classifier

# How to Run

Requires Python to run

pip install -r requirements.txt

When running the Jupyter networks, remember to set the Kernel properly.

The project does require fma_metadata and fma_small to be downloaded from the following links:

FMA Metadata ~342MB in Zipped format. 1.36GB unzipped.
https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
FMA Small ~7.2GB Data
https://os.unil.cloud.switch.ch/fma/fma_small.zip

The step_final_classify in notebooks can run standalone as it runs all the prior steps automatically as needed.

All python files or notebooks in the Notebooks folder can run standalone as well.

There are several secondary notebooks in the notebooks_secondary.

For the Yamnet Classifier please do the following:

To run Yamnet Model follow these steps,

- access cscigpu server -> cd /research2/<yourusername>

- create virtual environment using conda:

  • conda create -n genrewave python=3.9 -y

  • conda activate genrewave

  • conda install -c conda-forge tensorflow=2.12 numpy=1.23.5 tensorflow-hub librosa pandas scikit-learn matplotlib joblib

  • If encounter an error with missing tensorflow eg, “ModuleNotFoundError: No module named 'tensorflow’” run this command,

        - conda install -c conda-forge tensorflow=2.12

# data folder downloads:

FMA Metadata ~342MB in Zipped format. 1.36GB unzipped.
https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
FMA Small ~7.2GB Data
https://os.unil.cloud.switch.ch/fma/fma_small.zip

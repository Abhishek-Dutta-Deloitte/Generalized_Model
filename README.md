# :Step1: Create the virtual enviornment
'''
conda create -p heartClassificationVenv python==3.9
tconda activate heartClassificationVenv
'''
# :Step2: create setup.py and run it
python setup.py install

# :Step3: Incase of kernel error
conda install -p heartClassificationVenv ipykernel --update-deps --force-reinstall

# :Step4: Create utility files, logger and exception handlers


'''
The whole process can be automated with UI Integration by uploading CSV files
'''


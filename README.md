# Chesboard_size

Done
- Detect lines with 2 Hough functions 
- Parameter can be changed on a window 
- Detect intersection of the lines 
- periodic noise filter


TO DO : 
- Add salt and pepper noise filter 
- Find good parameters depending on the image

## Installation

Install `virtualenv`

    pip install virtualenv

Create enviroment called `venv`

    python -m venv venv

Activate the enviroment:

    # Windows
    ./venv/Scripts/activate

    # Linux
    source ./venv/bin/activate

Install python libs

    pip install -r requirements.txt

### Ubuntu WSL

Install OpenCV dependencies:

    apt-get update && apt-get install -y python3-opencv

### Windows Powershell

Set powershell modes for python venv execution:     
    
    # forbid .ps1 execution 
    Set-ExecutionPolicy Restricted -force 
    
    # allow .ps1 execution 
    Set-ExecutionPolicy Unrestricted -force 


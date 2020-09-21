# Face Attendance
## With supported supervised learning models

### Important Note For Windows system

dlib is made for unix-system. In a system test tests, the performance of this tool in Windows 10 was about a quarter in comparison with Ubuntu built with the same specs. But I haven't seen any difference between these two in other subjects.

You wil need to install numpy+mkl version of numpy to make the system work. Fortunately, anaconda distribution for windows came with numpy+mkl from Anaconda versions 2.5. If you did not use the anaconda distribution, you can grab the mkl version of numpy [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Remember to select the correct version of python.

### Important Note For Unix system

DISCLAIMER: Have not tried for MacOS yet. Also, I had only tested on LTS version only.

opencv2 library for python seems easier to install on windows, and supported most functions for image showing when using the conda packages.
However, you would find out that running on Unix is much faster and smoother than running on windows, especially if you choose to run without CUDA-supported on both systems. if you have not tried it out, I recommended try to run the program on Unix first


### Important Note For All System

You will need at least, Cmake (should be available on Ubuntu LTS and MacOS version already) and dlib.


### Pre-install Phrase

1. For Unix systems:

    a.Pre-reqs:
	
	- python 3 installed

	- MacOS: Xcode and homebrew installed
	
	- Linux: [List of required packages](https://github.com/ageitgey/face_recognition/blob/master/Dockerfile#L6-L34)

    b.Steps run the following commands one by one:
    ```
    git clone https://github.com/davisking/dlib.git
    cd dlib
    mkdir build; cd build; cmake ..; cmake --build .
    cd ..
    python3 setup.py install
    ```

2. For Windows systems:

    a.Pre-reqs:

	- Microsoft Visual Studio 2015 (or newer) with C/C++ Compiler installed

	- Python 3

	- CMake for windows and add it to your system environment variables.

	- (ONLY FOR older versions of dlib) Boost library version 1.63 or newer

    b.Steps:

	- Install scipy and numpy+mkl packages

	- Download Boost
	
	- If you downloaded the binary version skip to step 4 else follow these steps to compile and build Boost by yourself:

		- Extract the Boost source files into C:\local\boost_1_XX_X (X means the current version of Boost you have)

		- Create a system variable with these parameters:

		Name: VS140COMNTOOLS

		Value: C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\ (or any path where you have installed MSVC)

		- Open Developer Command Prompt for Visual Studio and go to the current directory of Boost extracted and try these commands to compile Boost:

            ```
            bootstrap
            b2 -a --with-python address-model=64 toolset=msvc runtime-link=static
            ```

	- (If you have already compiled Boost skip this step) If you already download the binary release just extract the contents to C:\local\boost_1_XX_X

	- Grab latest version of dlib from this repo and extract it.

	- Go to dlib directory and open cmd and follow these commands to build dlib: (remember to replace XX with the current version of Boost you have):

            ```
            set BOOST_ROOT=C:\local\boost_X_XX_X
            set BOOST_LIBRARYDIR=C:\local\boost_X_XX_X\stage\lib
            python setup.py install --yes USE_AVX_INSTRUCTIONS 
            <or> 
            python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
            ```
	
	- Now simply install face_recognition with pip install face_recognition

### Install python packages

All packages needed to run had been added to the requirements.txt file. All you have to do is type this following command:
```
pip install -r requirements.txt
```
Note that in order to install psycopg2 on ubuntu (and linux system in general), you will need a package on debian called libpq-dev, which in turn need a package called libpq5.
To install them, simply do this:
```
sudo apt-get update
sudo apt install libpq5
sudo apt install libpq-dev
```

Or use can use whatever package manager your system currently had.

**IMPORTANT NOTE FOR ANACONDA DIST USERS:** 
Currently we choose opencv library for image and video processing, however opencv is regrettably one of the few higher visibility packages that not natively supported by conda. Since that is the case, we will use pip to install opencv for this program. The package will be included into the requirements.txt file so you do not need to consider anything else if you have not use opencv on python before. However, if you have install it using conda channel, you will need to remove it first

```
conda remove <-n name_of_the_virtualenv> opencv
```

### How to use

1. Steps:
	- Get a lot of pictures of all the people in your insitution. Store them at the dataset folder first with the structure like:
	```
        dataset/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
    ```

	- Since what we need to run the system is a vector represented each pictures + the label for it, after run the load method of the LoadTrainset class (this must be done at least once), you can choose to save the dataframe to either a database (providing correct connection) or a excel file (providing correct filepath). You can load back from either source to save time.

	- With the LoadTrainset class already loaded, you have the array of vectors at LoadTrainset.encodings, while the label at  LoadTrainset.names

	- Declare the model with Trainer class with the array of vectors and the label above

	- Train it and either save out to a binary file to save time or get the already train model ready to use in the script.

	- Set known\_face\_encodings and known\_face\_name with the array of vectors and the label above.

	- Run face\_recognize method of the CheckIn class. You can choose whether or not to use a model to get the name.

2. How to stop:

    Press 'q' no the keyboard to exit system and release the camera. On the rare occasion of unexpected stop, the camera process will be left unattended. In that case, find the python process and kill it.

3. Note:

    - Only print out the name of the person in the console, does not mark the face in live feed
    - Return anyone that does not pass the double check as unknown, but will not print out anything 
    - Only has a very small sample size, thus the parameters for model is mostly the default parameters
    - Also with small sample size, the model may gives wrong prediction frequently
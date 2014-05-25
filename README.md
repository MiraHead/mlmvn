mlmvn
=====

Machine learning tool - simulator of multi-layered neural network (like MLP) with multi-valued neurons.

INSTALATION:

Installation and first run on Linux (tested on Ubuntu 12.04) systems 
using Advanced Packaging Tool (APT):
    1: Install python (in case that python is not pre-installed 
          on your system). So far only Python 2 is supported because of numpy
          and matlplotlib libraries.
          command: sudo apt-get install python
    2: Install Numpy.
          command: sudo apt-get install python-numpy
    3: Install Matplotlib. 
          command: sudo apt-get install python-matplotlib
          (PyMLMVN should work even without matplotlib, although graphs
           would not be created and some errors may occur in console)
    4: Download PyMLMVN from this repository to your computer:
    5: Run python script "pymlmvn.py" in PyMLMVN's repository.
          command: python pymlmvn.py
    6: For initial familiarization with GUI try to open iris dataset located
          in "test_data" folder in PyMLMVN's repository. 
          "iris.arff" contains human-readable original data
          "iris.mvnd" conatins data already transformed to complex domain

Although simulator is primarily designated for use on Linux, 
can be run on Windows (tested on Windows 7).
    1: Install Python 2.7 from official website: 
           www.python.org/download/releases
    2: Install Numpy. E.g. from CH.~Gohlke's unofficial binaries for windows:
           http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
    3: Install gi.repository modules via \term{PyGObject Win32} 
            which can be obtained at SourceForge:
            http://sourceforge.net/projects/pygobjectwin32/files/
            During installation process packages \term{GDK}, \term{GTK+3.x.x} 
            and \term{Pango} need to be selected.
    4: Download PyMLMVN from this repository to your computer:
    5: Install Matplotlib from official website:
         http://matplotlib.org/downloads.html
         (PyMLMVN should work even without Matplotlib, although graphs
          would not be created and some errors may occur in IDLE python's GUI)
    6: Run python IDLE gui and run commands:
        import os   # loads module with operating system functionality
        os.chdir("Path to PyMLMVN")   # changes working directory
        import pymlmvn  # prepares PyMLMVN
        pymlmvn.main()  # starts PyMLMVN simulator
    7: For initial familiarization with GUI try to open iris dataset located
          in "test_data" folder in PyMLMVN's repository. 
          "iris.arff" contains human-readable original data
          "iris.mvnd" conatins data already transformed to complex domain

NOTES:
PyMLMVN uses matplotlib's backend for GTK3+ (for graph plottting) which is not 
yet part of standard matplotlib distribution. Some functions of graph plotting 
(esp. saving to some graphical formats) may not work properly.

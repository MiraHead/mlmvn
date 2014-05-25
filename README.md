<h1>PyMLMVN</h1>
=====<br>

<b>Machine learning tool - simulator of multi-layered neural network (like MLP) with multi-valued neurons.</b>

<h2>INSTALATION:</h3>

<h3>LINUX</h3>

Installation and first run on Linux (tested on Ubuntu 12.04) systems using Advanced Packaging Tool (APT):

<ol>
    <li>Install python (in case that python is not pre-installed on your system).
          So far only Python 2 is supported because of
          numpy and matlplotlib libraries.<br>
        command: <i>sudo apt-get install python</i></li>
    <li>Install Numpy.<br>
        command: <i>sudo apt-get install python-numpy</i></li>
    <li>Install Matplotlib. <br>
        command: <i>sudo apt-get install python-matplotlib</i><br>
        (PyMLMVN should work even without matplotlib, although graphs
            would not be created and some errors may occur in console).</li>
    <li>Download PyMLMVN from github repository to your computer:<br>
        <a href="https://github.com/MiraHead/mlmvn">https://github.com/MiraHead/mlmvn</a></li>
    <li>Run python script "pymlmvn.py" in PyMLMVN's repository.<br>
            command: <i>python pymlmvn.py</i></li>
    <li>For initial familiarization with GUI try to open iris dataset located
          in "test_data" folder in PyMLMVN's repository. <br>
          "iris.arff" contains human-readable original data whereas<br>
          "iris.mvnd" conatins data already transformed to complex domain.</li>
</ol>

<h3>WINDOWS</h3>
Although simulator is primarily designated for use on Linux, can be run on Windows (tested on Windows 7).
<ol>
    <li>Install Python 2.7 from official website: <br> 
        <a href="www.python.org/download/releases">www.python.org/download/releases</a></li>
    <li>Install Numpy. E.g. from CH.~Gohlke's unofficial binaries for windows: <br>
    <a href="http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy">http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy</a></li>
    <li>Install gi.repository modules via <i>PyGObject Win32</i> which can be obtained at SourceForge: <br>
            <a href="http://sourceforge.net/projects/pygobjectwin32/files/">http://sourceforge.net/projects/pygobjectwin32/files/</a><br>
            During installation process packages <i>GDK</i>, <i>GTK+3.x.x</i> and <i>Pango</i> need to be selected.
        </li>
    <li>Download PyMLMVN from github repository to your computer:<br>
        <a href="https://github.com/MiraHead/mlmvn">https://github.com/MiraHead/mlmvn</a></li>
    <li>Install Matplotlib from official website:<br>
            <a href="http://matplotlib.org/downloads.html">http://matplotlib.org/downloads.html</a><br>
        (PyMLMVN should work even without Matplotlib, although graphs
        would not be created and some errors may occur in <i>IDLE</i> python's GUI).</li>
    <li>Run python <i>IDLE</i> python gui and run commands:<br>
            <i>import os</i>  \# loads module with operating system functionality<br>
            <i>os.chdir("Path to PyMLMVN")</i> \# changes working directory<br>
            <i>import pymlmvn</i> \# prepares PyMLMVN<br>
            <i>pymlmvn.main()</i> \# starts PyMLMVN simulator<br>
        </li>
    <li>For initial familiarization with GUI try to open iris dataset located
          in "test_data" folder in PyMLMVN's repository. <br>
          "iris.arff" contains human-readable original data whereas<br>
          "iris.mvnd" conatins data already transformed to complex domain.</li>
</ol>

<h2>NOTES:</h2>
PyMLMVN uses matplotlib's backend for GTK3+ (for graph plottting) which is not 
yet part of standard matplotlib distribution. Some functions of graph plotting 
(esp. saving to some graphical formats) may not work properly.


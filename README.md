# Epistemology of Deep Learning
**Course Information**: TBA

**Instructors**: [Anubav Vasudevan](mailto:anubav@uchicago.com?subject=Epistemology%20of%20Deep%20Learning) and [Malte Willer](mailto:willer@uchicago.com?subject=Epistemology%20of%20Deep%20Learning)


## Course Description

Philosophers have long drawn inspiration for their views about the nature of human cognition, the structure of language, and the foundations of knowledge, from developments in the field of artificial intelligence. In recent years, the study of artificial intelligence has undergone a remarkable resurgence, in large part owing to the invention of so-called ''deep'' neural networks, which attempt to instantiate models of cognitive neurological development in a computational setting. Deep neural networks have been successfully deployed to perform a wide variety of machine learning tasks, including image recognition, natural language processing, financial fraud detection, social network filtering, drug discovery, and cancer diagnoses, to name just a few. While, at present, the ethical implications of these new and powerful systems are a topic of much philosophical scrutiny, the epistemological significance of deep learning has garnered significantly less attention.

In this course, we will undertake a critical examination of some of the bold epistemological claims that have been made on behalf of deep neural networks. To what extent can deep learning be represented within the framework of existing theories of statistical and causal inference, and to what extent does it represent a new epistemological paradigm? Are deep neural networks genuinely theory-neutral, as it is sometimes claimed, or does the underlying architecture of these systems encode substantive theoretical assumptions and biases? Without the aid of a background theory or statistical model, how can we, the users of a deep neural network, be in a position to trust the reliability of its predictions? In principle, are there any cognitive tasks with respect to which deep neural networks are incapable of outperforming human expertise? Do recent developments in artificial intelligence shed any new light on traditional philosophical questions about the capacity of machines to act intelligently, or the computational and mechanistic bases of human cognition? 

## Instructions for Accessing the Course Files

### Manual Download and Local Installation of Dependencies

The following instructions assume that (1) you are using a Linux command-line environment to interface with your computer (on macOS you can access a command line via the [terminal app](https://support.apple.com/guide/terminal/welcome/mac). On windows, you can either install the [Window Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or use [Anaconda Prompt](https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/)); (2) you have installed [git](https://git-scm.com/), [conda](https://docs.conda.io/en/latest/), and [Python 3.6+](https://www.python.org/) and have command-line access to the relevant commands; and (3) you have installed [Jupyter-lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 

1. Clone the git repository into a directory of your choosing:
```sh
cd [repo-directory]
git clone https://github.com/anubav/epistemology_of_deep_learning.git
cd epistemology_of_deep_learning
```

2. Create and activate the virtual environment:
```sh
conda env create --prefix ./envs -f environment.yml
conda activate ./envs
```

3. Move the startup scripts into the appropriate folder:

```sh
cp -a src/startup/. ~/.ipython/profile-default/startup/
```

(if you wish to avoid having to do this each time you pull the repository, you may want to replace the folder `~/.ipython/profile-default/startup` with a symbolic link to the folder `[repo-directory]/src/startup`)

3. Start a jupyter-lab server in the directory containing the course files:
```sh
jupyter-lab
```

4. If the jupyter-lab server starts correctly you should be presented with the following sort of message:
```
    To access the notebook, open this file in a browser:
        file:///[home-directory]/.local/share/jupyter/runtime/nbserver-1765-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=2892b4690080becce06c8858ba970d4dbc99fdf04756f472
     or http://127.0.0.1:8888/?token=2892b4690080becce06c8858ba970d4dbc99fdf04756f472
```
Open a web browser and browse to "localhost:8888" to access the files on the server. The notebook files containing the course notes are located in the directory "notebooks".  

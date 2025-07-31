# 🧪 Getting Started with IMC Data Analysis Using Python

Welcome! 🎉 If you've found your way here, you're probably starting your journey into analyzing **spatial -omics** data using Python. This guide focuses on **Imaging Mass Cytometry (IMC)** data, but the tools will be expanded for other data types over time.

Whether you're a biologist, analyst, or student—with or without coding experience—this step-by-step guide will help you install Python, set up your environment, and run your first analysis using the tools in this GitHub repository.

---

## 🐍 Step 1: Install Python Using Anaconda

To run Python and manage analysis tools, we recommend downloading **[Anaconda](https://www.anaconda.com/)**—a user-friendly Python distribution.

### *💡 "What Is Anaconda?"*
Anaconda is a free and open-source platform that makes it easy to:
- Run Python code
- Manage software libraries and environments
- Use scientific and data analysis tools

Anaconda includes:
- Python 🐍
- Jupyter Notebooks 📓
- Conda (a package and environment manager)
- Hundreds of pre-installed packages for data science

👉 **Download the version appropriate for your operating system** (Windows, macOS, or Linux) from the [official Anaconda website](https://www.anaconda.com/products/distribution).

Once you've installed Anaconda, several of the next steps require us to interact with the computer and enter commands via a **command line**.

### *💻 "What Is the Command Line?"*

The **command line** (also called the *terminal*, *shell*, or *Anaconda Prompt* on Windows) is a text-based interface where you type commands for your computer to execute. Instead of clicking buttons in a graphical interface, you write commands like `cd` (change directory) or `conda install` to navigate and run programs. It might look a little intimidating at first, but it’s just a way to “talk” directly to your computer. For this guide, you’ll mostly copy and paste a few simple commands, but if you get stuck then there is plenty of information on Google. 

---

## 📦 Step 2: Set Up a Conda Environment

Once Anaconda is installed, you’ll use its built-in tool, **Conda**, to create an **environment** specifically for IMC data analysis with all the **packages** we need. We will then install the **package** stored in this **GitHub repository** that has specific tools for analysing  IMC data.

### *🔧 "What Is Conda?"*
**Conda** is a powerful tool included with Anaconda that helps you manage your Python setup. It allows you to install and update packages (libraries of reusable code), manage different versions of both Python and its packages, and create isolated environments for individual projects. This keeps your work organized and avoids conflicts between dependencies. Without Conda, installing scientific packages like NumPy, Pandas, or Scanpy—especially when they require specific versions—can be much more complicated.

### *📦 "What Are Environments"*?
A **Conda environment** is like a dedicated workspace on your computer for a specific project. Each environment can have its own version of Python and its own set of packages, completely isolated from other projects. This means you can work on multiple projects with different dependencies without them interfering with each other. By using environments, you keep your base Python clean and avoid installing unnecessary or conflicting packages. Switching between environments is simple, making it easy to manage different setups as needed.

### *📦 "What Are Packages?"*
**Packages** are collections of reusable code written by others. There are Python packages to do pretty much anything you can think of, but we will install specific ones that let us read and process images, analyze data, and create plots and charts. Think of packages like apps you install on your phone—they add functionality without needing to build things from scratch.


### *🐙 "What Is a GitHub repository?"*
**GitHub** is an online platform for sharing code and collaborating on projects—like a social network for programmers and researchers. It allows you to share code publicly or privately, track changes over time, and contribute to open-source software using version control with Git. Projects on GitHub are stored in **repositories** (or "repos"), which are just collections of files and folders for a particular project. Cloning a repository means making a local copy on your computer so you can use or edit it. If you’re not familiar with Git, you can also download the repo as a ZIP file. Cloning also makes it easy to update your local copy later using a simple command like `git pull` to fetch the latest changes from GitHub.


### 🧪 Setting Up the Environment

1. Open the **Anaconda Prompt** (Windows) or Terminal (macOS/Linux). This is a **command line* interface where you can input text commmands.
2. Navigate to where you would like to download this repo to. Example:
   <br>`cd path/to/your/folder`
3. **Clone** this GitHub repository and change directory into it
   <br>`git clone https://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@github.com/dr-michael-haley/imcanalysis.git`
   <br>`cd imcanalysis`
4. Create the environment using the provided YAML file. This file is essentially just a list of all the packages we need to install:
   <br>`conda env create -f Conda_environments/conda_environment.yml`
5. Activate the environment:
   <br>`conda activate spatbiotools`
6. Install the analysis package in "editable" mode. This will mean if I update the code, you can just redownload the latest version using a `git pull` command ran in this directory.
   <br>`pip install -e .`
7. Install **Jupyter Notebooks**:
   <br>`conda install jupyter`
8. Create a folder on your computer where you would like to save your analyses, and copy the *Tutorials* folder from where you cloned (i.e. downloaded locally) the repo into this folder. 
9. Navigate to the folder where you copied the *Tutorials*. Example:
   <br>`cd path/to/programming/folder`
10. Start Jupyter in this folder. By default, Jupyter can only see files in the folder it was ran in.
   <br>`jupyter lab`<br>
   This will open a browser window where you can access and run the notebooks. If a brower doesn't pop up, then you may need to follow the instructions in the Anaconda Prompt, which usually requires you to copy and paste a URL into your browser, which will then open Jupyter.

---

## 📓 Step 3: Explore the Jupyter Notebooks

You are now ready to start working with Jupyter Notebooks. These covers various stages of analysig IMC data. To get more information about the different notebooks, go to the [Tutorials](Tutorials/) folder on this GitHub and consult the `README`.

Once your environment is set up and you have ran Jupyter Lab, you can begin using the **Jupyter Notebooks** in the `Tutorials` folder that you copied into your programing folder.

### *🖥 "What Are Jupyter Notebooks?"*
Jupyter Notebooks are interactive documents that let you:
- Run Python code 🧑‍💻
- Display results instantly (e.g., tables, plots)
- Add notes and comments to explain your steps
- Mix code and documentation in one place

📚 These notebooks:
- Guide you through each stage of IMC data analysis
- Include explanations and comments
- Are beginner-friendly—no heavy coding experience needed

### Restarting Jupyter

1. Open the **Anaconda Prompt** (Windows) or Terminal (macOS/Linux).
2. Navigate to where you previously copied the `Tutorials` folder, and are planning on storing your analyses. Example:
   <br>`cd path/to/programming/folder`
3. Activate the environment:
   <br>`conda activate spatbiotools`
9. Start Jupyter as before.
   <br>`jupyter lab`<br>
   This will open a browser window where you can access and run the notebooks.

---

## 📓 Tips for Using Jupyter Notebooks

Jupyter Notebooks are a great way to run Python code step by step in your browser. If you're new to them, here are some best practices and common pitfalls.


### ✅ Best Practices

- **🔢 Run cells in order (unless you know what you are doing!):**  
  Notebooks run code in "cells." Always run cells from top to bottom in order. Skipping around can sometimes be useful for testing purpposes, but will cause unexpected errors if earlier variables or imports haven't been run yet, and can lead to very confusing results that are hard to interpret. Jupyter will keep track of the order that cells are ran by numbering them inside the square brackets, e.g. `[1]` indicates the first cell that was ran, followed by `[2]`, etc. Empty square brackets `[ ]` indicates cells not yet ran, amd cells currently running are indicated with an asterisks `[*]`

- **💾 Save your work often:**  
  By default, Jupyter will automatically save your work, but still use `File > Save and Checkpoint` or click the 💾 icon frequently. This saves your notebook and also creates a restore point. 

- **📌 Use Markdown cells for notes:**  
  You can change a cell type to "Markdown" and write plain text, formatted notes, or explanations using simple syntax (like `**bold**` or `# headers`). Think of it an electronic lab book for your computational work, and keep notes as thoroughly as you would a wet lab experiment.

- **🔍 Use small cells:**  
  Break your code into smaller, manageable pieces. This helps with troubleshooting and understanding your workflow later.

- **📤 Restart your kernel occasionally:**  
  Use `Kernel > Restart & Run All` to reset the notebook’s memory and rerun everything from scratch. This is a great way to ensure your code works end-to-end. However, it will also reset any analyses you have done, leaving only the results of analyses. 


### *🧠 "What Is the Kernel in Jupyter Notebooks?"*

In Jupyter Notebooks, the **kernel** is the computational engine that runs your Python code. Think of it as the "brain" behind the notebook—when you type code into a cell and press `Shift + Enter` (run the cell), the kernel executes that code and returns the result.

Here’s what you should know about kernels:

- 🟢 **Each notebook has its own kernel:** If you open multiple notebooks, each one runs in its own separate kernel (but they can share the same environment).
- 🔄 **Restarting the kernel clears memory:** This removes all variables, functions, and imports you've defined. It’s like starting fresh.
- 💣 **Crashes happen:** If your code causes an error or uses too much memory, the kernel might crash. You’ll need to restart it to continue.
- 💾 **Kernel state matters:** The order in which you run cells affects what the kernel knows. If you run a cell that uses a variable before it's defined, the kernel will give you an error.

You can control the kernel from the menu bar:
- `Kernel > Restart` – Clears everything and keeps the notebook open
- `Kernel > Restart & Run All` – Clears memory and runs the notebook from top to bottom
- `Kernel > Interrupt` – Stops a long-running or stuck cell

Understanding how the kernel works will help you troubleshoot and run your notebooks more effectively.

### ⚠️ Common Pitfalls

- **🚫 Running cells out of order:**  
  Jupyter remembers the *order* in which code is run, not the order it's written. Running a cell that depends on a previous one *before* that one can cause errors like `NameError` or missing variables.

- **📉 Forgetting to activate the correct environment:**  
  If you installed packages in a conda environment but launch Jupyter from outside it, your packages may not be available. Always activate your environment before starting Jupyter.

- **🧠 Variables persist between cells:**  
  If you define something in one cell, it stays in memory until the kernel is restarted—even if you delete the cell. This can lead to confusion during debugging. Use `Kernel > Restart` to clear everything.

- **📂 Not knowing where you are (in the file system):**  
  Notebooks run in the directory where they were opened. If you load a file using a relative path (`./data/myfile.csv`), make sure the file is actually there!

- **🧪 Mixing code and output:**  
  Outputs can stay visible even after you change the code. If things look weird, restart the kernel and re-run the notebook.


### 🎓 Extra Tips and Shortcuts

- You can run a cell by pressing **Shift + Enter**.
- Press **Esc** then **M** to convert a code cell to Markdown.
- Use `Tab` for auto-complete and suggestions.
- Use `Shift + Tab` inside parentheses to see function hints.
- Use the **"Help"** menu to explore Python and Jupyter tutorials.
- Selecting a cell and pressing **A** will create a new cell above, and **B** will create one below.
- Selecting a cell and pressing **D twice** will delete the cell.
- You can select multiple cells by shift clicking.
- You can rearrange the order of cells by drag and dropping them.
---



# 🧰 Other things on this GitHub

## 🧼 Preprocessing Scripts for University of Manchester CSF3

Scripts for preprocessing IMC data on CSF3 (a command-line cloud computing platform used at the University of Manchester) are available in the `CSF3` folder. These scripts are designed to work with SLURM-based systems or similar HPC environments.

---

## 🌟 IMC_Denoise (Updated November 2024)

This notebook has largerly been superceded by CSF3-compatible denoising scripts. The original notebook implementation of [IMC Denoise](https://github.com/PENGLU-WashU/IMC_Denoise/)—designed to integrate with the Bodenmiller pipeline—is still available in the repo, but we now recommend running denoising on CSF3 for best results.

---

## 🧬 REDSEA (Cell Segmentation Overlap Correction)

This tool is adapted from the original [REDSEA implementation](https://github.com/labsyspharm/redseapy) by Artem Sokolov, reworked for better integration with the Bodenmiller pipeline. Our testing indicates that it doesn't work particularly well, often losing so much positive signal that the signal to noise ratio is barely improved. Still, I've left it here just in case.

> [!CAUTION]  
> This code has not been thoroughly tested or maintained recently. Use with caution.  
> **Last update:** November 4th, 2022.

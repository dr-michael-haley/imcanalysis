# 🧪 Getting Started with IMC Data Analysis Using Python

Welcome! 🎉 If you've found your way here, you're probably starting your journey into analyzing **spatial -omics** data using Python. This guide focuses on **Imaging Mass Cytometry (IMC)** data, but the tools will be expanded for other data types over time.

Whether you're a biologist, analyst, or student—with or without coding experience—this step-by-step guide will help you install Python, set up your environment, and run your first analysis using the tools in this GitHub repository.

---

> [!TIP]
> **What is Anaconda?**
> Anaconda is a free and open-source platform that makes it easy to:
- Run Python code
- Manage software libraries and environments
- Use scientific and data analysis tools

## 🐍 Step 1: Install Python Using Anaconda

To run Python and manage analysis tools, we recommend downloading **[Anaconda](https://www.anaconda.com/)**—a user-friendly Python distribution.

### 💡 What Is Anaconda?
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

Once you've installed Anaconda, several of the next steps require us to interact with the computer and enter commands via a **command line**

### 💻 What Is the Command Line?

The **command line** (also called the *terminal*, *shell*, or *Anaconda Prompt* on Windows) is a text-based interface where you type commands for your computer to execute. Instead of clicking buttons in a graphical interface, you write commands like `cd` (change directory) or `conda install` to navigate and run programs. It might look a little intimidating at first, but it’s just a way to “talk” directly to your computer. For this guide, you’ll mostly copy and paste a few simple commands.

---

## 📦 Step 2: Set Up a Conda Environment

Once Anaconda is installed, you’ll use its built-in tool, **Conda**, to create a workspace specifically for IMC data analysis.

### 🔧 What Is Conda?
Conda helps you:
- Create **isolated environments** for different projects
- Install only the libraries (called **packages**) you need
- Avoid breaking things when updating tools

### 📦 What Are Packages?
Packages are collections of reusable code written by others. They allow you to:
- Read and process images
- Analyze data
- Create plots and charts

🛠 Think of packages like apps you install on your phone—they add functionality without needing to build things from scratch.

### 🧪 Setting Up the Environment

1. **Clone or download** this GitHub repository (use the green **<> Code** button).
2. Open the **Anaconda Prompt** (Windows) or Terminal (macOS/Linux). This is a **command line* interface where you can input text commmands. 
3. Navigate to the folder where you downloaded the repo. Example:
   ```bash
   cd path/to/your/folder
   ```
4. Create the environment using the provided YAML file:
   ```bash
   conda env create -f conda_environment.yml
   ```
5. Activate the environment:
   ```bash
   conda activate spatialbio
   ```
6. Install the analysis package in "editable" mode:
   ```bash
   pip install -e .
   ```
   🔁 This allows you to edit the code and have changes apply immediately. Omit `-e` for a regular install.

7. (Optional) If not already installed, add **Jupyter Notebooks**:
   ```bash
   conda install jupyter
   ```

8. Start Jupyter:
   ```bash
   jupyter lab
   ```
   This will open a browser window where you can access and run the notebooks.

---

## 📓 Step 3: Explore the Jupyter Notebooks

Once your environment is set up, you can begin using the **Jupyter Notebooks** in the `Tutorials` folder.

### 🖥 What Are Jupyter Notebooks?
Jupyter Notebooks are interactive documents that let you:
- Run Python code 🧑‍💻
- Display results instantly (e.g., tables, plots)
- Add notes and comments to explain your steps
- Mix code and documentation in one place

📚 These notebooks:
- Guide you through each stage of IMC data analysis
- Include explanations and comments
- Are beginner-friendly—no heavy coding experience needed

---

## 🧰 Extra Tools and Features

### 🧼 Preprocessing Scripts for University of Manchester CSF3

Scripts for preprocessing IMC data on CSF3 (a command-line cloud computing platform used at the University of Manchester) are available in the `CSF3` folder. These scripts are designed to work with SLURM-based systems or similar HPC environments.

---

## 🌟 IMC_Denoise (Updated November 2024)

> ⚠️ **Note:** This notebook has been replaced by CSF3-compatible denoising scripts.

The original notebook implementation of [IMC Denoise](https://github.com/PENGLU-WashU/IMC_Denoise/)—designed to integrate with the Bodenmiller pipeline—is still available in the repo, but we now recommend running denoising on CSF3 for best results.

---

## 🧬 REDSEA (Cell Segmentation Overlap Correction)

This tool is adapted from the original [REDSEA implementation](https://github.com/labsyspharm/redseapy) by Artem Sokolov, reworked for better integration with the Bodenmiller pipeline.

> [!CAUTION]  
> This code has not been thoroughly tested or maintained recently. Use with caution.  
> **Last update:** November 4th, 2022.

---

## ✅ Summary

| Step | What You’ll Do                              | Tools Used                     |
|------|----------------------------------------------|--------------------------------|
| 1    | Install Python environment                   | Anaconda                       |
| 2    | Create workspace with required packages      | Conda, pip                     |
| 3    | Install analysis tools and Jupyter           | SpatialBiologyToolkit, Jupyter |
| 4    | Run tutorials and explore example workflows  | Jupyter Notebooks              |

---

🚀 You're now ready to begin your spatial -omics analysis journey! If anything doesn’t work or you get stuck, check the repository README or raise an issue.

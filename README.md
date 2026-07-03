# Skin Melanoma Detection Using Graph Neural Networks

## Overview

This repository contains the source code used for the research project **"Skin Melanoma Detection Using Graph Neural Networks on Augmented ISIC 2019 Data."**

The project investigates the use of Graph Neural Networks (GNNs) for melanoma classification by converting dermoscopic skin lesion images into graph representations. Images are segmented into superpixels, transformed into Region Adjacency Graphs (RAGs), and classified using several GNN architectures.

This repository is intended to accompany the research paper and allow reviewers to inspect the implementation, data processing pipeline, model architectures, and experimental results.

---

## Dataset

The dataset used can be found [here](https://drive.google.com/drive/folders/1w_h_ntnSlB3nNm9_XAZNAyVoWBsAskak?usp=sharing)

Add it to the working directory under the name **image_dataset** in order to run the code.

---

## Structure

Under results and saved models, there is a **description.txt** file. This file serves to provide a brief description of the different runs stored within these folders. The numerical suffix on different files indicates association to the same run / experiment.

The majority of the code can be found in the **Workspace** folder. Visualizations created for the paper can be found in the **visualizations** subfolder.

---

## Purpose

This repository serves as an implementation archive for the accompanying research project. It is not intended to be a production system, web application, or end-user tool.

The primary goal is to provide transparency into the methodology, experimental pipeline, and code used during the study.

---

## Citation

If referencing this work, please cite the accompanying paper:

Lozzano, B.; Kumar, A.; Patel, M.; Kumar, A.; Vats, S.; Vatsa, A. Evaluating the Effectiveness of Graph Neural Networks on an Augmented Dataset for Melanoma Skin Cancer Detection. AI Engineering 2026, 2 (1), 7. https://doi.org/10.53941/aieng.2026.100007.

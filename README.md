# TMGN

This project proposes TMGN, a novel framework for Remote Sensing Visual Question Answering (RS VQA) that enhances the identification and comprehension of small objects in complex remote sensing images via contrast enhancement and a mutual guidance mechanism.

## Dataset

This project uses the **Remote Sensing VQA - Low Resolution (RSVQA-LR)** and **High Resolution (RSVQA-HR)** datasets, created by Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia.

* **Homepage:** More information about the dataset can be found on the official website(https://rsvqa.sylvainlobry.com/).
* **Download Links:** The dataset is available in two resolutions:
    * [Low Resolution](https://zenodo.org/api/records/6344334/files-archive)
    * [High Resolution](https://zenodo.org/api/records/6344367/files-archive)

We would like to express our sincere gratitude to Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia for creating and publicly sharing the valuable RSVQA dataset. This project would not have been possible without their significant contribution to the remote sensing research community.

## Install

```bash
pip install -r requirements.txt
pip install -e .

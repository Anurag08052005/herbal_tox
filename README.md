# Herbal-Tox: AI-Based Ayurvedic Safety Predictor

**Herbal-Tox** is a hackathon project designed to bridge the gap between traditional Ayurvedic knowledge and modern Computational Toxicology. 

While Ayurvedic medicines are often perceived as "natural and safe," improper formulations—especially those involving heavy metals (*Bhasmas*) or hepatotoxic phytochemicals—can pose health risks. This tool uses Artificial Intelligence and Cheminformatics to predict these risks instantly.

## 🚀 Key Features

* **Smart Recognition:** Instantly converts common herb names (e.g., "Ashwagandha", "Tulsi") into their active chemical structures (SMILES) using the PubChem API.
* **ADMET Analysis:** Predicts **A**bsorption, **D**istribution, **M**etabolism, **E**xcretion, and **T**oxicity properties.
* **Liver Safety Check:** Calculates Lipophilicity (LogP) and other molecular descriptors to flag potential liver strain.
* **Bhasma Detection:** Automatically scans chemical formulas for heavy metals often used in traditional medicine (Mercury, Lead, Arsenic) and issues safety alerts.
* **2D Visualization:** Generates real-time chemical structure diagrams using RDKit.

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **Streamlit:** For the interactive web interface.
* **RDKit:** For cheminformatics and molecular descriptor calculations.
* **PubChemPy:** For fetching chemical data from the NIH database.

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It relies on computational predictions and should not be used as a substitute for clinical trials or professional medical advice.

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Draw
from rdkit.Chem import rdMolDescriptors
from io import BytesIO
import base64

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Herbal-Tox | Ayurvedic ADMET AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Medical/AI" look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2E8B57;
    }
    .tox-badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .tox-badge-low {
        background-color: #2E8B57;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AYURVEDIC KNOWLEDGE BASE (MOCK DB)
# ==========================================
# In a real app, this would be a CSV or SQL Database
HERB_DATABASE = {
    "Ashwagandha (Withania somnifera)": [
        {"name": "Withaferin A", "smiles": "CC1=C(C(=O)OC1)C2(CCC3C4CCC5=CC(=O)C=CC5(C4C(CC3(C2)O)O)C)C", "role": "Anti-cancer, Adaptogen"},
        {"name": "Withanolide D", "smiles": "CC1=C(C(=O)OC1)C(C)C2CCC3C4(C(CC(C3(C2O)C)O)C5=CC(=O)C=CC5(C4)C)O", "role": "Anti-inflammatory"},
        {"name": "Anaferine", "smiles": "C1CCCC(N1)CC(=O)CC2CCCCN2", "role": "Alkaloid"}
    ],
    "Turmeric (Curcuma longa)": [
        {"name": "Curcumin", "smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O", "role": "Anti-inflammatory, Antioxidant"},
        {"name": "Demethoxycurcumin", "smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC=C(C=C2)O)O", "role": "Minor Curcuminoid"},
        {"name": "Turmerone", "smiles": "CC1=CCC(CC1)C(C)C2=CC(=O)C=C(2)C", "role": "Neuroprotective"}
    ],
    "Tulsi (Ocimum sanctum)": [
        {"name": "Eugenol", "smiles": "COC1=C(C=CC(=C1)CC=C)O", "role": "Analgesic, Antimicrobial"},
        {"name": "Ursolic Acid", "smiles": "CC1CCC2(CCC3(C(C3(CCC2C1C)C)CCC4(C(C4)C(C)C(=O)O)C)C)C", "role": "Anti-cancer"},
        {"name": "Rosmarinic Acid", "smiles": "C1=CC(=C(C=C1CC(C(=O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O", "role": "Antioxidant"}
    ],
    "Neem (Azadirachta indica)": [
        {"name": "Azadirachtin", "smiles": "CC(=O)OC1C2C(C(C3(C2(C(C1(C)O)OC(=O)C=C(C)C)O)OC4C3(C(C(C4O)C(=O)OC)OC(=O)C)C)OC(=O)C)C5(C(C5)C6=COC=C6)O", "role": "Insecticidal, Anti-fungal"},
        {"name": "Nimbin", "smiles": "CC1=C(C(=O)OC2C1(C34C(C2)C(C3C(C=C(C4)C5=COC=C5)OC(=O)C)OC(=O)C)C)C(=O)OC", "role": "Anti-viral"}
    ],
    "Ginger (Zingiber officinale)": [
        {"name": "6-Gingerol", "smiles": "CCCCC(O)CCC(=O)CC1=CC(=C(C=C1)O)OC", "role": "Anti-nausea, Digestive"},
        {"name": "Zingiberene", "smiles": "CC1=CCC(CC1)C(C)C2=CCC(C=C2)C", "role": "Carminative"}
    ],
    "Triphala (Polyherbal)": [
        {"name": "Gallic Acid", "smiles": "C1=C(C=C(C(=C1O)O)O)C(=O)O", "role": "Antioxidant (Amla)"},
        {"name": "Chebulagic Acid", "smiles": "C1C2C(C(C(O2)C(=O)OC3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O)OC(=O)C5=CC(=C(C(=C5)O)O)O", "role": "Hepatoprotective (Haritaki)"},
        {"name": "Ellagic Acid", "smiles": "C1=C2C3=C(C(=C1O)O)OC(=O)C4=CC(=C(C(=C43)OC2=O)O)O", "role": "Anti-proliferative"}
    ]
}

# ==========================================
# 3. BACKEND AI LOGIC (RDKit & Heuristics)
# ==========================================

def get_molecule_image(smiles):
    """Generates a 2D image of the molecule from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(400, 300))
        return img
    return None

def calculate_admet(smiles):
    """
    Calculates real physicochemical properties using RDKit.
    Acts as a 'Heuristic Model' for ADMET prediction.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # 1. Calculate Descriptors
    mw = Descriptors.MolWt(mol)           # Molecular Weight
    logp = Descriptors.MolLogP(mol)       # Lipophilicity (Solubility/Permeability)
    tpsa = Descriptors.TPSA(mol)          # Topological Polar Surface Area (Absorption)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    qed_score = QED.qed(mol)              # Drug-likeness score (0-1)

    # 2. ADMET Rules (Heuristic "AI" Logic)
    
    # Absorption Rule (Lipinski Rule of 5)
    # Poor absorption if: MW > 500, LogP > 5, H-Donors > 5, H-Acceptors > 10
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if h_donors > 5: violations += 1
    if h_acceptors > 10: violations += 1
    
    absorption_status = "High" if violations <= 1 else "Low"

    # Blood Brain Barrier (BBB) Logic
    # Usually crosses BBB if TPSA < 90 and MW < 400
    bbb_penetration = "Likely" if (tpsa < 90 and mw < 400) else "Unlikely"

    # Toxicity Flag (Simplistic simulation)
    # In a real app, this would be: model.predict(fingerprints)
    # Here we use QED as a proxy: Very low drug-likeness often correlates with issues
    toxicity_risk = "Low" if qed_score > 0.4 else "Moderate/High (Check Structure)"

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "H_Donors": h_donors,
        "QED": round(qed_score, 2),
        "Absorption": absorption_status,
        "BBB": bbb_penetration,
        "Toxicity_Risk": toxicity_risk,
        "Violations": violations
    }

# ==========================================
# 4. FRONTEND UI (Streamlit)
# ==========================================

def home_page():
    st.image("https://img.freepik.com/free-vector/science-laboratory-research-concept-illustration_114360-1033.jpg", width=700)
    st.title("🌿 Herbal-Tox: AI for Ayurveda")
    st.markdown("""
    **Predict the Toxicity & Efficacy of Ancient Herbs using Modern AI.**
    
    Ayurveda is powerful, but complex. Herbal-Tox helps researchers:
    * **Deconstruct** herbs into their active phytochemicals.
    * **Predict** ADMET properties (Absorption, Metabolism, Toxicity).
    * **Visualize** chemical structures instantly.
    """)
    
    st.info("👈 Select **'Run Analysis'** from the sidebar to start.")

def analysis_page():
    st.title("🔬 Phytochemical ADMET Analyzer")
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_type = st.radio("Search Mode:", ["Select from Database", "Enter Custom SMILES"])
        
        if search_type == "Select from Database":
            selected_herb = st.selectbox("Choose an Ayurvedic Herb:", list(HERB_DATABASE.keys()))
            compounds = HERB_DATABASE[selected_herb]
            st.success(f"Loaded {len(compounds)} active compounds from **{selected_herb}**")
        else:
            custom_smiles = st.text_input("Enter SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O")
            compounds = [{"name": "Custom Input", "smiles": custom_smiles, "role": "User Defined"}]

    with col2:
        st.markdown("### How it works")
        st.caption("""
        1. **SMILES Parsing:** Converts text to chemical graph.
        2. **RDKit Engine:** Calculates LogP, TPSA, MW.
        3. **Safety Logic:** Applies Lipinski's Rule of 5 to flag toxicity risks.
        """)

    # Initialize storage for download
    results_for_export = []

    if st.button("🚀 Analyze Compounds"):
        st.divider()
        
        for cmp in compounds:
            name = cmp["name"]
            smiles = cmp["smiles"]
            role = cmp.get("role", "N/A")
            
            # Perform Backend Calculation
            admet = calculate_admet(smiles)
            
            if admet:
                # Store data for export
                export_data = admet.copy()
                export_data['Compound Name'] = name
                export_data['Role'] = role
                results_for_export.append(export_data)

                # --- Result Card ---
                with st.container():
                    st.subheader(f"🧪 {name}")
                    st.markdown(f"**Role:** *{role}*")
                    
                    c1, c2, c3 = st.columns([1, 1, 2])
                    
                    with c1:
                        img = get_molecule_image(smiles)
                        st.image(img, caption="2D Structure", use_container_width=True)
                        st.code(smiles, language='text')

                    with c2:
                        st.markdown("#### Physicochemical Profile")
                        st.metric("Molecular Weight", f"{admet['MW']} Da")
                        st.metric("Lipophilicity (LogP)", admet['LogP'])
                        st.metric("Drug-Likeness (QED)", admet['QED'])

                    with c3:
                        st.markdown("#### ⚠️ AI Safety Predictions")
                        
                        # Dynamic Badges
                        tox_color = "red" if "High" in admet['Toxicity_Risk'] else "green"
                        abs_color = "green" if admet['Absorption'] == "High" else "orange"
                        
                        st.markdown(f"**Toxicity Risk:** :{tox_color}[{admet['Toxicity_Risk']}]")
                        st.markdown(f"**Oral Absorption:** :{abs_color}[{admet['Absorption']}]")
                        st.markdown(f"**BBB Penetration:** {admet['BBB']}")
                        
                        st.progress(admet['QED'], text="Drug-Likeness Score")
                        
                        if admet['Violations'] > 0:
                            st.warning(f"⚠️ Lipinski Violations: {admet['Violations']} (Check solubility)")
                        else:
                            st.success("✅ Lipinski Rule Compliant (Good drug-like properties)")
                
                st.divider()

        # --- SAVE FUNCTIONALITY ---
        if results_for_export:
            st.markdown("### 📥 Save Results")
            df = pd.DataFrame(results_for_export)
            
            # Reorder columns for better readability in Excel
            cols = ['Compound Name', 'Role', 'Toxicity_Risk', 'Absorption', 'LogP', 'MW', 'QED', 'Violations']
            # Add any extra columns that might be there
            remaining_cols = [c for c in df.columns if c not in cols]
            df = df[cols + remaining_cols]
            
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📄 Download Report (CSV)",
                data=csv,
                file_name=f"{selected_herb.split(' ')[0]}_Analysis.csv" if search_type == "Select from Database" else "Custom_Analysis.csv",
                mime="text/csv",
            )

# ==========================================
# 5. APP NAVIGATION
# ==========================================
def main():
    st.sidebar.title("Herbal-Tox 🐍")
    page = st.sidebar.radio("Navigate", ["Home", "Run Analysis", "About"])
    
    if page == "Home":
        home_page()
    elif page == "Run Analysis":
        analysis_page()
    elif page == "About":
        st.title("About Project")
        st.write("Built for Hackathon 2025. Uses RDKit for cheminformatics.")

if __name__ == "__main__":
    main()
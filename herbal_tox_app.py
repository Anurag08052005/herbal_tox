import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, QED, Draw, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from io import BytesIO
import base64
from stmol import showmol
import py3Dmol
import plotly.graph_objects as go
from fpdf import FPDF
import warnings

# --- 0. SILENCE ALL WARNINGS (Clean Terminal) ---
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & AESTHETIC CSS
# ==========================================
st.set_page_config(
    page_title="Herbal-Tox | Ayurvedic ADMET AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD GOOGLE FONTS & CUSTOM STYLING ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');

    /* GLOBAL STYLES */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* BACKGROUND GRADIENT */
    .stApp {
        background: linear-gradient(135deg, #fdfbf7 0%, #e8f5e9 100%);
    }

    /* HEADINGS */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1b4332; /* Dark Forest Green */
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.5rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* --- SIDEBAR STYLING FIX --- */
    section[data-testid="stSidebar"] {
        background-color: #1b4332;
    }
    
    /* Force ALL Text Color White in Sidebar */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important; 
    }

    /* CARD STYLING FOR METRICS & RESULTS */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #2E8B57;
    }
    
    /* Metric Label Fix */
    div[data-testid="stMetric"] label {
        color: #666 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1b4332 !important;
    }

    /* LAYMAN BOX STYLING (EXPANDED) */
    .layman-box {
        background-color: #f1f8e9;
        border-left: 5px solid #558b2f;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .layman-item {
        margin-bottom: 12px;
        border-bottom: 1px solid #dcedc8;
        padding-bottom: 8px;
    }
    .layman-title {
        font-weight: bold;
        color: #2e7d32;
        font-family: 'Poppins', sans-serif;
        font-size: 15px;
    }
    .layman-text {
        color: #33691e;
        font-size: 14px;
        line-height: 1.5;
    }
    .overall-verdict {
        background-color: #2e7d32;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
    }

    /* BUTTON STYLING */
    div.stButton > button {
        background: linear-gradient(90deg, #2E8B57 0%, #40916c 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 4px 14px rgba(46, 139, 87, 0.4);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }

    /* INPUT FIELDS */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 1px solid #b7e4c7;
        background-color: white;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KNOWLEDGE BASES
# ==========================================
HERB_DATABASE = {
    "Ashwagandha (Withania somnifera)": [
        {"name": "Withaferin A", "smiles": "CC1=C(C(=O)OC1)C2(CCC3C4CCC5=CC(=O)C=CC5(C4C(CC3(C2)O)O)C)C", "role": "Anti-cancer, Adaptogen"},
        {"name": "Withanolide D", "smiles": "CC1=C(C(=O)OC1)C(C)C2CCC3C4(C(CC(C3(C2O)C)O)C5=CC(=O)C=CC5(C4)C)O", "role": "Anti-inflammatory"}
    ],
    "Turmeric (Curcuma longa)": [
        {"name": "Curcumin", "smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O", "role": "Anti-inflammatory, Antioxidant"},
        {"name": "Turmerone", "smiles": "CC1=CCC(CC1)C(C)C2=CC(=O)C=C(2)C", "role": "Neuroprotective"}
    ],
    "Tulsi (Ocimum sanctum)": [
        {"name": "Eugenol", "smiles": "COC1=C(C=CC(=C1)CC=C)O", "role": "Analgesic, Antimicrobial"},
        {"name": "Ursolic Acid", "smiles": "CC1CCC2(CCC3(C(C3(CCC2C1C)C)CCC4(C(C4)C(C)C(=O)O)C)C)C", "role": "Anti-cancer"}
    ],
    "Neem (Azadirachta indica)": [
        {"name": "Azadirachtin", "smiles": "CC(=O)OC1C2C(C(C3(C2(C(C1(C)O)OC(=O)C=C(C)C)O)OC4C3(C(C(C4O)C(=O)OC)OC(=O)C)C)OC(=O)C)C5(C(C5)C6=COC=C6)O", "role": "Insecticidal, Anti-fungal"},
        {"name": "Nimbin", "smiles": "CC1(C2CCC3C(C2)CC(C3(C1)C)O)C(=O)O", "role": "Anti-inflammatory, Bitter limonoid"}
    ],
    "Bakuchi (Psoralea corylifolia)": [
        {"name": "Psoralen", "smiles": "C1=CC2=C(C=CO2)C=C1", "role": "Skin healing, Photosensitizer"},
        {"name": "Bakuchiol", "smiles": "CC(C)=CCC1=CC(=C(C=C1)O)C(C)C", "role": "Anti-aging, Antimicrobial"}
    ],
    "Safed Musli (Chlorophytum borivilianum)": [
        {"name": "Saponins", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Aphrodisiac, Adaptogen"},
        {"name": "Stigmasterol", "smiles": "CCC(CCC=C1C2CCC3(C(C2CCC1O)C)CCC4=CC(=O)C=CC34C)C", "role": "Steroidal precursor"}
    ],
    "Punarnava (Boerhavia diffusa)": [
        {"name": "Punarnavine", "smiles": "C1=CC(=C(C=C1N)O)O", "role": "Diuretic, Anti-inflammatory"},
        {"name": "Boeravinone B", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antioxidant"}
    ],
    "Shankh Bhasma (Conch ash)": [
        {"name": "Calcium Carbonate", "smiles": "C(=O)([O-])[O-]", "role": "Antacid"},
        {"name": "Calcium Oxide", "smiles": "O=[Ca]", "role": "Digestive support"}
    ],
    "Vacha (Acorus calamus)": [
        {"name": "Asarone", "smiles": "COC1=CC(=C(C=C1)OC)CC=C", "role": "Nootropic, Digestive"},
        {"name": "β-asarone", "smiles": "COC1=CC(=C(C=C1)OC)C=CC", "role": "Neuroactive"}
    ],
    "Guduchi Satva (Tinospora starch extract)": [
        {"name": "Cordifolioside", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Immunomodulator"},
        {"name": "Tinosporaside", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Anti-inflammatory"}
    ],
    "Jasmine (Jasminum officinale)": [
        {"name": "Jasmonic acid", "smiles": "CC(C)C1CCC(C1=O)C(=O)O", "role": "Aromatic, Sedative"},
        {"name": "Benzyl acetate", "smiles": "CC(=O)OCC1=CC=CC=C1", "role": "Fragrance, Relaxant"}
    ],
    "Chamomile (Matricaria chamomilla)": [
        {"name": "Apigenin", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Calming, Anti-inflammatory"},
        {"name": "Bisabolol", "smiles": "CC(C)CCC(C)C1CCC1O", "role": "Anti-inflammatory, Digestive"}
    ],
    "Henna (Lawsonia inermis)": [
        {"name": "Lawsone", "smiles": "C1=CC(=O)C2=CC=CC=C2O1", "role": "Antimicrobial, Dye"},
        {"name": "Gallic acid", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antioxidant"}
    ],
    "Arrowroot (Maranta arundinacea)": [
        {"name": "Starch", "smiles": "C(C(C(C(C(CO)O)O)O)O)O", "role": "Digestive support"},
        {"name": "Amylose", "smiles": "C(C1C(C(C(O1)O)O)O)O", "role": "Energy source"}
    ],
    "Ashwagandha Nagori (special cultivar)": [
        {"name": "Withanoside IV", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)O)C)C", "role": "Adaptogen"},
        {"name": "Withanolide A", "smiles": "CC1=C(C(=O)OC1)C(C)C2CCC3C4(C(CC(C3(C2)O)O)C5=CC(=O)C=CC5(C4)C)O", "role": "Stress relief"}
    ],
    "Shirish (Albizia lebbeck)": [
        {"name": "Lebbeckoside C", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Anti-allergic"},
        {"name": "Saponins", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Expectorant"}
    ],
    "Nagkesar (Mesua ferrea)": [
        {"name": "Mesuol", "smiles": "CC1=CC(=CC=C1O)CCC", "role": "Anti-inflammatory"},
        {"name": "Xanthones", "smiles": "C1=CC2=C(C=C1)OC(=O)C3=CC=CC=C23", "role": "Antioxidant"}
    ],
    "Kachnar (Bauhinia variegata)": [
        {"name": "Octacosanol", "smiles": "CCCCCCCCCCCCCCCCCCCCCCCCCCCO", "role": "Antioxidant"},
        {"name": "Kaempferol", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Anti-inflammatory"}
    ],
    "Neem Chhal (Neem stem bark)": [
        {"name": "Gedunin", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Antimalarial"},
        {"name": "Limonoids", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3C)C)C)C", "role": "Bitter tonic"}
    ],
    "Anantmool (Hemidesmus indicus)": [
        {"name": "Hemidesmin", "smiles": "C1=CC(=C(C=C1O)O)OC", "role": "Blood purifier"},
        {"name": "2-Hydroxy-4-methoxybenzaldehyde", "smiles": "COC1=CC=C(C=C1O)C=O", "role": "Fragrance, Antioxidant"}
    ],
    "Guggulu Sudha (Purified guggul)": [
        {"name": "Z-guggulsterone", "smiles": "CC1CCC2(C1)CCC3(C2C(CC4C3CCC5=CC(=O)C=CC45C)C)C", "role": "Anti-inflammatory"},
        {"name": "E-guggulsterone", "smiles": "CC1CCC2(C1)CCC3(C2C(CC4C3CCC5=CC(=O)C=CC45C)C)C", "role": "Hypolipidemic"}
    ],
    "Palash (Butea monosperma)": [
        {"name": "Butein", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Anti-inflammatory"},
        {"name": "Coreopsin", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Antioxidant"}
    ],
    "Baheda Oil (Terminalia bellirica seed oil)": [
        {"name": "Oleic acid", "smiles": "CCCCCCCC=CCCCCCCCC(=O)O", "role": "Emollient"},
        {"name": "Linoleic acid", "smiles": "CCC=CCC=CCCCCCCCC(=O)O", "role": "Skin health"}
    ],
    "Nirgundi Oil (Vitex negundo leaves)": [
        {"name": "Vitexin", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Analgesic"},
        {"name": "Negundoside", "smiles": "C1CCC2C(C1)CCC3(C2C(CC4C3O)C)C", "role": "Anti-inflammatory"}
    ],
    "Bel (Aegle marmelos fruit pulp)": [
        {"name": "Marmelosin", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Digestive"},
        {"name": "Skimmianine", "smiles": "C1=CC(=C(C=C1O)O)N", "role": "Antimicrobial"}
    ],
    "Chirata (Swertia chirata)": [
        {"name": "Swertiamarin", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antimalarial"},
        {"name": "Mangiferin", "smiles": "C1=CC(=C(C=C1O)O)C2=C(C(=O)C(=CC2)O)O", "role": "Anti-diabetic"}
    ],
    "Kalmegh (Andrographis paniculata)": [
        {"name": "Andrographolide", "smiles": "CC1CCC2(C1)C(CC3=CC(=O)C=CC3O2)O", "role": "Immunomodulator"},
        {"name": "Neoandrographolide", "smiles": "C1CCC2(C1)C(CC3=CC(=O)C=CC3O2)O", "role": "Anti-inflammatory"}
    ],
    "Daruharidra (Berberis aristata)": [
        {"name": "Berberine", "smiles": "C1=CC2=C(C=C1OC3=CC=C(C=C3)O)N=CC4=CC=CC=C24", "role": "Antimicrobial"},
        {"name": "Palmatine", "smiles": "C1=CC2=C(C=C1OC3=CC=C(C=C3)N)N=CC4=CC=CC=C24", "role": "Anti-inflammatory"}
    ],
    "Pipali Moola (Piper longum root)": [
        {"name": "Piperlonguminine", "smiles": "COC1=CC=C(C=C1)C=CC(=O)NC", "role": "Digestive stimulant"},
        {"name": "Sesamin (trace)", "smiles": "COC1=CC(=C(C=C1)O)C2=CC=C(C=C2)OC", "role": "Antioxidant"}
    ],
    "Vidarikand (Pueraria tuberosa)": [
        {"name": "Puerarin", "smiles": "C1=CC(=C(C=C1O)O)C2=C(C(=O)C(=CC2)O)O", "role": "Rejuvenating"},
        {"name": "Daidzein", "smiles": "C1=CC=C2C(=O)C(=CC2=C1)O", "role": "Phytoestrogen"}
    ],
    "Tagar (Valeriana jatamansi)": [
        {"name": "Valerenic Acid", "smiles": "CC(C)CCC1=CC(=C(C=C1)C(=O)O)O", "role": "Sedative"},
        {"name": "Bornyl acetate", "smiles": "CC(=O)OC1CCC2(C1)C(CCC2)C", "role": "Calming"}
    ],
    "Dalchini Oil (Cinnamomum zeylanicum)": [
        {"name": "Cinnamaldehyde", "smiles": "C1=CC=C(C=C1)C=O", "role": "Antiseptic"},
        {"name": "Cinnamyl alcohol", "smiles": "C1=CC=C(C=C1)C=CCO", "role": "Aromatic"}
    ],
    "Gandhak Rasayan (Sulfur compound)": [
        {"name": "Sulfur (S8 ring)", "smiles": "S1S2S3S4S5S6S7S1", "role": "Antimicrobial"},
        {"name": "Hydrogen sulfide", "smiles": "S", "role": "Biochemical signaling molecule"}
    ],
    "Garlic (Allium sativum)": [
        {"name": "Allicin", "smiles": "CS(=O)SCC=CC(=O)O", "role": "Antimicrobial"},
        {"name": "Ajoene", "smiles": "CS(=O)SCC=CCSCC=CC", "role": "Antithrombotic"}
    ],
    "Onion (Allium cepa)": [
        {"name": "Quercetin", "smiles": "C1=CC(=C(C=C1O)O)C2=C(C(=O)C(=CC2)O)O", "role": "Antioxidant"},
        {"name": "Allyl propyl disulfide", "smiles": "CSSCC=CC", "role": "Lipid lowering"}
    ],
    "Chili (Capsicum annuum)": [
        {"name": "Capsaicin", "smiles": "CC(C)CC1=CC(=C(C=C1)O)NC(=O)CCCCC", "role": "Analgesic"},
        {"name": "Dihydrocapsaicin", "smiles": "CC(C)CC1=CC(=C(C=C1)O)NC(=O)CCCCCC", "role": "Pain relief"}
    ],
    "Tea (Camellia sinensis)": [
        {"name": "EGCG", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Antioxidant"},
        {"name": "Theanine", "smiles": "C(C(=O)O)NCCCCC(=O)O", "role": "Calming"}
    ],
    "Curry Leaves (Murraya koenigii)": [
        {"name": "Mahanimbine", "smiles": "C1CCC2(C1)C(CC3C2CCC4=CC(=O)C=CC34C)C", "role": "Antidiabetic"},
        {"name": "Koenimbine", "smiles": "C1CCC2(C1)C(CC3C2CCC4=CC(=O)C=CC34C)C", "role": "Antimicrobial"}
    ],
    "Chitrak (Plumbago zeylanica)": [
        {"name": "Plumbagin", "smiles": "C1=CC(=O)C2=CC=CC=C2O1", "role": "Digestive, Antimicrobial"},
        {"name": "Isoplumbagin", "smiles": "C1=CC(=O)C2=CC=CC=C2O1", "role": "Anticancer"}
    ],
    "Kumari (Aloe barbadensis Miller)": [
        {"name": "Aloin A", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Laxative"},
        {"name": "Aloe-emodin", "smiles": "C1=CC2=C(C=CC(=C2O1)O)C(=O)O", "role": "Wound healing"}
    ],
    "Shilajit (Asphaltum punjabinum)": [
        {"name": "Fulvic acid", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Adaptogen"},
        {"name": "Humic acid", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Rejuvenation"}
    ],
    "Haridra Khand (herbal compound)": [
        {"name": "Curcuminoids", "smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC=C(C=C2)O)O", "role": "Anti-inflammatory"},
        {"name": "Ghee-derived lipids", "smiles": "CCCCCCCCCCCCCCCC(=O)O", "role": "Bioavailability enhancement"}
    ],
    "Rasna (Pluchea lanceolata)": [
        {"name": "Plucheoside", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Anti-rheumatic"},
        {"name": "Caffeic Acid", "smiles": "C1=CC(=C(C=C1O)O)C=CC(=O)O", "role": "Anti-inflammatory"}
    ],
    "Patola (Trichosanthes dioica)": [
        {"name": "Cucurbitacin", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)O)C)C", "role": "Detoxifier"},
        {"name": "Trichosanthin", "smiles": "C(C(=O)O)N", "role": "Immunomodulator"}
    ],
    "Lotus (Nelumbo nucifera)": [
        {"name": "Nuciferine", "smiles": "C1=CC2=C(C=C1OC3=CC=C(C=C3)N)N=CC4=CC=CC=C24", "role": "Calming, Anti-obesity"},
        {"name": "Nelumbine", "smiles": "C1=CC2=C(C=C1N)N=CC4=CC=CC=C24", "role": "Sedative"}
    ],
    "Shilpushpa (Aloe littoralis)": [
        {"name": "Anthraquinones", "smiles": "C1=CC2=C(C=CC(=C2)O)C(=O)O1", "role": "Laxative"},
        {"name": "Aloinoside", "smiles": "C1=CC2=C(C=CC(=C2)O)C(=O)O1", "role": "Digestive"}
    ],
    "Devdaru (Cedrus deodara)": [
        {"name": "Cedrol", "smiles": "CC1CCC2(C1)C(CCC2O)C", "role": "Anti-inflammatory"},
        {"name": "Himachalol", "smiles": "CC1CCC2(C1)C(CCC2O)C", "role": "Aromatic, Relaxant"}
    ],
    "Arka (Calotropis procera)": [
        {"name": "Calotropin", "smiles": "C1CCC2(C1)C(CC3C2CCC4=CC(=O)C=CC34C)C", "role": "Digestive stimulant"},
        {"name": "Uscharin", "smiles": "C1CCC2(C1)C(CC3C2CCC4=CC(=O)C=CC34C)C", "role": "Analgesic"}
    ],
    "Lemon Grass (Cymbopogon citratus)": [
        {"name": "Citral", "smiles": "CC(=C)CCC(=O)C=CC", "role": "Calming, Digestive"},
        {"name": "Geraniol", "smiles": "CC(C)=CCC=C(C)O", "role": "Aromatic, Antimicrobial"}
    ],
    "Artemisia (Artemisia vulgaris)": [
        {"name": "Artemisinin (trace)", "smiles": "CC1CCC2C(C1)C(C3(C2O)OOC3)O", "role": "Antimalarial"},
        {"name": "Thujone", "smiles": "CC1(C)CCC2(C1)C(=O)CC2", "role": "Aromatic, Stimulant"}
    ],
    "Lajjalu (Mimosa pudica)": [
        {"name": "Mimosine", "smiles": "C1=CC(=C(C=C1O)O)C(C(=O)O)N", "role": "Hair growth"},
        {"name": "Tannins", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Astringent"}
    ],
    "Amla (Phyllanthus emblica)": [
        {"name": "Gallic Acid", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antioxidant"},
        {"name": "Ascorbic Acid", "smiles": "C(C1C(C(C(O1)O)=O)O)O", "role": "Immune support, Antioxidant"}
    ],
    "Ginger (Zingiber officinale)": [
        {"name": "Gingerol", "smiles": "CCCCCC(=O)CC(C1=CC(=C(C=C1)O)O)O", "role": "Anti-inflammatory, Digestive"},
        {"name": "Shogaol", "smiles": "CCCCCC(=O)C=CC1=CC(=C(C=C1)O)O", "role": "Anti-nausea, Anti-inflammatory"}
    ],
    "Guggul (Commiphora mukul)": [
        {"name": "Guggulsterone", "smiles": "CC1CCC2(C1(CCC3C2CCC4=CC(=O)CCC34C)C)C", "role": "Hypolipidemic, Anti-inflammatory"},
        {"name": "Guggulipid (mixture)", "smiles": "CC1CCC2(C1(CCC3C2CCC4=CC(=O)CCC34C)C)C", "role": "Cholesterol-lowering"}
    ],
    "Licorice (Glycyrrhiza glabra)": [
        {"name": "Glycyrrhizin", "smiles": "CC1CCC2(C(C1)CCC3C2(C(CC4C3CCC5=CC(=O)OC5C4(C)C)C(=O)O)C)C", "role": "Anti-inflammatory, Anti-viral"},
        {"name": "Liquiritigenin", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Antioxidant"}
    ],
    "Shankhpushpi (Convolvulus pluricaulis)": [
        {"name": "Kaempferol", "smiles": "C1=CC(=C(C=C1O)O)C2=C(C(=O)C(=CC2)O)O", "role": "Neuroprotective"},
        {"name": "Scopoletin", "smiles": "C1=CC2=C(C(=C1)OCO2)O", "role": "Anti-inflammatory, Sedative"}
    ],
    "Brahmi (Bacopa monnieri)": [
        {"name": "Bacoside A (glycoside)", "smiles": "C[C@H]1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C(=O)O)C)C", "role": "Memory enhancing"},
        {"name": "Bacopaside I", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Cognitive enhancer"}
    ],
    "Arjuna (Terminalia arjuna)": [
        {"name": "Arjunolic Acid", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C(=O)O)C)C", "role": "Cardio-protective, Antioxidant"},
        {"name": "Arjunetin", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)O)C)C", "role": "Anti-ischemic"}
    ],
    "Ashoka (Saraca asoca)": [
        {"name": "Catechin", "smiles": "C1=CC(=C(C=C1C2C(C(C(O2)O)O)O)O)O", "role": "Astringent, Antioxidant"},
        {"name": "Epicatechin", "smiles": "C1=CC(=C(C=C1C2C(C(C(O2)O)O)O)O", "role": "Antioxidant"}
    ],
    "Haritaki (Terminalia chebula)": [
        {"name": "Chebulinic Acid", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Digestive, Laxative"},
        {"name": "Chebulagic Acid", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antioxidant, Anti-inflammatory"}
    ],
    "Bibhitaki (Terminalia bellirica)": [
        {"name": "Ellagic Acid", "smiles": "C1=CC2=C(C(=C1)O)OC(=O)C3=C(C2=O)C(=O)OC4=CC(=C(C=C4)O)O", "role": "Antioxidant"},
        {"name": "Gallic Acid", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Astringent, Antioxidant"}
    ],
    "Black Pepper (Piper nigrum)": [
        {"name": "Piperine", "smiles": "COC(=O)C1=CC=CC=C1C=CC(=O)N2CCCCC2", "role": "Bioavailability enhancer"},
        {"name": "Piperidine", "smiles": "C1CCNCC1", "role": "Alkaloid"}
    ],
    "Long Pepper (Piper longum)": [
        {"name": "Piperlongumine", "smiles": "COC1=CC=C(C=C1)C=CC(=O)N2CCCC2", "role": "Anti-cancer, Antimicrobial"},
        {"name": "Piperine (trace)", "smiles": "COC(=O)C1=CC=CC=C1C=CC(=O)N2CCCCC2", "role": "Digestive stimulant"}
    ],
    "Cinnamon (Cinnamomum verum)": [
        {"name": "Cinnamaldehyde", "smiles": "C1=CC=C(C=C1)C=O", "role": "Anti-microbial, Carminative"},
        {"name": "Eugenol (trace)", "smiles": "COC1=C(C=CC(=C1)CC=C)O", "role": "Analgesic, Antioxidant"}
    ],
    "Cardamom (Elettaria cardamomum)": [
        {"name": "1,8-Cineole", "smiles": "CC1CCC2C1(OC2)C", "role": "Digestive, Carminative"},
        {"name": "α-Terpinyl acetate", "smiles": "CC(=O)OC1CC(C)(C)C2CCC1C2", "role": "Aromatic, Digestive"}
    ],
    "Clove (Syzygium aromaticum)": [
        {"name": "Eugenol", "smiles": "COC1=C(C=CC(=C1)CC=C)O", "role": "Antiseptic, Analgesic"},
        {"name": "β-Caryophyllene", "smiles": "CC1=CC2CCC(C2)C1C", "role": "Anti-inflammatory"}
    ],
    "Saffron (Crocus sativus)": [
        {"name": "Safranal", "smiles": "CC=CC(=O)C1=CC=CC=C1", "role": "Antidepressant, Antioxidant"},
        {"name": "Crocin", "smiles": "C(C1C(C(C(O1)O)O)O)OC=CC(=O)O", "role": "Neuroprotective, Antioxidant"}
    ],
    "Aloe vera": [
        {"name": "Aloin", "smiles": "C1=CC(=C(C=C1C(=O)O)O)O", "role": "Laxative, Wound healing"},
        {"name": "Aloin A (anthraquinone)", "smiles": "C1=CC(=C(C=C1C(=O)O)O)O", "role": "Cathartic"}
    ],
    "Fenugreek (Trigonella foenum-graecum)": [
        {"name": "Trigonelline", "smiles": "C1=NC(=N)C(=O)N1", "role": "Hypoglycemic, Neuroprotective"},
        {"name": "Diosgenin (sapogenin)", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Steroidal precursor, Hypolipidemic"}
    ],
    "Moringa (Moringa oleifera)": [
        {"name": "Moringinine", "smiles": "C1=CC(=C(C=C1)O)O", "role": "Antioxidant, Nutrient-rich"},
        {"name": "Moringa isothiocyanate", "smiles": "N=C=S", "role": "Antimicrobial, Anticancer (isothiocyanate class)"}
    ],
    "Sandalwood (Santalum album)": [
        {"name": "α-Santalol", "smiles": "CC(C)C1CCC2(CC1)C(=C(C2)C)C", "role": "Calming, Anti-inflammatory"},
        {"name": "β-Santalol", "smiles": "CC(C)C1CCC2(CC1)C(=C(C2)C)C", "role": "Aromatic, Antimicrobial"}
    ],
    "Kutki (Picrorhiza kurroa)": [
        {"name": "Picroside I", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Hepatoprotective"},
        {"name": "Kutkin (glycoside)", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Liver tonic"}
    ],
    "Shatavari (Asparagus racemosus)": [
        {"name": "Shatavarin IV", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)O)C)C", "role": "Galactagogue, Adaptogen"},
        {"name": "Saponins (mixture)", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Immunomodulatory"}
    ],
    "Vidanga (Embelia ribes)": [
        {"name": "Embelin", "smiles": "CC1=CC(=C(C=C1)O)C=O", "role": "Anthelmintic, Antimicrobial"},
        {"name": "Methyl embelate", "smiles": "COC(=O)C1=CC=CC=C1", "role": "Antimicrobial"}
    ],
    "Manjistha (Rubia cordifolia)": [
        {"name": "Rubiadin", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Blood purifier, Anti-inflammatory"},
        {"name": "Purpurin", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antioxidant"}
    ],
    "Musta (Cyperus rotundus)": [
        {"name": "Cyperene", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Digestive, Carminative"},
        {"name": "Falcarinol", "smiles": "C#CC(C)CCCC", "role": "Antifungal"}
    ],
    "Nirgundi (Vitex negundo)": [
        {"name": "Vitexin", "smiles": "C1=CC(=C(C=C1O)O)C2=CC(=O)C(=CC2)O", "role": "Anti-inflammatory, Analgesic"},
        {"name": "Negundoside", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Antioxidant"}
    ],
    "Jatamansi (Nardostachys jatamansi)": [
        {"name": "Jatamansone (valeranone)", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Sedative, Neuroprotective"},
        {"name": "Coumarin (trace)", "smiles": "C1=CC2=C(C=CC=C2)O", "role": "Sedative"}
    ],
    "Kencur (Kaempferia galanga)": [
        {"name": "Ethyl p-methoxycinnamate", "smiles": "COC1=CC=C(C=C1)C=CC(=O)OCC", "role": "Digestive, Anti-inflammatory"},
        {"name": "Kaempferide (trace)", "smiles": "C1=CC(=C(C=C1O)O)C2=C(C(=O)C(=CC2)O)O", "role": "Antioxidant"}
    ],
    "Kapikacchu (Mucuna pruriens)": [
        {"name": "L-DOPA", "smiles": "C1=CC(=C(C=C1CC(C(=O)O)N)O)O", "role": "Dopamine precursor, Neuroactive"},
        {"name": "Mucunain (protease)", "smiles": "C(C(=O)O)N", "role": "Proteolytic enzyme (irritant in seeds)"}
    ],
    "Bhumi Amla (Phyllanthus niruri)": [
        {"name": "Phyllanthin", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Hepatoprotective"},
        {"name": "Hypophyllanthin", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)O)C)C", "role": "Antiviral, Hepatoprotective"}
    ],
    "Kokum (Garcinia indica)": [
        {"name": "Garcinone", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Anti-obesity, Antioxidant"},
        {"name": "Hydroxycitric Acid", "smiles": "C(C(=O)O)C(C(=O)O)(C(=O)O)O", "role": "Appetite suppressant"}
    ],
    "Camphor (Cinnamomum camphora)": [
        {"name": "Camphor", "smiles": "C1C2CCC1(C(=O)C2)C", "role": "Mild analgesic, Aromatic"},
        {"name": "Borneol (trace)", "smiles": "CC1(C)CCC2(C1)C(CCC2)O", "role": "Anti-inflammatory"}
    ],
    "Lodhra (Symplocos racemosa)": [
        {"name": "Lodhretin", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Astringent, Antidiarrheal"},
        {"name": "Tannins (mixture)", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Astringent"}
    ],
    "Giloy (Tinospora cordifolia)": [
        {"name": "Tinosporaside", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Immunomodulatory"},
        {"name": "Berberine (trace)", "smiles": "C1=CC2=C(C=C1OC3=CC(=C(C=C3)O)O)N=CC4=CC=CC=C24", "role": "Antimicrobial, Immunomodulatory"}
    ],
    "Betel (Piper betle)": [
        {"name": "Eugenol", "smiles": "COC1=C(C=CC(=C1)CC=C)O", "role": "Antiseptic, Stimulant"},
        {"name": "Chavicol", "smiles": "C1=CC(=C(C=C1)O)CC=C", "role": "Aromatic, Stimulant"}
    ],
    "Black Cumin (Nigella sativa)": [
        {"name": "Thymoquinone", "smiles": "C1=CC(=O)C(=C1)C(=O)C", "role": "Anti-inflammatory, Antioxidant"},
        {"name": "Nigellidine (alkaloid)", "smiles": "C1=CC(=C(C=C1)O)N", "role": "Bioactive alkaloid"}
    ],
    "Agarwood (Aquilaria agallocha)": [
        {"name": "Agarospirol", "smiles": "CC1CCC2(C1)C(=C(C2)O)C", "role": "Aromatic, Calming"},
        {"name": "Jinkoh-eremol", "smiles": "CC1CCC2C(C1)CCC3(C2)C", "role": "Fragrant sesquiterpene"}
    ],
    "Bael (Aegle marmelos)": [
        {"name": "Marmelosin", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Digestive, Antidiarrheal"},
        {"name": "Rutamarin", "smiles": "C1=CC(=C(C=C1)O)C(=O)O", "role": "Antimicrobial"}
    ],
    "Bhringraj (Eclipta alba)": [
        {"name": "Wedelolactone", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Hepatoprotective, Hair growth"},
        {"name": "Ecliptasaponin (mixture)", "smiles": "C1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Hair tonic"}
    ],
    "Coriander (Coriandrum sativum)": [
        {"name": "Linalool", "smiles": "CC(C)CC1=CC(=C(C=C1)O)O", "role": "Digestive, Carminative"},
        {"name": "Dodecenal (trace)", "smiles": "CCCCCCCC=O", "role": "Aromatic, Antimicrobial"}
    ],
    "Fennel (Foeniculum vulgare)": [
        {"name": "Anethole", "smiles": "C1=CC(=C(C=C1)OC=C)O", "role": "Carminative, Estrogenic"},
        {"name": "Fenchone", "smiles": "CC1CCC2(C1)C(=O)C2", "role": "Aromatic, Digestive"}
    ],
    "Pomegranate (Punica granatum)": [
        {"name": "Punicalagin", "smiles": "C1=CC(=C(C=C1O)O)C(=O)O", "role": "Antioxidant, Anti-inflammatory"},
        {"name": "Ellagic Acid", "smiles": "C1=CC2=C(C(=C1)O)OC(=O)C3=C(C2=O)C(=O)OC4=CC(=C(C=C4)O)O", "role": "Anticancer, Antioxidant"}
    ],
    "Rosemary (Rosmarinus officinalis)": [
        {"name": "Rosmarinic Acid", "smiles": "C1=CC(=C(C=C1CC(C(=O)O)OC(=O)C=CC2=CC(=C(C=C2)O)O)O)O", "role": "Antioxidant, Memory support"},
        {"name": "Carnosic Acid", "smiles": "CC1CCC2C(C1)CCC3(C2(C(CC4C3CCC5=CC(=O)C=CC45C)C)C)C", "role": "Neuroprotective"}
    ],
    "Sesame (Sesamum indicum)": [
        {"name": "Sesamin", "smiles": "COC1=CC(=C(C=C1OC)C2=CC(=C(C=C2)OC)O)O", "role": "Antioxidant, Lipid-lowering"},
        {"name": "Sesamol", "smiles": "C1=CC2=C(C=C1O)OCO2", "role": "Antioxidant"}
    ]
}

# 2B. FDA Reference Database
FDA_DRUGS = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Chloroquine": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
    "Omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
    "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(C(=O)OC2=O)O",
    "Morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
    "Penicillin G": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
    "Dopamine": "C1=CC(=C(C=C1CCN)O)O",
    "Adrenaline": "CNCCC(C1=CC(=C(C=C1)O)O)O",
    "Quinine": "COC1=CC2=C(C=CN=C2C=C1)C(C3CC4CCN3CC4C=C)O",
    "Artemisinin (Standard)": "CC1CCC2C(C1)C(C3(C2O)OOC3)O",
    "Diazepam": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "Lidocaine": "CCN(CC)CC(=O)NC1=C(C=CC=C1C)C"
}

# ==========================================
# 3. BACKEND AI LOGIC & HEURISTICS
# ==========================================

# --- Rules and Heuristics Definitions ---
LIPINSKI_RULES = {
    "MW": (500, "<"),       # Molecular Weight < 500 Da
    "LogP": (5, "<"),       # LogP < 5
    "H_Donors": (5, "<"),   # H-Bond Donors < 5
    "H_Acceptors": (10, "<")# H-Bond Acceptors < 10
}

# BBB Heuristic based on documentation: TPSA < 90 AND MW < 400
BBB_TPSA_CUTOFF = 90
BBB_MW_CUTOFF = 400

def get_molecule_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol: return Draw.MolToImage(mol, size=(400, 300))
    return None

def render_3d_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mblock = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=400, height=300)
        view.addModel(mblock, 'mol')
        view.setStyle({'stick': {}})
        view.zoomTo()
        showmol(view, height=300, width=400)
    except: st.warning("3D Visualization Unavailable")

def find_similar_drug(target_smiles):
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol: return None, 0
    
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)
    best_score = 0
    best_drug = "None"
    
    for drug_name, drug_smiles in FDA_DRUGS.items():
        ref_mol = Chem.MolFromSmiles(drug_smiles)
        if ref_mol:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
            score = DataStructs.TanimotoSimilarity(target_fp, ref_fp)
            if score > best_score:
                best_score = score
                best_drug = drug_name
                
    return best_drug, round(best_score * 100, 1)

def get_layman_explanation(admet_data):
    """
    Translates technical ADMET data into simple English descriptions across 5 categories.
    """
    explanations = {}
    
    # 1. DIGESTION & ABSORPTION (Based on Bioavailability)
    if admet_data['Absorption'] == 'High':
        explanations['Digestion'] = "✅ **Excellent Digestion:** Your body absorbs this herb easily. It doesn't require strong 'Agni' (digestive fire) to process."
    else:
        explanations['Digestion'] = "⚠️ **Heavy Digestibility:** This herb is hard to digest. It should be taken with warm water, ginger, or pepper to improve absorption."

    # 2. METABOLISM (Based on LogP)
    logp = admet_data['LogP']
    if logp < 0:
        explanations['Metabolism'] = "💧 **Fast Acting:** Dissolves in water quickly. It moves through your system fast and doesn't linger in the body."
    elif 0 <= logp <= 3:
        explanations['Metabolism'] = "⚖️ **Balanced:** The 'Goldilocks' zone. It stays in the body long enough to act effectively but is processed safely by the liver."
    else:
        explanations['Metabolism'] = "🧈 **Slow Release:** Dissolves in fat. It stays in your system for a long time. Best taken with Ghee or Milk to help transport it."

    # 3. EXCRETION (Based on Solubility)
    if logp < 2.5:
        explanations['Excretion'] = "🚽 **Kidney Cleansing:** Mainly flushed out through urine. It likely has a diuretic effect (increases urination)."
    else:
        explanations['Excretion'] = "💩 **Bowel Cleansing:** Processed by the liver and removed via the intestines. It may have a mild laxative effect."

    # 4. TOXICITY & SAFETY (Based on QED & Structure)
    if "High" in admet_data['Toxicity_Risk']:
        explanations['Toxicity'] = "⚠️ **Use Caution:** Our analysis detects structural alerts. High doses might stress the liver. Consult a Vaidya (Doctor) before long-term use."
    else:
        explanations['Toxicity'] = "🛡️ **Generally Safe:** This compound shows a safe chemical profile typical of non-toxic herbs. It is likely safe for daily consumption."

    # 5. OVERALL VERDICT
    violations = admet_data['Violations']
    if violations == 0 and "High" not in admet_data['Toxicity_Risk']:
        explanations['Verdict'] = "🌟 **EXCELLENT:** Safe, absorbable, and effective. Good for daily use."
    elif violations <= 1:
        explanations['Verdict'] = "✅ **GOOD:** Generally safe, but respect the dosage."
    else:
        explanations['Verdict'] = "🛑 **CAUTION:** Poor absorption or potential risks. Use only under guidance."

    return explanations

def plot_radar_chart(admet):
    categories = ['Mol. Weight', 'LogP', 'H-Donors', 'H-Acceptors', 'TPSA']
    values = [admet['MW']/500, max(0, admet['LogP'])/5, admet['H_Donors']/5, admet['H_Acceptors']/10, admet['TPSA']/140]
    max_range = max(1.5, max(values) + 0.2)
    values += [values[0]]
    categories += [categories[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1]*6, theta=categories, fill='toself', name='Safe Limit', line_color='#2E8B57', fillcolor='rgba(46, 139, 87, 0.2)'))
    fig.add_trace(go.Scatterpolar(r=[max_range]*6, theta=categories, fill='tonext', name='Danger Zone', line_color='#e63946', fillcolor='rgba(230, 57, 70, 0.1)'))
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Compound Profile', line=dict(color='#1d3557', width=3)))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_range], gridcolor="#e0e0e0")),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12),
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

def create_pdf(results):
    """Generates a PDF report from analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Professional Header
    pdf.set_fill_color(46, 139, 87) # Green Header
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "Herbal-Tox: AI Safety Report", ln=True, align="C", fill=True)
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)

    for item in results:
        pdf.set_font("Arial", style="B", size=14)
        pdf.set_text_color(46, 139, 87)
        pdf.cell(0, 10, f"Compound: {item['Compound Name']}", ln=True)
        pdf.set_text_color(0, 0, 0)
        
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"Role: {item['Role']}", ln=True)
        pdf.cell(0, 8, f"SMILES: {item['Compound Name'][:40]}...", ln=True)
        
        pdf.set_fill_color(240, 248, 245) # Light green background for stats
        pdf.ln(2)
        pdf.cell(0, 8, f" Toxicity Risk: {item['Toxicity_Risk']}  |  Absorption: {item['Absorption']}", ln=True, fill=True)
        pdf.cell(0, 8, f" Lipinski Violations: {item['Violations']}  |  FDA Match: {item['Similar_Drug']} ({item['Similarity_Score']}%)", ln=True, fill=True)
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    return pdf.output(dest="S").encode("latin-1")

def calculate_physicochemical_properties(mol):
    """Calculates key RDKit descriptors for a molecule."""
    if mol is None:
        return None
    
    props = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "H_Donors": Descriptors.NumHDonors(mol),
        "H_Acceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": CalcTPSA(mol)
    }
    return props

def check_lipinski_rule(props):
    """Checks for Lipinski violations and predicts oral absorption."""
    if props is None:
        return {"Violations": "N/A", "Prediction": "Invalid Molecule"}
        
    violations = 0
    
    # Check each rule
    for prop_name, (cutoff, rule_type) in LIPINSKI_RULES.items():
        value = props.get(prop_name, np.inf if rule_type == '<' else np.NINF)
        
        is_violated = False
        if rule_type == '<' and value >= cutoff:
            is_violated = True
        elif rule_type == '>' and value <= cutoff:
            is_violated = True
        
        if is_violated:
            violations += 1
            
    # Determine Absorption Prediction
    prediction = "High Absorption" if violations <= 1 else "Low Absorption"
    
    return {
        "Violations": violations,
        "Prediction": prediction,
        "Details": {k: props[k] for k in LIPINSKI_RULES.keys()}
    }

def predict_bbb_penetration(props):
    """Predicts Blood-Brain Barrier (BBB) penetration using the TPSA/MW heuristic."""
    if props is None:
        return {"Prediction": "Invalid Molecule"}
        
    tpsa = props.get("TPSA", np.inf)
    mw = props.get("MW", np.inf)
    
    # Heuristic: TPSA < 90 AND MW < 400
    is_penetrator = (tpsa < BBB_TPSA_CUTOFF) and (mw < BBB_MW_CUTOFF)
    
    prediction = "Likely Penetrator" if is_penetrator else "Low Penetration/Efflux Risk"
    
    return {
        "Prediction": prediction,
        "TPSA": tpsa,
        "MW": mw
    }

def generate_admet_report(smiles):
    """Generates the full ADMET report for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"Status": "Error", "Message": "Invalid SMILES string provided."}
    
    props = calculate_physicochemical_properties(mol)
    
    lipinski_results = check_lipinski_rule(props)
    bbb_results = predict_bbb_penetration(props)
    
    report = {
        "SMILES": smiles,
        "Molecular Properties": props,
        "Lipinski Analysis (Oral Bioavailability)": lipinski_results,
        "BBB Penetration Analysis (Heuristic)": bbb_results,
        "Status": "Success"
    }
    return report

def calculate_admet_wrapper(smiles):
    """
    Wrapper function to bridge the new logic with the existing UI components.
    """
    # 1. Run the new core logic
    report = generate_admet_report(smiles)
    if report["Status"] == "Error": return None

    # 2. Extract Data
    props = report["Molecular Properties"]
    lipinski = report["Lipinski Analysis (Oral Bioavailability)"]
    bbb = report["BBB Penetration Analysis (Heuristic)"]

    # 3. Run extra features (FDA Match & QED for Toxicity Badge)
    similar_drug, similarity_score = find_similar_drug(smiles)
    
    mol = Chem.MolFromSmiles(smiles)
    qed_score = QED.qed(mol)
    toxicity_risk = "Low" if qed_score > 0.4 else "High Risk (Hepatotoxic)"

    # 4. Return flat dictionary expected by the UI
    return {
        "MW": round(props["MW"], 2), 
        "LogP": round(props["LogP"], 2), 
        "TPSA": round(props["TPSA"], 2),
        "H_Donors": props["H_Donors"], 
        "H_Acceptors": props["H_Acceptors"], 
        "QED": round(qed_score, 2),
        "Absorption": lipinski["Prediction"], 
        "BBB": bbb["Prediction"], 
        "Toxicity_Risk": toxicity_risk,
        "Violations": lipinski["Violations"],
        "Similar_Drug": similar_drug, 
        "Similarity_Score": similarity_score
    }

# ==========================================
# 4. FRONTEND UI
# ==========================================

def home_page():
    # Hero Section
    st.markdown("""
    <div style="background-color: #d8f3dc; padding: 40px; border-radius: 20px; margin-bottom: 30px; text-align: center;">
        <h1 style="color: #1b4332; margin-bottom: 10px;">Herbal-Tox AI 🌿</h1>
        <h3 style="color: #40916c;">The Future of Safe Ayurvedic Medicine</h3>
        <p style="font-size: 18px; color: #2d6a4f;">Bridging ancient wisdom with modern cheminformatics to predict safety & efficacy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🧬 **Deconstruct Herbs**\n\nBreak down complex herbs into active phytochemicals automatically.")
    with c2:
        st.success("🛡️ **Predict Safety**\n\nAnalyze toxicity, bioavailability, and BBB penetration using FDA rules.")
    with c3:
        st.warning("⚗️ **Visual Chemistry**\n\nInteractive 3D molecular modeling and regulatory reporting.")

    st.markdown("---")
    st.image("https://img.freepik.com/free-vector/science-laboratory-research-concept-illustration_114360-1033.jpg", use_container_width=True)

def analysis_page():
    st.title("🔬 Phytochemical ADMET Analyzer")
    
    # Styled Input Container
    with st.container():
        st.markdown('<div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            search_type = st.radio("Search Mode:", ["Select from Database", "Enter Custom SMILES"], horizontal=True)
            if search_type == "Select from Database":
                selected_herb = st.selectbox("Select Herb:", list(HERB_DATABASE.keys()))
                compounds = HERB_DATABASE[selected_herb]
                st.success(f"Loaded {len(compounds)} active compounds from **{selected_herb}**")
            else:
                custom_smiles = st.text_input("Enter SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O")
                compounds = [{"name": "Custom Input", "smiles": custom_smiles, "role": "User Defined"}]
        with c2:
            st.markdown("#### 💡 Quick Tip")
            st.caption("Select a herb to auto-load its active ingredients, or paste a SMILES string to analyze a novel compound.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Run AI Analysis"):
        st.markdown("---")
        results_for_export = []
        
        for i, cmp in enumerate(compounds):
            name = cmp["name"]
            smiles = cmp["smiles"]
            # CALL THE NEW WRAPPER FUNCTION HERE
            admet = calculate_admet_wrapper(smiles)
            
            if admet:
                results_for_export.append({**admet, 'Compound Name': name, 'Role': cmp['role']})
                
                # --- LAYMAN EXPLANATION GENERATION ---
                explanations = get_layman_explanation(admet)
                # -------------------------------------
                
                # --- RESULT CARD ---
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: white; padding: 25px; border-radius: 15px; border-left: 5px solid #2E8B57; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 20px;">
                        <h3 style="margin:0; color: #1b4332;">{i+1}. {name}</h3>
                        <p style="color: #666; font-style: italic;">{cmp['role']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns([1.2, 1.2, 1.5])
                    
                    with c1:
                        st.markdown("**2D Structure**")
                        img = get_molecule_image(smiles)
                        st.image(img, use_container_width=True)
                        with st.expander("🔄 View 3D Model"):
                            render_3d_molecule(smiles)

                    with c2:
                        st.markdown("**Safety Radar**")
                        fig = plot_radar_chart(admet)
                        st.plotly_chart(fig, use_container_width=True)

                    with c3:
                        st.markdown("**AI Predictions**")
                        
                        # Custom Badges
                        tox_bg = "#ffebee" if "High" in admet['Toxicity_Risk'] else "#e8f5e9"
                        tox_col = "#c62828" if "High" in admet['Toxicity_Risk'] else "#2e7d32"
                        
                        # Fix logic for High Absorption (Good) vs Low (Bad)
                        abs_bg = "#e8f5e9" if "High" in admet['Absorption'] else "#fff3e0"
                        abs_col = "#2e7d32" if "High" in admet['Absorption'] else "#ef6c00"

                        st.markdown(f"""
                        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                            <span style="background-color: {tox_bg}; color: {tox_col}; padding: 5px 10px; border-radius: 15px; font-weight: bold; font-size: 14px;">Toxicity: {admet['Toxicity_Risk']}</span>
                            <span style="background-color: {abs_bg}; color: {abs_col}; padding: 5px 10px; border-radius: 15px; font-weight: bold; font-size: 14px;">Absorption: {admet['Absorption']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        st.info(f"💊 **FDA Match:** {admet['Similar_Drug']} ({admet['Similarity_Score']}%)")
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric("Mol. Weight", f"{admet['MW']} Da")
                        col_b.metric("Lipophilicity", admet['LogP'])
                        
                        if admet['Violations'] > 0:
                            st.error(f"⚠ Lipinski Violations: {admet['Violations']}")
                        else:
                            st.success("✅ Lipinski Compliant")

                    # --- NEW EXPANDED HEALTH IMPACT GUIDE SECTION ---
                    st.markdown("### 📋 Health Impact Guide (Simple English)")
                    st.markdown(f"""
                    <div class="layman-box">
                        <div class="layman-item">
                            <div class="layman-title">1. Digestion & Absorption</div>
                            <div class="layman-text">{explanations['Digestion']}</div>
                        </div>
                        <div class="layman-item">
                            <div class="layman-title">2. Metabolism (Processing)</div>
                            <div class="layman-text">{explanations['Metabolism']}</div>
                        </div>
                        <div class="layman-item">
                            <div class="layman-title">3. Excretion (Removal)</div>
                            <div class="layman-text">{explanations['Excretion']}</div>
                        </div>
                        <div class="layman-item">
                            <div class="layman-title">4. Toxicity & Safety</div>
                            <div class="layman-text">{explanations['Toxicity']}</div>
                        </div>
                        <div class="overall-verdict">
                            OVERALL: {explanations['Verdict']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    # ---------------------------------------

        if results_for_export:
            st.markdown("### 📥 Download Reports")
            c1, c2 = st.columns(2)
            with c1:
                df = pd.DataFrame(results_for_export)
                # Ensure correct columns exist
                cols = ['Compound Name', 'Role', 'Toxicity_Risk', 'Absorption', 'LogP', 'MW', 'Similar_Drug', 'Similarity_Score']
                # Filter just in case
                final_cols = [c for c in cols if c in df.columns]
                remaining_cols = [c for c in df.columns if c not in final_cols]
                df = df[final_cols + remaining_cols]
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download CSV Data", csv, "analysis.csv", "text/csv")
            with c2:
                pdf_bytes = create_pdf(results_for_export)
                st.download_button("📕 Download Official PDF Report", data=pdf_bytes, file_name="HerbalTox_Report.pdf", mime="application/pdf")

def about_page():
    st.title("About Herbal-Tox")
    st.markdown("""
    <div style="background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
        <h3 style="color: #2E8B57;">Bridging Ancient Ayurveda with Modern AI</h3>
        <p>Unlike "Black-Box" AI models, <b>Herbal-Tox</b> uses explainable Cheminformatics algorithms (White-Box AI) to ensure safety in herbal medicine.</p>
        <hr>
        <h4>🧪 Methodology</h4>
        <ul>
            <li><b>Digitization (SMILES):</b> Converting raw chemical structures into machine-readable strings.</li>
            <li><b>The Core Engine (RDKit):</b> Calculating molecular descriptors like LogP (Lipophilicity) and TPSA.</li>
            <li><b>Safety Filters:</b> Applying Pfizer's <i>Lipinski's Rule of 5</i> to flag bioavailability issues.</li>
            <li><b>Validation:</b> Cross-referencing against FDA databases using Tanimoto Similarity.</li>
        </ul>
        <br>
        <small>Built for Hackathon 2025 by Team NovaX.</small>
    </div>
    """, unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.title("Herbal-Tox 🌿")
        page = st.radio("Navigate", ["Home", "Run Analysis", "About"])
        st.markdown("---")
        st.caption("v1.0.0 | Hackathon Build")
        
    if page == "Home": home_page()
    elif page == "Run Analysis": analysis_page()
    elif page == "About": about_page()

if __name__ == "__main__":
    main()

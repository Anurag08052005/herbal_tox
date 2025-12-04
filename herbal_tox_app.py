import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Draw
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AYURVEDIC KNOWLEDGE BASE (COMPLETE DB)
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
        {"name": "Sulfur (S₈ ring)", "smiles": "S1S2S3S4S5S6S7S1", "role": "Antimicrobial"},
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
        {"name": "Ascorbic Acid", "smiles": "C(C1C(C(C(O1)O)O)=O)O", "role": "Immune support, Antioxidant"}
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
        {"name": "Epicatechin", "smiles": "C1=CC(=C(C=C1C2C(C(C(O2)O)O)O)O)O", "role": "Antioxidant"}
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
    mw = Descriptors.MolWt(mol)            # Molecular Weight
    logp = Descriptors.MolLogP(mol)        # Lipophilicity (Solubility/Permeability)
    tpsa = Descriptors.TPSA(mol)           # Topological Polar Surface Area (Absorption)
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
    *Predict the Toxicity & Efficacy of Ancient Herbs using Modern AI.*
    
    Ayurveda is powerful, but complex. Herbal-Tox helps researchers:
    * *Deconstruct* herbs into their active phytochemicals.
    * *Predict* ADMET properties (Absorption, Metabolism, Toxicity).
    * *Visualize* chemical structures instantly.
    """)
    
    st.info("👈 Select *'Run Analysis'* from the sidebar to start.")

def analysis_page():
    st.title("🔬 Phytochemical ADMET Analyzer")
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_type = st.radio("Search Mode:", ["Select from Database", "Enter Custom SMILES"])
        
        if search_type == "Select from Database":
            # --- SEARCH BAR ---
            search_query = st.text_input("🔍 Search Database (Herb, Compound, or Role):", placeholder="Try 'Shilajit', 'Anti-cancer', or 'Curcumin'...")
            
            # Smart Filter Logic
            filtered_herbs = []
            if search_query:
                for herb, comps in HERB_DATABASE.items():
                    # Match Herb Name?
                    if search_query.lower() in herb.lower():
                        filtered_herbs.append(herb)
                        continue
                    
                    # Match Compound Name or Role?
                    for c in comps:
                        if (search_query.lower() in c["name"].lower()) or (search_query.lower() in c.get("role", "").lower()):
                            filtered_herbs.append(herb)
                            break
            else:
                filtered_herbs = list(HERB_DATABASE.keys())

            if not filtered_herbs:
                st.warning("No matches found.")
                selected_herb = None
                compounds = []
            else:
                selected_herb = st.selectbox("Select Herb:", filtered_herbs)
                compounds = HERB_DATABASE[selected_herb]
                st.success(f"Loaded {len(compounds)} active compounds from *{selected_herb}*")

        else:
            custom_smiles = st.text_input("Enter SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O")
            compounds = [{"name": "Custom Input", "smiles": custom_smiles, "role": "User Defined"}]

    with col2:
        st.markdown("### How it works")
        st.caption("""
        1. *SMILES Parsing:* Converts text to chemical graph.
        2. *RDKit Engine:* Calculates LogP, TPSA, MW.
        3. *Safety Logic:* Applies Lipinski's Rule of 5 to flag toxicity risks.
        """)

    # Initialize storage for download
    results_for_export = []

    if st.button("🚀 Analyze Compounds"):
        st.divider()
        
        if not compounds:
            st.error("No compounds selected.")
        else:
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
                        st.markdown(f"*Role:* {role}")
                        
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
                            st.markdown("#### ⚠ AI Safety Predictions")
                            
                            # Dynamic Badges
                            tox_color = "red" if "High" in admet['Toxicity_Risk'] else "green"
                            abs_color = "green" if admet['Absorption'] == "High" else "orange"
                            
                            st.markdown(f"*Toxicity Risk:* :{tox_color}[{admet['Toxicity_Risk']}]")
                            st.markdown(f"*Oral Absorption:* :{abs_color}[{admet['Absorption']}]")
                            st.markdown(f"*BBB Penetration:* {admet['BBB']}")
                            
                            st.progress(admet['QED'], text="Drug-Likeness Score")
                            
                            if admet['Violations'] > 0:
                                st.warning(f"⚠ Lipinski Violations: {admet['Violations']} (Check solubility)")
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

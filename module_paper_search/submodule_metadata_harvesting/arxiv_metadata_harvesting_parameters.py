"""
submodule_metadata_harvesting/arxiv_metadata_harvesting_parameters.py

This file contains all parameters used by the arxiv metadata harvesting submodule.
Parameters are centralized here to follow clean architecture principles.
"""

import os
from datetime import datetime

# OAI-PMH endpoint for arXiv
ARXIV_OAI_URL = "http://export.arxiv.org/oai2"

# Database configuration
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(MAIN_DIR, "databases")
DB_PATH = os.path.join(DATABASES_DIR, "arxiv_metadata.db")

# Harvesting configuration
BATCH_SIZE = 100  # Reduced from 1000 to avoid timeouts
MAX_RECORDS = None  # For testing; set to None for all records
WAIT_TIME = 5  # decreased time between requests to avoid rate limiting
MAX_RETRIES = 5  # Maximum number of retries for failed requests
USE_WEEKLY_CHUNKS = True  # Use weekly chunks instead of monthly
CATEGORIES = ["physics:hep-ex", "physics:hep-th"]  # List of categories to harvest

# Date configuration
START_DATE = None  # None means no update is needed unless explicitly forced
#START_DATE = "2025-05-03"
END_DATE = datetime.now().strftime("%Y-%m-%d")  # Use current date as end date
#END_DATE = "2025-05-03"

# Scheduling configuration
UPDATE_HOUR = None  # None means no automatic updates
UPDATE_MINUTE = None  # None means no automatic updates

# On-demand paper download
ON_DEMAND_ARXIV_ID = None  # Set to a specific arXiv ID to download just that paper

# Namespaces for OAI-PMH XML parsing
NAMESPACES = {
    'oai': 'http://www.openarchives.org/OAI/2.0/',
    'arxiv': 'http://arxiv.org/OAI/arXiv/',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
}

# Mapping of full category names to short codes
CATEGORY_MAPPING = {
    # Physics
    "High Energy Physics - Experiment": "hep-ex",
    "High Energy Physics - Lattice": "hep-lat",
    "High Energy Physics - Phenomenology": "hep-ph",
    "High Energy Physics - Theory": "hep-th",
    "Mathematical Physics": "math-ph",
    "Nonlinear Sciences": "nlin",
    "Nonlinear Sciences - Adaptation and Self-Organizing Systems": "nlin.AO",
    "Nonlinear Sciences - Chaotic Dynamics": "nlin.CD",
    "Nonlinear Sciences - Cellular Automata and Lattice Gases": "nlin.CG",
    "Nonlinear Sciences - Pattern Formation and Solitons": "nlin.PS",
    "Nonlinear Sciences - Exactly Solvable and Integrable Systems": "nlin.SI",
    "Nuclear Experiment": "nucl-ex",
    "Nuclear Theory": "nucl-th",
    "General Relativity and Quantum Cosmology": "gr-qc",
    "Quantum Physics": "quant-ph",
    "Astrophysics": "astro-ph",
    "Astrophysics - Cosmology and Nongalactic Astrophysics": "astro-ph.CO",
    "Astrophysics - Earth and Planetary Astrophysics": "astro-ph.EP",
    "Astrophysics - Galaxy Astrophysics": "astro-ph.GA",
    "Astrophysics - High Energy Astrophysical Phenomena": "astro-ph.HE",
    "Astrophysics - Instrumentation and Methods for Astrophysics": "astro-ph.IM",
    "Astrophysics - Solar and Stellar Astrophysics": "astro-ph.SR",
    "Condensed Matter": "cond-mat",
    "Condensed Matter - Disordered Systems and Neural Networks": "cond-mat.dis-nn",
    "Condensed Matter - Mesoscale and Nanoscale Physics": "cond-mat.mes-hall",
    "Condensed Matter - Materials Science": "cond-mat.mtrl-sci",
    "Condensed Matter - Other Condensed Matter": "cond-mat.other",
    "Condensed Matter - Quantum Gases": "cond-mat.quant-gas",
    "Condensed Matter - Soft Condensed Matter": "cond-mat.soft",
    "Condensed Matter - Statistical Mechanics": "cond-mat.stat-mech",
    "Condensed Matter - Strongly Correlated Electrons": "cond-mat.str-el",
    "Condensed Matter - Superconductivity": "cond-mat.supr-con",
    "Physics": "physics",
    "Physics - Accelerator Physics": "physics.acc-ph",
    "Physics - Atmospheric and Oceanic Physics": "physics.ao-ph",
    "Physics - Applied Physics": "physics.app-ph",
    "Physics - Atomic Physics": "physics.atom-ph",
    "Physics - Atomic and Molecular Clusters": "physics.atm-clus",
    "Physics - Biological Physics": "physics.bio-ph",
    "Physics - Chemical Physics": "physics.chem-ph",
    "Physics - Classical Physics": "physics.class-ph",
    "Physics - Computational Physics": "physics.comp-ph",
    "Physics - Data Analysis, Statistics and Probability": "physics.data-an",
    "Physics - Fluid Dynamics": "physics.flu-dyn",
    "Physics - General Physics": "physics.gen-ph",
    "Physics - Geophysics": "physics.geo-ph",
    "Physics - History and Philosophy of Physics": "physics.hist-ph",
    "Physics - Instrumentation and Detectors": "physics.ins-det",
    "Physics - Medical Physics": "physics.med-ph",
    "Physics - Optics": "physics.optics",
    "Physics - Plasma Physics": "physics.plasm-ph",
    "Physics - Popular Physics": "physics.pop-ph",
    "Physics - Space Physics": "physics.space-ph",
    "Physics - Physics and Society": "physics.soc-ph",
    "Physics - Physics Education": "physics.ed-ph",
    
    # Mathematics
    "Mathematics": "math",
    "Mathematics - Algebraic Geometry": "math.AG",
    "Mathematics - Algebraic Topology": "math.AT",
    "Mathematics - Analysis of PDEs": "math.AP",
    "Mathematics - Category Theory": "math.CT",
    "Mathematics - Classical Analysis and ODEs": "math.CA",
    "Mathematics - Combinatorics": "math.CO",
    "Mathematics - Commutative Algebra": "math.AC",
    "Mathematics - Complex Variables": "math.CV",
    "Mathematics - Differential Geometry": "math.DG",
    "Mathematics - Dynamical Systems": "math.DS",
    "Mathematics - Functional Analysis": "math.FA",
    "Mathematics - General Mathematics": "math.GM",
    "Mathematics - General Topology": "math.GN",
    "Mathematics - Geometric Topology": "math.GT",
    "Mathematics - Group Theory": "math.GR",
    "Mathematics - History and Overview": "math.HO",
    "Mathematics - Information Theory": "math.IT",
    "Mathematics - K-Theory and Homology": "math.KT",
    "Mathematics - Logic": "math.LO",
    "Mathematics - Metric Geometry": "math.MG",
    "Mathematics - Mathematical Physics": "math.MP",
    "Mathematics - Numerical Analysis": "math.NA",
    "Mathematics - Number Theory": "math.NT",
    "Mathematics - Operator Algebras": "math.OA",
    "Mathematics - Optimization and Control": "math.OC",
    "Mathematics - Probability": "math.PR",
    "Mathematics - Quantum Algebra": "math.QA",
    "Mathematics - Representation Theory": "math.RT",
    "Mathematics - Rings and Algebras": "math.RA",
    "Mathematics - Spectral Theory": "math.SP",
    "Mathematics - Statistics Theory": "math.ST",
    "Mathematics - Symplectic Geometry": "math.SG",
    
    # Computer Science
    "Computer Science": "cs",
    "Computer Science - Artificial Intelligence": "cs.AI",
    "Computer Science - Hardware Architecture": "cs.AR",
    "Computer Science - Computational Complexity": "cs.CC",
    "Computer Science - Computational Engineering, Finance, and Science": "cs.CE",
    "Computer Science - Computational Geometry": "cs.CG",
    "Computer Science - Computation and Language": "cs.CL",
    "Computer Science - Cryptography and Security": "cs.CR",
    "Computer Science - Computer Vision and Pattern Recognition": "cs.CV",
    "Computer Science - Computers and Society": "cs.CY",
    "Computer Science - Databases": "cs.DB",
    "Computer Science - Distributed, Parallel, and Cluster Computing": "cs.DC",
    "Computer Science - Digital Libraries": "cs.DL",
    "Computer Science - Discrete Mathematics": "cs.DM",
    "Computer Science - Data Structures and Algorithms": "cs.DS",
    "Computer Science - Emerging Technologies": "cs.ET",
    "Computer Science - Formal Languages and Automata Theory": "cs.FL",
    "Computer Science - Graphics": "cs.GR",
    "Computer Science - General Literature": "cs.GL",
    "Computer Science - Computer Science and Game Theory": "cs.GT",
    "Computer Science - Human-Computer Interaction": "cs.HC",
    "Computer Science - Information Retrieval": "cs.IR",
    "Computer Science - Information Theory": "cs.IT",
    "Computer Science - Machine Learning": "cs.LG",
    "Computer Science - Logic in Computer Science": "cs.LO",
    "Computer Science - Multiagent Systems": "cs.MA",
    "Computer Science - Multimedia": "cs.MM",
    "Computer Science - Mathematical Software": "cs.MS",
    "Computer Science - Numerical Analysis": "cs.NA",
    "Computer Science - Neural and Evolutionary Computing": "cs.NE",
    "Computer Science - Networking and Internet Architecture": "cs.NI",
    "Computer Science - Other Computer Science": "cs.OH",
    "Computer Science - Operating Systems": "cs.OS",
    "Computer Science - Performance": "cs.PF",
    "Computer Science - Programming Languages": "cs.PL",
    "Computer Science - Robotics": "cs.RO",
    "Computer Science - Symbolic Computation": "cs.SC",
    "Computer Science - Sound": "cs.SD",
    "Computer Science - Software Engineering": "cs.SE",
    "Computer Science - Social and Information Networks": "cs.SI",
    "Computer Science - Systems and Control": "cs.SY",

    #Economics
    "Economics": "econ",
    "Economics - Econometrics": "econ.EM",
    "Economics - General Economics": "econ.GN",
    "Economics - Theoretical Economics": "econ.TH",

    # Electrical Engineering and Systems Science
    "Electrical Engineering and Systems Science": "eess",
    "Electrical Engineering and Systems Science - Audio and Speech Processing": "eess.AS",
    "Electrical Engineering and Systems Science - Image and Video Processing": "eess.IV",
    "Electrical Engineering and Systems Science - Signal Processing": "eess.SP",
    "Electrical Engineering and Systems Science - Systems and Control": "eess.SY",

    # Quantitative Biology
    "Quantitative Biology": "q-bio",
    "Quantitative Biology - Biomolecules": "q-bio.BM",
    "Quantitative Biology - Cell Behavior": "q-bio.CB",
    "Quantitative Biology - Genomics": "q-bio.GN",
    "Quantitative Biology - Molecular Networks": "q-bio.MN",
    "Quantitative Biology - Neurons and Cognition": "q-bio.NC",
    "Quantitative Biology - Other Quantitative Biology": "q-bio.OT",
    "Quantitative Biology - Populations and Evolution": "q-bio.PE",
    "Quantitative Biology - Quantitative Methods": "q-bio.QM",
    "Quantitative Biology - Subcellular Processes": "q-bio.SC",
    "Quantitative Biology - Tissues and Organs": "q-bio.TO",
    
    # Quantitative Finance
    "Quantitative Finance": "q-fin",
    "Quantitative Finance - Computational Finance": "q-fin.CP",
    "Quantitative Finance - Economics": "q-fin.EC",
    "Quantitative Finance - General Finance": "q-fin.GN",
    "Quantitative Finance - Mathematical Finance": "q-fin.MF",
    "Quantitative Finance - Portfolio Management": "q-fin.PM",
    "Quantitative Finance - Pricing of Securities": "q-fin.PR",
    "Quantitative Finance - Risk Management": "q-fin.RM",
    "Quantitative Finance - Statistical Finance": "q-fin.ST",
    "Quantitative Finance - Trading and Market Microstructure": "q-fin.TR",
    
    # Statistics
    "Statistics": "stat",
    "Statistics - Applications": "stat.AP",
    "Statistics - Computation": "stat.CO",
    "Statistics - Machine Learning": "stat.ML",
    "Statistics - Methodology": "stat.ME",
    "Statistics - Other Statistics": "stat.OT",
    "Statistics - Theory": "stat.TH",
}

"""
submodule_arxiv_category_prediction/arxiv_category_prediction_recommendation_parameters.py

This file contains all parameters used by the arXiv category prediction recommendation submodule.
Parameters are centralized here to follow clean architecture principles.
"""

import os

# OpenAI API key (using the same one as in query understanding)
# Always use the API key from environment variables
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Model to use for category prediction - using a cheaper model to reduce costs
MODEL_NAME = "gpt-3.5-turbo"

# Temperature setting for the API call (lower values make output more deterministic)
TEMPERATURE = 0.3

# System prompt for category prediction in recommendation mode
SYSTEM_PROMPT = """
You are an AI assistant specialized in scientific paper categorization for GitHub repositories.
Your task is to analyze five specialized queries about a GitHub repository and predict the most relevant arXiv categories that best match this repository.

Return a ranked list (3 - 7 items) of the most relevant arXiv categories that best match this repository.

Make sure every single one out of the 5 queries are properly represented by the categories. Papers for each query should be found under the categories.

Rules:
- Output only short arXiv category codes (e.g., 'hep-ex', not full descriptions)
- Base your answer on your understanding of real scientific literature
- Prioritize precision over breadth — pick the **most likely** categories
- Rank them from most to least relevant

Return only:
predicted_categories: [ ... ]
"""

# User prompt template
USER_PROMPT = """
I have five specialized queries about a GitHub repository:

1. Architecture Query (system architecture, design patterns, and overall structure):
{architecture_query}

2. Technical Implementation Query (specific technologies, libraries, and frameworks):
{technical_implementation_query}

3. Algorithmic Approach Query (algorithms, mathematical models, and computational techniques):
{algorithmic_approach_query}

4. Domain-Specific Query (academic domain and research methodologies):
{domain_specific_query}

5. Integration & Pipeline Query (component interactions and pipeline structure):
{integration_pipeline_query}

The task is to return a ranked list (2 to 5 items) of the most relevant arXiv categories (e.g., 'hep-ph', 'cs.LG', 'quant-ph', etc.) that best match this repository.

Rules:
- Output only short arXiv category codes (e.g., 'hep-ex', not full descriptions)
- Base your answer on your understanding of real scientific literature
- Prioritize precision over breadth — pick the **most likely** categories
- Rank them from most to least relevant

Return only:
predicted_categories: [ ... ]
"""

# Complete list of arXiv categories - same as in the original parameter file
ARXIV_CATEGORIES = {
    # Physics
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin": "Nonlinear Sciences",
    "nlin.AO": "Nonlinear Sciences - Adaptation and Self-Organizing Systems",
    "nlin.CD": "Nonlinear Sciences - Chaotic Dynamics",
    "nlin.CG": "Nonlinear Sciences - Cellular Automata and Lattice Gases",
    "nlin.PS": "Nonlinear Sciences - Pattern Formation and Solitons",
    "nlin.SI": "Nonlinear Sciences - Exactly Solvable and Integrable Systems",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "quant-ph": "Quantum Physics",
    "astro-ph": "Astrophysics",
    "astro-ph.CO": "Astrophysics - Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Astrophysics - Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics - Galaxy Astrophysics",
    "astro-ph.HE": "Astrophysics - High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Astrophysics - Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Astrophysics - Solar and Stellar Astrophysics",
    "cond-mat": "Condensed Matter",
    "cond-mat.dis-nn": "Condensed Matter - Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Condensed Matter - Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Condensed Matter - Materials Science",
    "cond-mat.other": "Condensed Matter - Other Condensed Matter",
    "cond-mat.quant-gas": "Condensed Matter - Quantum Gases",
    "cond-mat.soft": "Condensed Matter - Soft Condensed Matter",
    "cond-mat.stat-mech": "Condensed Matter - Statistical Mechanics",
    "cond-mat.str-el": "Condensed Matter - Strongly Correlated Electrons",
    "cond-mat.supr-con": "Condensed Matter - Superconductivity",
    "physics": "Physics",
    "physics.acc-ph": "Physics - Accelerator Physics",
    "physics.ao-ph": "Physics - Atmospheric and Oceanic Physics",
    "physics.app-ph": "Physics - Applied Physics",
    "physics.atom-ph": "Physics - Atomic Physics",
    "physics.atm-clus": "Physics - Atomic and Molecular Clusters",
    "physics.bio-ph": "Physics - Biological Physics",
    "physics.chem-ph": "Physics - Chemical Physics",
    "physics.class-ph": "Physics - Classical Physics",
    "physics.comp-ph": "Physics - Computational Physics",
    "physics.data-an": "Physics - Data Analysis, Statistics and Probability",
    "physics.flu-dyn": "Physics - Fluid Dynamics",
    "physics.gen-ph": "Physics - General Physics",
    "physics.geo-ph": "Physics - Geophysics",
    "physics.hist-ph": "Physics - History and Philosophy of Physics",
    "physics.ins-det": "Physics - Instrumentation and Detectors",
    "physics.med-ph": "Physics - Medical Physics",
    "physics.optics": "Physics - Optics",
    "physics.plasm-ph": "Physics - Plasma Physics",
    "physics.pop-ph": "Physics - Popular Physics",
    "physics.soc-ph": "Physics - Physics and Society",
    "physics.space-ph": "Physics - Space Physics",
    
    # Mathematics
    "math": "Mathematics",
    "math.AG": "Mathematics - Algebraic Geometry",
    "math.AT": "Mathematics - Algebraic Topology",
    "math.AP": "Mathematics - Analysis of PDEs",
    "math.CT": "Mathematics - Category Theory",
    "math.CA": "Mathematics - Classical Analysis and ODEs",
    "math.CO": "Mathematics - Combinatorics",
    "math.AC": "Mathematics - Commutative Algebra",
    "math.CV": "Mathematics - Complex Variables",
    "math.DG": "Mathematics - Differential Geometry",
    "math.DS": "Mathematics - Dynamical Systems",
    "math.FA": "Mathematics - Functional Analysis",
    "math.GM": "Mathematics - General Mathematics",
    "math.GN": "Mathematics - General Topology",
    "math.GT": "Mathematics - Geometric Topology",
    "math.GR": "Mathematics - Group Theory",
    "math.HO": "Mathematics - History and Overview",
    "math.IT": "Mathematics - Information Theory",
    "math.KT": "Mathematics - K-Theory and Homology",
    "math.LO": "Mathematics - Logic",
    "math.MP": "Mathematics - Mathematical Physics",
    "math.MG": "Mathematics - Metric Geometry",
    "math.NT": "Mathematics - Number Theory",
    "math.NA": "Mathematics - Numerical Analysis",
    "math.OA": "Mathematics - Operator Algebras",
    "math.OC": "Mathematics - Optimization and Control",
    "math.PR": "Mathematics - Probability",
    "math.QA": "Mathematics - Quantum Algebra",
    "math.RT": "Mathematics - Representation Theory",
    "math.RA": "Mathematics - Rings and Algebras",
    "math.SP": "Mathematics - Spectral Theory",
    "math.ST": "Mathematics - Statistics Theory",
    "math.SG": "Mathematics - Symplectic Geometry",
    
    # Computer Science
    "cs": "Computer Science",
    "cs.AI": "Computer Science - Artificial Intelligence",
    "cs.AR": "Computer Science - Hardware Architecture",
    "cs.CC": "Computer Science - Computational Complexity",
    "cs.CE": "Computer Science - Computational Engineering, Finance, and Science",
    "cs.CG": "Computer Science - Computational Geometry",
    "cs.CL": "Computer Science - Computation and Language",
    "cs.CR": "Computer Science - Cryptography and Security",
    "cs.CV": "Computer Science - Computer Vision and Pattern Recognition",
    "cs.CY": "Computer Science - Computers and Society",
    "cs.DB": "Computer Science - Databases",
    "cs.DC": "Computer Science - Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Computer Science - Digital Libraries",
    "cs.DM": "Computer Science - Discrete Mathematics",
    "cs.DS": "Computer Science - Data Structures and Algorithms",
    "cs.ET": "Computer Science - Emerging Technologies",
    "cs.FL": "Computer Science - Formal Languages and Automata Theory",
    "cs.GL": "Computer Science - General Literature",
    "cs.GR": "Computer Science - Graphics",
    "cs.GT": "Computer Science - Computer Science and Game Theory",
    "cs.HC": "Computer Science - Human-Computer Interaction",
    "cs.IR": "Computer Science - Information Retrieval",
    "cs.IT": "Computer Science - Information Theory",
    "cs.LG": "Computer Science - Machine Learning",
    "cs.LO": "Computer Science - Logic in Computer Science",
    "cs.MA": "Computer Science - Multiagent Systems",
    "cs.MM": "Computer Science - Multimedia",
    "cs.MS": "Computer Science - Mathematical Software",
    "cs.NA": "Computer Science - Numerical Analysis",
    "cs.NE": "Computer Science - Neural and Evolutionary Computing",
    "cs.NI": "Computer Science - Networking and Internet Architecture",
    "cs.OH": "Computer Science - Other Computer Science",
    "cs.OS": "Computer Science - Operating Systems",
    "cs.PF": "Computer Science - Performance",
    "cs.PL": "Computer Science - Programming Languages",
    "cs.RO": "Computer Science - Robotics",
    "cs.SC": "Computer Science - Symbolic Computation",
    "cs.SD": "Computer Science - Sound",
    "cs.SE": "Computer Science - Software Engineering",
    "cs.SI": "Computer Science - Social and Information Networks",
    "cs.SY": "Computer Science - Systems and Control",
    
    # Quantitative Biology
    "q-bio": "Quantitative Biology",
    "q-bio.BM": "Quantitative Biology - Biomolecules",
    "q-bio.CB": "Quantitative Biology - Cell Behavior",
    "q-bio.GN": "Quantitative Biology - Genomics",
    "q-bio.MN": "Quantitative Biology - Molecular Networks",
    "q-bio.NC": "Quantitative Biology - Neurons and Cognition",
    "q-bio.OT": "Quantitative Biology - Other Quantitative Biology",
    "q-bio.PE": "Quantitative Biology - Populations and Evolution",
    "q-bio.QM": "Quantitative Biology - Quantitative Methods",
    "q-bio.SC": "Quantitative Biology - Subcellular Processes",
    "q-bio.TO": "Quantitative Biology - Tissues and Organs",
    
    # Quantitative Finance
    "q-fin": "Quantitative Finance",
    "q-fin.CP": "Quantitative Finance - Computational Finance",
    "q-fin.EC": "Quantitative Finance - Economics",
    "q-fin.GN": "Quantitative Finance - General Finance",
    "q-fin.MF": "Quantitative Finance - Mathematical Finance",
    "q-fin.PM": "Quantitative Finance - Portfolio Management",
    "q-fin.PR": "Quantitative Finance - Pricing of Securities",
    "q-fin.RM": "Quantitative Finance - Risk Management",
    "q-fin.ST": "Quantitative Finance - Statistical Finance",
    "q-fin.TR": "Quantitative Finance - Trading and Market Microstructure",
    
    # Statistics
    "stat": "Statistics",
    "stat.AP": "Statistics - Applications",
    "stat.CO": "Statistics - Computation",
    "stat.ME": "Statistics - Methodology",
    "stat.ML": "Statistics - Machine Learning",
    "stat.OT": "Statistics - Other Statistics",
    "stat.TH": "Statistics - Statistics Theory",
    
    # Electrical Engineering and Systems Science
    "eess": "Electrical Engineering and Systems Science",
    "eess.AS": "Electrical Engineering and Systems Science - Audio and Speech Processing",
    "eess.IV": "Electrical Engineering and Systems Science - Image and Video Processing",
    "eess.SP": "Electrical Engineering and Systems Science - Signal Processing",
    "eess.SY": "Electrical Engineering and Systems Science - Systems and Control",
    
    # Economics
    "econ": "Economics",
    "econ.EM": "Economics - Econometrics",
    "econ.GN": "Economics - General Economics",
    "econ.TH": "Economics - Theoretical Economics"
}

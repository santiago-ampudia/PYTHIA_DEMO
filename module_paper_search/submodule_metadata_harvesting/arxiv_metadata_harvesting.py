import os
import sqlite3
import requests
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timedelta
import logging
import calendar
import argparse
import sys
import re

# Import parameters from the parameters file
from module_paper_search.submodule_metadata_harvesting.arxiv_metadata_harvesting_parameters import (
    ARXIV_OAI_URL,
    MAIN_DIR,
    DATABASES_DIR,
    DB_PATH,
    BATCH_SIZE,
    MAX_RECORDS,
    WAIT_TIME,
    MAX_RETRIES,
    USE_WEEKLY_CHUNKS,
    START_DATE,
    END_DATE,
    NAMESPACES,
    CATEGORY_MAPPING,
    CATEGORIES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('arxiv_harvester')

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Namespaces for OAI-PMH XML parsing
# NAMESPACES = {
#     'oai': 'http://www.openarchives.org/OAI/2.0/',
#     'arxiv': 'http://arxiv.org/OAI/arXiv/',
#     'dc': 'http://purl.org/dc/elements/1.1/',
#     'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
# }

def fetch_oai_records(resumption_token=None, metadata_prefix="oai_dc", set_spec=None, from_date=None, until_date=None):
    """
    Fetch records using OAI-PMH protocol with retry logic
    
    Args:
        resumption_token: Token for continuing a previous request
        metadata_prefix: Format of metadata (oai_dc is more reliable)
        set_spec: Optional set to harvest (e.g., "physics:hep-th")
        from_date: Start date for harvesting (YYYY-MM-DD format)
        until_date: End date for harvesting (YYYY-MM-DD format)
        
    Returns:
        XML response as text
    """
    params = {
        "verb": "ListRecords"
    }
    
    if resumption_token:
        params["resumptionToken"] = resumption_token
    else:
        params["metadataPrefix"] = metadata_prefix
        
        # Use date ranges to limit result size
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date
            
        if set_spec:
            params["set"] = set_spec
    
    logger.info(f"Fetching OAI-PMH records with params: {params}")
    
    # Add delay to avoid overwhelming the API
    time.sleep(WAIT_TIME)
    
    # Implement retry logic
    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            response = requests.get(ARXIV_OAI_URL, params=params, timeout=60)  # Increased timeout to 60 seconds
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
        
        # Exponential backoff
        retry_count += 1
        if retry_count <= MAX_RETRIES:
            wait_time = WAIT_TIME * (2 ** (retry_count - 1))
            logger.warning(f"Retrying in {wait_time} seconds (attempt {retry_count}/{MAX_RETRIES})...")
            time.sleep(wait_time)
    
    raise Exception(f"Failed to fetch OAI records after {MAX_RETRIES} retries")


def safe_get_text(element, xpath, namespaces=None):
    """Safely extract text from an XML element"""
    if element is None:
        return ""
    
    found = element.find(xpath, namespaces) if namespaces else element.find(xpath)
    if found is not None and found.text:
        return found.text.strip()
    return ""


def parse_oai_response(xml_data):
    """
    Parse OAI-PMH response to extract metadata
    
    Returns:
        tuple: (list of papers, resumption_token or None)
    """
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {e}")
        logger.debug(f"Problematic XML: {xml_data[:500]}...")  # Log first 500 chars
        return [], None
    
    papers = []
    
    # Check for error response
    error = root.find('.//oai:error', NAMESPACES)
    if error is not None:
        error_code = error.attrib.get('code', 'unknown')
        error_msg = error.text if error.text else "Unknown error"
        logger.error(f"OAI-PMH error: {error_code} - {error_msg}")
        # If it's a no records match error, just return empty
        if error_code == 'noRecordsMatch':
            return [], None
        # For other errors, raise exception
        raise Exception(f"OAI-PMH error: {error_code} - {error_msg}")
    
    # Find all records
    records = root.findall('.//oai:record', NAMESPACES)
    logger.info(f"Found {len(records)} records in response")
    
    for record in records:
        try:
            # Extract header info - handle potential deleted records
            header = record.find('.//oai:header', NAMESPACES)
            if header is None:
                continue
                
            # Skip deleted records
            if 'status' in header.attrib and header.attrib['status'] == 'deleted':
                continue
                
            identifier_elem = header.find('.//oai:identifier', NAMESPACES)
            if identifier_elem is None or not identifier_elem.text:
                continue
                
            identifier = identifier_elem.text
            # Extract arxiv ID from identifier (format varies)
            if 'arxiv.org' in identifier.lower():
                arxiv_id = identifier.split(':')[-1]
            else:
                # Try to extract ID from other formats
                parts = identifier.split('/')
                arxiv_id = parts[-1] if parts else identifier
            
            # Extract datestamp from header (when the record was added/modified in the repository)
            datestamp = ""
            datestamp_elem = header.find('.//oai:datestamp', NAMESPACES)
            if datestamp_elem is not None and datestamp_elem.text:
                datestamp = datestamp_elem.text.strip()
            
            # Extract metadata - check if metadata section exists
            metadata = record.find('.//oai:metadata', NAMESPACES)
            if metadata is None:
                continue
            
            # Initialize with defaults
            title = "No Title"
            abstract = ""
            categories = ""
            created = ""
            updated = ""
            authors = ""
            comments = ""
            doi = ""
            
            # Try Dublin Core format first
            dc_container = metadata.find('./oai_dc:dc', NAMESPACES)
            
            if dc_container is not None:
                # Process Dublin Core elements
                title_elem = dc_container.find('./dc:title', NAMESPACES)
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.strip()
                
                # Get all descriptions (abstract is usually in description)
                desc_elems = dc_container.findall('./dc:description', NAMESPACES)
                
                # First, look for a dedicated comments description
                for desc in desc_elems:
                    desc_text = desc.text if desc.text else ""
                    # Check if this is a dedicated comments element
                    if desc_text.startswith("Comment:") or desc_text.startswith("Comments:"):
                        comments = desc_text.replace("Comment:", "").replace("Comments:", "").strip()
                        break
                
                # Then process the main abstract and other descriptions
                for desc in desc_elems:
                    desc_text = desc.text if desc.text else ""
                    
                    # Skip if this is just a comment element (already processed)
                    if desc_text.startswith("Comment:") or desc_text.startswith("Comments:"):
                        continue
                    
                    # Check if this description contains metadata
                    if "Comments:" in desc_text or "DOI:" in desc_text:
                        # Extract comments if not already found
                        if not comments:
                            comments_match = re.search(r"(?:^|\n)(?:Comments?|Comm\.?|Note):\s*([^\n]+)", desc_text, re.IGNORECASE)
                            if comments_match:
                                comments = comments_match.group(1).strip()
                                # Remove "Comments:" prefix if it was accidentally captured
                                comments = re.sub(r'^(?:Comments?|Comm\.?|Note):\s*', '', comments, flags=re.IGNORECASE)
                        
                        # Extract DOI
                        doi_match = re.search(r"(?:^|\n)?(?:DOI|doi):\s*(10\.[0-9]+\/[^\s\n]+)", desc_text)
                        if doi_match:
                            doi = doi_match.group(1).strip()
                    
                    # Use the longest text as the abstract
                    if desc_text and len(desc_text) > len(abstract):
                        # Remove metadata lines from abstract
                        clean_abstract = re.sub(r"(?:^|\n)(?:Comments?|DOI):[^\n]+", "", desc_text, flags=re.IGNORECASE)
                        abstract = clean_abstract.strip()
                
                # Get all creators/authors
                creator_elems = dc_container.findall('./dc:creator', NAMESPACES)
                if creator_elems:
                    authors = ','.join([c.text.strip() for c in creator_elems if c.text])
                
                # Check for DOI in identifier fields
                identifier_elems = dc_container.findall('./dc:identifier', NAMESPACES)
                for identifier in identifier_elems:
                    if identifier.text and identifier.text.strip().startswith('doi:'):
                        doi = identifier.text.strip().replace('doi:', '').strip()
                    elif identifier.text and 'doi.org' in identifier.text:
                        doi_match = re.search(r'doi\.org/(10\.[0-9]+/[^\s]+)', identifier.text)
                        if doi_match:
                            doi = doi_match.group(1).strip()
                
                # Get all subjects (categories) and convert to short codes
                subject_elems = dc_container.findall('./dc:subject', NAMESPACES)
                if subject_elems:
                    # Convert full category names to short codes where possible
                    category_codes = []
                    for s in subject_elems:
                        if not s.text:
                            continue
                        category_text = s.text.strip()
                        # Use the short code if available, otherwise keep the full name
                        category_code = CATEGORY_MAPPING.get(category_text, category_text)
                        category_codes.append(category_code)
                    categories = ','.join(category_codes)
                
                # Get dates
                date_elems = dc_container.findall('./dc:date', NAMESPACES)
                if date_elems:
                    # Sort dates to get earliest for created and latest for updated
                    date_texts = [d.text for d in date_elems if d.text]
                    if date_texts:
                        date_texts.sort()
                        created = date_texts[0]
                        updated = date_texts[-1]
            else:
                # Try to extract from any available elements as fallback
                for elem in metadata.findall('.//*', NAMESPACES):
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    text = elem.text.strip() if elem.text else ""
                    
                    if not text:
                        continue
                    
                    if tag.lower() in ('title', 'arttitle') and title == "No Title":
                        title = text
                    elif tag.lower() in ('description', 'abstract', 'summary') and not abstract:
                        abstract = text
                    elif tag.lower() in ('subject', 'category', 'categories'):
                        if categories:
                            categories += ',' + text
                        else:
                            categories = text
                    elif tag.lower() in ('date', 'created', 'published') and not created:
                        created = text
                    elif tag.lower() in ('updated', 'modified') and not updated:
                        updated = text
                    elif tag.lower() in ('author', 'creator'):
                        if authors:
                            authors += ',' + text
                        else:
                            authors = text
                    elif tag.lower() in ('comment'):
                        comments = text
                    elif tag.lower() in ('doi'):
                        doi = text
            
            # If we still don't have dates, use datestamp from header
            if not created:
                created = datestamp
                updated = datestamp
            
            # Ensure updated is not empty if created exists
            if created and not updated:
                updated = created
            
            paper = {
                'arxiv_id': arxiv_id,
                'title': title,
                'summary': abstract,
                'published': created,
                'updated': updated,
                'categories': categories,
                'datestamp': datestamp,
                'authors': authors,
                'comments': comments,
                'doi': doi
            }
            papers.append(paper)
            
        except Exception as e:
            logger.warning(f"Error parsing record: {e}")
            continue
    
    # Check for resumption token
    resumption_token_elem = root.find('.//oai:resumptionToken', NAMESPACES)
    resumption_token = None
    if resumption_token_elem is not None and resumption_token_elem.text:
        resumption_token = resumption_token_elem.text.strip()
        
    return papers, resumption_token


def create_db(db_path=DB_PATH):
    """Create SQLite database for storing arXiv metadata"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create papers table with comprehensive schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            published TEXT,
            updated TEXT,
            categories TEXT,
            datestamp TEXT,
            authors TEXT,
            comments TEXT,
            doi TEXT
        )
    ''')
    
    # Add indexes for faster searching
    c.execute('CREATE INDEX IF NOT EXISTS idx_categories ON papers(categories)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_published ON papers(published)')
    
    # Create a metadata table to track harvesting progress
    c.execute('''
        CREATE TABLE IF NOT EXISTS harvesting_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


def insert_papers(papers, db_path=DB_PATH):
    """Insert papers into SQLite database"""
    if not papers:
        logger.info("No papers to insert.")
        return 0
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    inserted = 0
    for paper in papers:
        try:
            c.execute('''
                INSERT OR REPLACE INTO papers (arxiv_id, title, summary, published, updated, categories, datestamp, authors, comments, doi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper['arxiv_id'],
                paper['title'],
                paper['summary'],
                paper['published'],
                paper['updated'],
                paper['categories'],
                paper['datestamp'],
                paper['authors'],
                paper['comments'],
                paper['doi']
            ))
            inserted += 1
        except sqlite3.Error as e:
            logger.warning(f"Error inserting paper {paper.get('arxiv_id', 'unknown')}: {e}")
    
    # Update last harvesting timestamp
    c.execute('''
        INSERT OR REPLACE INTO harvesting_metadata (key, value)
        VALUES (?, ?)
    ''', ('last_harvest', datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    return inserted


def get_db_count(db_path=DB_PATH):
    """Get count of papers in database"""
    if not os.path.exists(db_path):
        logger.info(f"Database file does not exist at: {db_path}")
        return 0
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        # Check if papers table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not c.fetchone():
            logger.info("Papers table does not exist in database yet")
            return 0
            
        c.execute('SELECT COUNT(*) FROM papers')
        count = c.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"Error getting database count: {e}")
        return 0
    finally:
        conn.close()


def save_harvesting_state(from_date, until_date, resumption_token=None, db_path=DB_PATH):
    """
    Save harvesting state to database for potential resuming later.
    
    Args:
        from_date: Start date of current harvesting chunk
        until_date: End date of current harvesting chunk
        resumption_token: OAI-PMH resumption token to continue from
        db_path: Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS harvesting_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Save current state
    if from_date:
        c.execute("INSERT OR REPLACE INTO harvesting_metadata VALUES (?, ?)", 
                 ("from_date", from_date))
    
    if until_date:
        c.execute("INSERT OR REPLACE INTO harvesting_metadata VALUES (?, ?)", 
                 ("until_date", until_date))
    
    # Save or clear resumption token
    if resumption_token:
        c.execute("INSERT OR REPLACE INTO harvesting_metadata VALUES (?, ?)", 
                 ("resumption_token", resumption_token))
        logger.info(f"Saved resumption token: {resumption_token}")
    else:
        # If no resumption token, we've completed this chunk, so clear it
        c.execute("DELETE FROM harvesting_metadata WHERE key = 'resumption_token'")
    
    # Save current timestamp
    c.execute("INSERT OR REPLACE INTO harvesting_metadata VALUES (?, ?)", 
             ("last_updated", datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def get_harvesting_state(db_path=DB_PATH):
    """Get saved harvesting state from database"""
    if not os.path.exists(db_path):
        return None, None, None
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get from_date
    c.execute('SELECT value FROM harvesting_metadata WHERE key = ?', ('from_date',))
    from_date_result = c.fetchone()
    from_date = from_date_result[0] if from_date_result else None
    
    # Get until_date
    c.execute('SELECT value FROM harvesting_metadata WHERE key = ?', ('until_date',))
    until_date_result = c.fetchone()
    until_date = until_date_result[0] if until_date_result else None
    
    # Get resumption token
    c.execute('SELECT value FROM harvesting_metadata WHERE key = ?', ('resumption_token',))
    token_result = c.fetchone()
    resumption_token = token_result[0] if token_result else None
    
    conn.close()
    
    return from_date, until_date, resumption_token


def generate_date_chunks(start_date_str=None, end_date_str=None):
    """
    Generate date chunks for harvesting.
    
    Args:
        start_date_str (str): Start date in YYYY-MM-DD format
        end_date_str (str): End date in YYYY-MM-DD format
        
    Returns:
        list: List of (start_date, end_date) tuples for each chunk
    """
    # Use configured start date if none provided
    if not start_date_str:
        start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    else:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        
    # Use configured end date if none provided
    if not end_date_str:
        end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    else:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Generate chunks
    chunks = []
    chunk_start = start_date
    
    while chunk_start <= end_date:
        # Get the end of the week (6 days after start)
        chunk_end = chunk_start + timedelta(days=6)
        
        # Ensure we don't go past the end date
        if chunk_end > end_date:
            chunk_end = end_date
        
        chunks.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        
        # Move to the next week
        chunk_start = chunk_end + timedelta(days=1)
    
    return chunks


def run_harvesting(start_date=None, end_date=None, set_spec=None):
    """
    Run the harvesting process for the specified date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (default: START_DATE)
        end_date: End date in YYYY-MM-DD format (default: END_DATE)
        set_spec: OAI-PMH set specification (default: None, will use CATEGORIES)
    """
    # Use default values if not provided
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    
    # Start timing
    start_time = time.time()
    
    # Check if DB exists
    db_exists = os.path.exists(DB_PATH)
    
    if not db_exists:
        logger.info(f"Creating new database at {DB_PATH}")
        create_db()
    else:
        logger.info(f"Using existing database at {DB_PATH}")
        # Ensure the harvesting_metadata table exists even for existing DBs
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS harvesting_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    logger.info(f"Starting arXiv metadata harvesting using OAI-PMH...")
    
    # Get current paper count
    total_papers = get_db_count()
    logger.info(f"Current paper count in database: {total_papers}")
    
    # Check if we have a saved state to resume from
    from_date, until_date, resumption_token = get_harvesting_state()
    
    # If we have a resumption token, continue from where we left off
    if resumption_token:
        logger.info(f"Resuming harvesting from saved token for date range {from_date} to {until_date}")
        process_date_chunk(from_date, until_date, resumption_token)
        
        # Update from_date for next chunk
        if until_date:
            from_date = (datetime.strptime(until_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    elif from_date and until_date:
        # We were in the middle of a date range but don't have a resumption token
        # This means we completed that range, so move to the next one
        logger.info(f"Completed previous date range {from_date} to {until_date}")
        from_date = (datetime.strptime(until_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Use the provided start_date if available, otherwise use from_date or default
    if start_date:
        effective_start_date = start_date
    elif from_date:
        effective_start_date = from_date
    else:
        effective_start_date = START_DATE
    
    # Use the provided end_date if available, otherwise use default
    effective_end_date = end_date if end_date else END_DATE
    
    # Generate date chunks starting from where we left off
    date_chunks = generate_date_chunks(effective_start_date, effective_end_date)
    
    # Get the categories to process
    if set_spec:
        # If a specific set_spec is provided, use it
        categories = [set_spec]
    else:
        # Otherwise use the categories from the parameters file
        categories = CATEGORIES
    
    # Process each date chunk
    for chunk_from, chunk_until in date_chunks:
        logger.info(f"\nProcessing date chunk: {chunk_from} to {chunk_until}")
        
        # Save current chunk state
        save_harvesting_state(chunk_from, chunk_until)
        
        # Process this chunk for each category
        for category in categories:
            logger.info(f"Processing category: {category}")
            
            # Process this chunk
            process_date_chunk(chunk_from, chunk_until, set_spec=category)
            
            # Check if we've reached our limit
            total_papers = get_db_count()
            if MAX_RECORDS and total_papers >= MAX_RECORDS:
                logger.info(f"Reached maximum records limit ({MAX_RECORDS}). Stopping.")
                break
        
        # If we've reached our limit, break out of the date chunks loop too
        if MAX_RECORDS and total_papers >= MAX_RECORDS:
            break
    
    # Final count and timing
    final_count = get_db_count()
    papers_added = final_count - total_papers
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nHarvesting session complete.")
    logger.info(f"Total papers in database: {final_count}")
    logger.info(f"Papers added in this session: {papers_added}")
    logger.info(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Database stored at: {DB_PATH}")
    
    return final_count


def process_date_chunk(from_date, until_date, resumption_token=None, set_spec=None):
    """Process a single date chunk, with optional resumption token"""
    batch_num = 1
    
    try:
        while True:
            # Log more detailed information about the current state
            if resumption_token:
                logger.info(f"Processing batch #{batch_num} for {from_date} to {until_date} (resuming with token)")
            else:
                logger.info(f"Processing batch #{batch_num} for {from_date} to {until_date}...")
            
            # Save state before making the request, so we can resume if it fails
            save_harvesting_state(from_date, until_date, resumption_token)
            
            # Fetch records
            xml_data = fetch_oai_records(
                resumption_token=resumption_token,
                from_date=from_date if not resumption_token else None,
                until_date=until_date if not resumption_token else None,
                set_spec=set_spec
            )
            
            # Parse response
            papers, resumption_token = parse_oai_response(xml_data)
            
            # Insert into database
            inserted = insert_papers(papers)
            
            logger.info(f"Inserted {inserted} papers from {from_date} to {until_date}")
            
            # Update the last update file with the current chunk's end date
            # This ensures we have a record of progress even if interrupted
            try:
                from arxiv_metadata_harvesting_scheduler import save_update_date
                save_update_date(chunk_end_date=until_date)
            except ImportError:
                logger.warning("Could not import save_update_date function. Last update file not updated.")
            
            # Check if we should continue
            if not resumption_token:
                logger.info(f"Completed harvesting for date range {from_date} to {until_date}")
                # Clear the resumption token in the database to mark this chunk as complete
                save_harvesting_state(from_date, until_date, None)
                break
                
            # Save state with resumption token for the next batch
            save_harvesting_state(from_date, until_date, resumption_token)
            
            # Check if we've reached our limit
            total_papers = get_db_count()
            if MAX_RECORDS and total_papers >= MAX_RECORDS:
                logger.info(f"Reached maximum records limit ({MAX_RECORDS}). Stopping.")
                break
                
            batch_num += 1
            
            # Add delay between requests to avoid rate limiting
            logger.info(f"Waiting {WAIT_TIME} seconds before next request...")
            time.sleep(WAIT_TIME)
            
    except KeyboardInterrupt:
        logger.info("\nHarvesting interrupted by user.")
        logger.info(f"Resumption token saved. You can resume later from paper {batch_num * BATCH_SIZE}.")
        # The state is already saved before the fetch_oai_records call, so we're good
    except Exception as e:
        logger.error(f"Error during harvesting: {e}")
        logger.info("You can resume later using the saved state.")


def download_specific_paper(arxiv_id):
    """
    Download metadata for a specific paper by its arXiv ID.
    
    Args:
        arxiv_id (str): The arXiv ID of the paper to download
        
    Returns:
        bool: True if the paper was successfully downloaded and added to the database,
              False if it was already in the database or couldn't be found
    """
    logger.info(f"Attempting to download specific paper with arXiv ID: {arxiv_id}")
    
    # Check if the paper already exists in the database
    if paper_exists(arxiv_id):
        logger.info(f"Paper with arXiv ID {arxiv_id} already exists in the database")
        return False
    
    # Construct the OAI identifier
    oai_identifier = f"oai:arXiv.org:{arxiv_id}"
    logger.info(f"Fetching paper with OAI identifier: {oai_identifier}")
    
    # Fetch the paper using the GetRecord verb
    try:
        params = {
            'verb': 'GetRecord',
            'identifier': oai_identifier,
            'metadataPrefix': 'oai_dc'
        }
        
        response = requests.get(ARXIV_OAI_URL, params=params)
        response.raise_for_status()
        
        # Parse the response
        root = ET.fromstring(response.text)
        
        # Check for errors
        error = root.find('.//{http://www.openarchives.org/OAI/2.0/}error')
        if error is not None:
            logger.error(f"OAI-PMH error: {error.attrib.get('code')} - {error.text}")
            return False
        
        # Extract the record
        record = root.find('.//{http://www.openarchives.org/OAI/2.0/}record')
        if record is None:
            logger.error("No record found in response")
            return False
        
        # Parse the record
        papers, _ = parse_oai_response(response.text)
        
        if not papers:
            logger.error("Failed to parse paper metadata")
            return False
        
        # Insert the paper into the database
        inserted = insert_papers(papers)
        
        if inserted > 0:
            logger.info(f"Successfully downloaded and inserted paper with arXiv ID: {arxiv_id}")
            return True
        else:
            logger.warning(f"Paper with arXiv ID {arxiv_id} was found but not inserted (may already exist)")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching paper: {e}")
        return False
    except ET.ParseError as e:
        logger.error(f"Error parsing XML response: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def get_most_recent_paper_date(db_path=DB_PATH):
    """
    Get the date of the most recently added paper in the database.
    
    Returns:
        date object or None if no papers found
    """
    # Check if database file exists
    if not os.path.exists(db_path):
        logger.info(f"Database file does not exist at: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        # Check if papers table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not c.fetchone():
            logger.info("Papers table does not exist in database yet")
            return None
        
        # First try to get the most recent paper by date_updated
        c.execute("""
            SELECT date_updated FROM papers 
            ORDER BY date_updated DESC 
            LIMIT 1
        """)
        result = c.fetchone()
        
        if result and result[0]:
            try:
                # Try to parse the date
                date_str = result[0]
                # Handle different date formats
                if 'T' in date_str:
                    # ISO format with time
                    date_obj = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d").date()
                else:
                    # Just date
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                logger.info(f"Most recent paper date from database: {date_obj}")
                return date_obj
            except ValueError as e:
                logger.warning(f"Could not parse date from database: {date_str}, error: {e}")
                return None
        else:
            logger.info("No papers found in database")
            return None
    except Exception as e:
        logger.error(f"Error getting most recent paper date: {e}")
        return None
    finally:
        conn.close()


def paper_exists(arxiv_id, db_path=DB_PATH):
    """Check if a paper exists in the database"""
    if not os.path.exists(db_path):
        logger.info(f"Database file does not exist at: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        # Check if papers table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not c.fetchone():
            logger.info("Papers table does not exist in database yet")
            return False
        
        c.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        result = c.fetchone()
        return result is not None
    except Exception as e:
        logger.error(f"Error checking paper existence: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest arXiv metadata using OAI-PMH")
    parser.add_argument("--reset", action="store_true", help="Delete existing database and start fresh")
    parser.add_argument("--analyze", action="store_true", help="Analyze the database contents")
    parser.add_argument("--start-date", type=str, help="Start date for harvesting (YYYY-MM-DD format)")
    parser.add_argument("--end-date", type=str, help="End date for harvesting (YYYY-MM-DD format)")
    parser.add_argument("--download-paper", type=str, help="Download a specific paper by its arXiv ID")
    args = parser.parse_args()
    
    if args.reset:
        if os.path.exists(DB_PATH):
            logger.info(f"Deleting existing database at {DB_PATH}")
            os.remove(DB_PATH)
            logger.info("Database deleted. Starting fresh.")
    
    if args.analyze:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_count = cursor.fetchone()[0]
        print(f"Total papers in database: {total_count}")
        
        # Check for the specific paper mentioned
        cursor.execute("SELECT arxiv_id, title, published, updated, datestamp, categories, authors, comments, doi FROM papers WHERE arxiv_id = '2411.04526'")
        paper = cursor.fetchone()
        if paper:
            print(f"\nPaper 2411.04526 found:")
            print(f"  ID: {paper[0]}")
            print(f"  Title: {paper[1]}")
            print(f"  Published: {paper[2]}")
            print(f"  Updated: {paper[3]}")
            print(f"  Repository Datestamp: {paper[4]}")
            print(f"  Categories: {paper[5]}")
            print(f"  Authors: {paper[6]}")
            print(f"  Comments: {paper[7]}")
            print(f"  DOI: {paper[8]}")
        else:
            print("Paper 2411.04526 not found in database.")
        
        conn.close()
    elif args.download_paper:
        download_specific_paper(args.download_paper)
    else:
        # Run the harvesting process
        run_harvesting(args.start_date, args.end_date)

"""
submodule_metadata_harvesting/arxiv_metadata_harvesting_scheduler.py

This module schedules and manages the execution of the arxiv metadata harvesting process.
It determines when to run the harvesting based on configured parameters and handles the
scheduling of regular updates.
"""

import logging
import time
import schedule
from datetime import datetime, timedelta
import os

# Import parameters from the parameters file
from module_paper_search.submodule_metadata_harvesting.arxiv_metadata_harvesting_parameters import (
    UPDATE_HOUR, 
    UPDATE_MINUTE,
    START_DATE,
    END_DATE,
    DB_PATH,
    DATABASES_DIR,
    ON_DEMAND_ARXIV_ID
)

# Import the actual harvesting module
from module_paper_search.submodule_metadata_harvesting.arxiv_metadata_harvesting import run_harvesting, download_specific_paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('arxiv_harvester_scheduler')

def ensure_database_directory():
    """Ensure the database directory exists"""
    os.makedirs(DATABASES_DIR, exist_ok=True)
    logger.info(f"Ensured database directory exists at: {DATABASES_DIR}")

def get_last_update_date():
    """
    Get the date of the last successful update from a metadata file.
    If no file exists, return None to trigger a full update.
    """
    # Extract database name from DB_PATH
    db_name = os.path.basename(DB_PATH).split('.')[0]
    metadata_file = os.path.join(DATABASES_DIR, f"last_update_{db_name}.txt")
    
    if not os.path.exists(metadata_file):
        logger.info(f"No last update file found at: {metadata_file}")
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            last_update = f.read().strip()
            logger.info(f"Last update date found: {last_update}")
            return datetime.strptime(last_update, "%Y-%m-%d").date()
    except (ValueError, IOError) as e:
        logger.error(f"Error reading last update date: {e}")
        return None

def save_update_date(date=None, chunk_end_date=None):
    """
    Save the current date as the last update date.
    
    Args:
        date: The date to save (defaults to today)
        chunk_end_date: The end date of the current chunk being processed
    """
    # Extract database name from DB_PATH
    db_name = os.path.basename(DB_PATH).split('.')[0]
    metadata_file = os.path.join(DATABASES_DIR, f"last_update_{db_name}.txt")
    
    # If chunk_end_date is provided, use it as the last update date
    # This allows for continuous updating as chunks are processed
    update_date = None
    if chunk_end_date:
        update_date = datetime.strptime(chunk_end_date, "%Y-%m-%d").date()
    else:
        update_date = date or datetime.now().date()
    
    try:
        with open(metadata_file, 'w') as f:
            f.write(update_date.strftime("%Y-%m-%d"))
        logger.info(f"Saved last update date: {update_date}")
    except IOError as e:
        logger.error(f"Error saving update date: {e}")

def scheduled_update():
    """
    Function to be called by the scheduler to perform the update.
    """
    logger.info("Running scheduled arXiv metadata update...")
    
    # Get last update date from file
    last_update = get_last_update_date()
    
    # Import functions to check database
    from arxiv_metadata_harvesting import get_most_recent_paper_date, create_db, DB_PATH
    
    # Ensure database exists before trying to query it
    if not os.path.exists(DB_PATH):
        logger.info(f"Database doesn't exist yet. Creating it at {DB_PATH}")
        create_db()
        most_recent_paper_date = None
    else:
        # Get most recent paper date
        most_recent_paper_date = get_most_recent_paper_date()
    
    # Determine the effective start date
    effective_start_date = None
    
    # Convert dates to comparable objects
    start_date_obj = None
    if START_DATE:
        start_date_obj = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    
    day_after_last_update = None
    if last_update:
        day_after_last_update = last_update + timedelta(days=1)
    
    day_after_most_recent_paper = None
    if most_recent_paper_date:
        day_after_most_recent_paper = most_recent_paper_date + timedelta(days=1)
    
    # Find the most recent date among all sources
    candidate_dates = []
    if start_date_obj:
        candidate_dates.append(("START_DATE", start_date_obj))
    if day_after_last_update:
        candidate_dates.append(("last update file", day_after_last_update))
    if day_after_most_recent_paper:
        candidate_dates.append(("most recent paper", day_after_most_recent_paper))
    
    if not candidate_dates:
        logger.info("No valid start date found. No update needed.")
        return
    
    # Find the most recent date
    most_recent_source, most_recent_date = max(candidate_dates, key=lambda x: x[1])
    effective_start_date = most_recent_date.strftime("%Y-%m-%d")
    
    logger.info(f"Using date from {most_recent_source} ({effective_start_date}) as it is the most recent")
    
    # Use the configured END_DATE from parameters instead of today's date
    end_date = END_DATE
    logger.info(f"Using configured END_DATE: {end_date}")
    
    logger.info(f"Updating arXiv metadata from {effective_start_date} to {end_date}")
    run_harvesting(effective_start_date, end_date)
    
    # Save the END_DATE as the last update date, not today's date
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
    save_update_date(date=end_date_obj)
    logger.info(f"Saved {end_date} as the last update date")
    
    logger.info("Scheduled update completed")

def setup_daily_schedule():
    """
    Set up the daily schedule for metadata harvesting.
    """
    # If UPDATE_HOUR or UPDATE_MINUTE is None, don't set up scheduling
    if UPDATE_HOUR is None or UPDATE_MINUTE is None:
        logger.info("Automatic updates are disabled (UPDATE_HOUR or UPDATE_MINUTE is None)")
        return
    
    schedule_time = f"{UPDATE_HOUR:02d}:{UPDATE_MINUTE:02d}"
    logger.info(f"Setting up daily updates at {schedule_time}")
    
    schedule.every().day.at(schedule_time).do(scheduled_update)
    logger.info(f"Daily update scheduled for {schedule_time}")

def is_update_needed():
    """
    Check if an update is needed based on the last update date.
    Returns True if there was no previous update or if the last update
    was not today.
    """
    # If START_DATE is None, no update is needed unless forced
    if START_DATE is None:
        logger.info("START_DATE is None. No update needed unless forced.")
        return False
    
    last_update = get_last_update_date()
    today = datetime.now().date()
    
    if last_update is None:
        logger.info("No previous update found. Update is needed.")
        return True
    
    days_since_update = (today - last_update).days
    if days_since_update > 0:
        logger.info(f"Last update was {days_since_update} days ago. Update is needed.")
        return True
    
    logger.info(f"Database was already updated today ({last_update}). No update needed.")
    return False

def run_harvesting_scheduler(force_update=False, run_continuously=False, arxiv_id=None, start_date=None):
    """
    Run the harvesting scheduler.
    
    Args:
        force_update: Force an update regardless of the last update date
        run_continuously: Keep the scheduler running continuously
        arxiv_id: Specific arXiv ID to download (on-demand download)
        start_date: Override the START_DATE parameter with this value
    """
    # Ensure database directory exists
    os.makedirs(DATABASES_DIR, exist_ok=True)
    logger.info(f"Ensured database directory exists at: {DATABASES_DIR}")
    
    # Check if automatic updates are enabled
    if UPDATE_HOUR is None or UPDATE_MINUTE is None:
        logger.info("Automatic updates are disabled (UPDATE_HOUR or UPDATE_MINUTE is None)")
    else:
        logger.info(f"Automatic updates scheduled for {UPDATE_HOUR:02d}:{UPDATE_MINUTE:02d}")
    
    # Handle on-demand paper download
    if arxiv_id:
        logger.info(f"On-demand download requested for arXiv ID: {arxiv_id}")
        from arxiv_metadata_harvesting import download_specific_paper
        
        # Download the specific paper without updating the last_update date
        if download_specific_paper(arxiv_id):
            logger.info(f"Successfully downloaded paper with arXiv ID: {arxiv_id}")
        else:
            logger.error(f"Failed to download paper with arXiv ID: {arxiv_id}")
        
        # Return without updating the last_update date
        return
    
    # Check if an update is needed
    if not force_update and not is_update_needed():
        logger.info("No update needed at this time.")
        
        # Set up daily schedule if requested
        if run_continuously and UPDATE_HOUR is not None and UPDATE_MINUTE is not None:
            setup_daily_schedule()
            
            # Keep the scheduler running
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        return
    
    logger.info("Running immediate update...")
    scheduled_update()
    
    # Set up daily schedule if requested
    if run_continuously and UPDATE_HOUR is not None and UPDATE_MINUTE is not None:
        setup_daily_schedule()
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    logger.info("Scheduler setup complete. Not running continuously.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Schedule arXiv metadata harvesting")
    parser.add_argument("--force", action="store_true", help="Force immediate update")
    parser.add_argument("--continuous", action="store_true", help="Run scheduler continuously")
    parser.add_argument("--arxiv-id", type=str, help="Download a specific paper by its arXiv ID")
    args = parser.parse_args()
    
    run_harvesting_scheduler(force_update=args.force, run_continuously=args.continuous, arxiv_id=args.arxiv_id)

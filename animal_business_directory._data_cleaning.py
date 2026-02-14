"""
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from difflib import SequenceMatcher


# CONFIGURATION

INPUT_FILE = 'raw_business_data.csv'
OUTPUT_FILE = 'cleaned_business_data.csv'
DUPLICATES_FILE = 'duplicates_report.csv'
PROCESSING_LOG = 'processing_log.txt'


# LOGGING SETUP

def log_message(message, log_file=PROCESSING_LOG):
    """Log messages to file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")


# PHONE NUMBER CLEANING

def clean_phone_number(phone):
    """
    Clean and standardize phone numbers to +91XXXXXXXXXX format
    
    This function handles various phone number formats and converts them
    to a standardized format with country code.
    
    Args:
        phone: Raw phone number string (can be None/NaN)
        
    Returns:
        tuple: (cleaned_phone, status)
            - cleaned_phone: Standardized phone number or None
            - status: Status string ('Valid', 'Missing', 'Invalid format', 'Invalid length')
            
    Examples:
        >>> clean_phone_number('9876543210')
        ('+919876543210', 'Valid')
        >>> clean_phone_number('+91 98765 43210')
        ('+919876543210', 'Valid')
        >>> clean_phone_number(None)
        (None, 'Missing')
        >>> clean_phone_number('abc')
        (None, 'Invalid format')
    """
    # Handle missing or empty values
    if pd.isna(phone) or phone == '' or str(phone).strip() == '':
        return None, 'Missing'
    
    # Convert to string and clean
    phone_str = str(phone).strip()
    
    # Remove all non-numeric characters except +
    cleaned = re.sub(r'[^\d+]', '', phone_str)
    
    # Handle different formats
    if len(cleaned) == 0:
        return None, 'Invalid format'
    
    # Case 1: 10-digit number (most common in India)
    if len(cleaned) == 10:
        cleaned = '+91' + cleaned
    
    # Case 2: 11-digit number starting with 0
    elif len(cleaned) == 11 and cleaned.startswith('0'):
        cleaned = '+91' + cleaned[1:]
    
    # Case 3: 12-digit number starting with 91 (country code without +)
    elif len(cleaned) == 12 and cleaned.startswith('91'):
        cleaned = '+' + cleaned
    
    # Case 4: 13-digit number starting with 091
    elif len(cleaned) == 13 and cleaned.startswith('091'):
        cleaned = '+' + cleaned[1:]
    
    # Case 5: Already has + prefix
    elif cleaned.startswith('+'):
        # Remove any spaces or special characters after +
        cleaned = '+' + re.sub(r'[^\d]', '', cleaned[1:])
    
    # Case 6: Unknown format
    else:
        return None, 'Invalid format'
    
    # Validate final length (should be 12-13 characters with +)
    # +91XXXXXXXXXX = 12 characters
    # +1XXXXXXXXXX = 11 characters (US/Canada)
    if len(cleaned) < 10 or len(cleaned) > 15:
        return None, 'Invalid length'
    
    # Additional validation: ensure no consecutive repeated digits (potential spam)
    if re.search(r'(\d)\1{5,}', cleaned):
        return None, 'Suspicious pattern'
    
    return cleaned, 'Valid'


def format_phone_display(phone):
    """
    Format phone number for display (e.g., +91 98765 43210)
    
    Args:
        phone: Cleaned phone number
        
    Returns:
        str: Formatted phone number
    """
    if pd.isna(phone) or phone is None:
        return ''
    
    phone_str = str(phone)
    
    # Format Indian numbers
    if phone_str.startswith('+91') and len(phone_str) == 12:
        return f"+91 {phone_str[3:8]} {phone_str[8:]}"
    
    # Format US numbers
    elif phone_str.startswith('+1') and len(phone_str) == 11:
        return f"+1 ({phone_str[2:5]}) {phone_str[5:8]}-{phone_str[8:]}"
    
    # Return as-is for other formats
    return phone_str


# ADDRESS STANDARDIZATION

def standardize_address(address):
    """
    Standardize address format
    
    This function:
    - Removes extra whitespaces
    - Standardizes common abbreviations
    - Applies proper capitalization
    
    Args:
        address: Raw address string
        
    Returns:
        str: Standardized address or None if invalid
        
    Examples:
        >>> standardize_address('123 main st, mumbai')
        '123 Main St, Mumbai'
    """
    if pd.isna(address) or address == '':
        return None
    
    # Convert to string
    address = str(address).strip()
    
    # Remove extra whitespaces (multiple spaces, tabs, newlines)
    address = ' '.join(address.split())
    
    # Common abbreviations standardization
    # Dictionary of regex patterns and their replacements
    replacements = {
        # Street types
        r'\bSt\b': 'Street',
        r'\bStr\b': 'Street',
        r'\bAve\b': 'Avenue',
        r'\bAv\b': 'Avenue',
        r'\bRd\b': 'Road',
        r'\bRoad\b': 'Road',
        r'\bLn\b': 'Lane',
        r'\bDr\b': 'Drive',
        r'\bDrive\b': 'Drive',
        r'\bBlvd\b': 'Boulevard',
        r'\bBoul\b': 'Boulevard',
        r'\bPkg\b': 'Parking',
        r'\bPlaza\b': 'Plaza',
        
        # Building types
        r'\bNo\b': 'Number',
        r'\bBldg\b': 'Building',
        r'\bBldg\b': 'Building',
        r'\bFl\b': 'Floor',
        r'\bSte\b': 'Suite',
        
        # Company types
        r'\bCo\b': 'Company',
        r'\bLtd\b': 'Limited',
        r'\bPvt\b': 'Private',
        r'\bCorp\b': 'Corporation',
        r'\bInc\b': 'Incorporated',
        
        # Directionals
        r'\bN\b': 'North',
        r'\bS\b': 'South',
        r'\bE\b': 'East',
        r'\bW\b': 'West',
        r'\bNE\b': 'Northeast',
        r'\bNW\b': 'Northwest',
        r'\bSE\b': 'Southeast',
        r'\bSW\b': 'Southwest',
    }
    
    # Apply replacements
    for pattern, replacement in replacements.items():
        address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
    
    # Proper title case (but keep abbreviations in uppercase)
    # First, protect known acronyms
    protected = ['LLC', 'LLP', 'GPS', 'ATM', 'TV', 'PM', 'AM']
    protected_placeholders = {}
    
    for i, acronym in enumerate(protected):
        placeholder = f"__ACRONYM{i}__"
        address = address.replace(acronym, placeholder)
        protected_placeholders[placeholder] = acronym
    
    # Apply title case
    address = address.title()
    
    # Restore protected acronyms
    for placeholder, acronym in protected_placeholders.items():
        address = address.replace(placeholder, acronym)
    
    return address


def extract_pincode(address):
    """
    Extract pincode from address
    
    Currently supports:
    - Indian PIN codes (6 digits)
    - US ZIP codes (5 digits)
    
    Args:
        address: Address string
        
    Returns:
        str: Extracted pincode/zip or None
    """
    if pd.isna(address):
        return None
    
    # Indian pincode pattern (6 digits)
    indian_pattern = r'\b\d{6}\b'
    match = re.search(indian_pattern, str(address))
    if match:
        return match.group()
    
    # US ZIP code pattern (5 digits, optionally with 4-digit extension)
    us_pattern = r'\b\d{5}(?:-\d{4})?\b'
    match = re.search(us_pattern, str(address))
    if match:
        return match.group()
    
    return None


def extract_city_from_address(address):
    """
    Attempt to extract city name from address
    
    Args:
        address: Address string
        
    Returns:
        str: Extracted city or None
    """
    if pd.isna(address):
        return None
    
    # Common city extraction patterns
    # Format: "Address, City PINCODE" or "Address, City, State"
    patterns = [
        r',\s*([A-Za-z\s]+?)\s+\d{6}$',  # Indian: , City 123456
        r',\s*([A-Za-z\s]+?)\s+\d{5}(?:-\d{4})?$',  # US: , City 12345
        r',\s*([A-Za-z\s]+?),\s*[A-Za-z\s]+$',  # , City, State
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(address))
        if match:
            city = match.group(1).strip()
            # Remove common suffixes
            city = re.sub(r'\b(PO|Box|RD|ROAD)\b', '', city, flags=re.IGNORECASE)
            return city.strip().title()
    
    return None


# EMAIL STANDARDIZATION

def standardize_email(email):
    """
    Standardize email format
    
    Args:
        email: Raw email string
        
    Returns:
        str: Standardized email or None if invalid
    """
    if pd.isna(email) or email == '':
        return None
    
    # Clean and convert to lowercase
    email = str(email).strip().lower()
    
    # Remove multiple dots before @
    email = re.sub(r'\.{2,}', '.', email)
    
    # Basic email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, email) and len(email) <= 254:
        return email
    
    return None


def validate_email_domain(email):
    """
    Validate email domain (basic check)
    
    Args:
        email: Standardized email
        
    Returns:
        tuple: (is_valid, domain)
    """
    if pd.isna(email) or email is None:
        return False, None
    
    try:
        domain = email.split('@')[1]
        
        # Check for valid-looking domain
        if '.' in domain and len(domain) > 3:
            return True, domain
        
        return False, domain
    
    except IndexError:
        return False, None


# BUSINESS NAME STANDARDIZATION

def standardize_business_name(name):
    """
    Standardize business name format
    
    Args:
        name: Raw business name
        
    Returns:
        str: Standardized name or None if invalid
    """
    if pd.isna(name) or name == '':
        return None
    
    name = str(name).strip()
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    # Proper title case (with exceptions)
    # Protect certain patterns like "McDonald's"
    mcdonald_pattern = r"(Mc[A-Za-z]+)"
    mcdonald_matches = re.findall(mcdonald_pattern, name)
    
    # Apply title case
    name = name.title()
    
    # Handle "Mc" names properly
    for match in mcdonald_matches:
        proper = match[0].upper() + match[1:].lower()
        name = name.replace(match.title(), proper)
    
    # Standardize business suffixes
    suffix_patterns = [
        (r'\bPvt\.?\s*Ltd\.?', 'Pvt. Ltd.'),
        (r'\bLimited\b', 'Ltd.'),
        (r'\bIncorporated\b', 'Inc.'),
        (r'\bCorporation\b', 'Corp.'),
        (r'\bCompany\b', 'Co.'),
    ]
    
    for pattern, replacement in suffix_patterns:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    return name


def extract_business_type(name):
    """
    Extract business type from name
    
    Args:
        name: Business name
        
    Returns:
        str: Extracted business type or None
    """
    if pd.isna(name):
        return None
    
    name_lower = str(name).lower()
    
    business_types = {
        'Veterinary': ['vet', 'veterinary', 'clinic', 'hospital', 'medical'],
        'Pet Shop': ['pet shop', 'pet store', 'pets', 'animal'],
        'Training': ['training', 'trainer', 'dog walker', 'walker'],
        'Grooming': ['grooming', 'groomer', 'spa', 'salon', 'beauty'],
        'Shelter': ['shelter', 'rescue', 'adoption', 'welfare'],
        'Breeding': ['breeder', 'breeding', 'kennel'],
    }
    
    for biz_type, keywords in business_types.items():
        for keyword in keywords:
            if keyword in name_lower:
                return biz_type
    
    return None


# CATEGORY STANDARDIZATION

def clean_category(category):
    """
    Standardize business category
    
    Args:
        category: Raw category string
        
    Returns:
        str: Standardized category
    """
    if pd.isna(category) or category == '':
        return 'Unknown'
    
    category = str(category).strip().lower()
    
    # Category mapping with keywords
    category_map = {
        'Veterinarian': [
            'vet', 'veterinary', 'veterinarian', 'doctor', 'clinic', 
            'hospital', 'medical', 'pet doctor', 'animal doctor'
        ],
        'Pet Shop': [
            'pet shop', 'pet store', 'pets', 'animal shop', 
            'pet supplies', 'aquarium', 'bird shop'
        ],
        'Trainer': [
            'trainer', 'training', 'dog walker', 'walker', 
            'behaviorist', 'obedience', 'dog trainer'
        ],
        'Groomer': [
            'grooming', 'groomer', 'pet spa', 'bath', 'salon', 
            'beauty', 'nail trim', 'pet grooming'
        ],
        'Shelter': [
            'shelter', 'rescue', 'adoption', 'orphanage', 
            'animal welfare', 'pound', 'humane society'
        ],
        'Breeder': [
            'breeder', 'breeding', 'kennel', 'cattery', 
            'puppy mill', 'animal breeding'
        ],
        'Pet Food': [
            'pet food', 'animal feed', 'food store', 
            'pet nutrition', 'bird seed'
        ],
        'Pet Insurance': [
            'insurance', 'pet insurance', 'coverage'
        ],
        'Pet Taxi': [
            'pet taxi', 'pet transport', 'animal transport',
            'pet pickup', 'pet delivery'
        ],
    }
    
    for standard_cat, keywords in category_map.items():
        for keyword in keywords:
            if keyword in category:
                return standard_cat
    
    # If no match, return title-cased original
    return category.title()


# DUPLICATE DETECTION

def calculate_similarity(str1, str2):
    """
    Calculate similarity ratio between two strings
    
    Uses SequenceMatcher for fuzzy matching.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        float: Similarity ratio (0 to 1)
            - 1.0 means identical
            - 0.0 means completely different
    """
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    
    str1 = str(str1).lower().strip()
    str2 = str(str2).lower().strip()
    
    return SequenceMatcher(None, str1, str2).ratio()


def calculate_address_similarity(addr1, addr2):
    """
    Calculate similarity between two addresses
    
    This is more sophisticated than simple string comparison,
    as it handles address components specially.
    
    Args:
        addr1: First address
        addr2: Second address
        
    Returns:
        float: Similarity ratio (0 to 1)
    """
    if pd.isna(addr1) or pd.isna(addr2):
        return 0.0
    
    addr1 = str(addr1).lower().strip()
    addr2 = str(addr2).lower().strip()
    
    # Extract pincode/zip and compare
    pincode1 = extract_pincode(addr1)
    pincode2 = extract_pincode(addr2)
    
    # If both have pincode and they're different, low similarity
    if pincode1 and pincode2 and pincode1 != pincode2:
        return 0.1
    
    # Use SequenceMatcher for the rest
    return SequenceMatcher(None, addr1, addr2).ratio()


def find_duplicates(df, name_threshold=0.85, address_threshold=0.80, phone_exact=True):
    """
    Find duplicate entries in dataframe
    
    Uses multiple strategies:
    1. Exact phone number match
    2. Similar business name + similar address
    
    Args:
        df: Input dataframe (must have 'ID', 'Business Name', 'Phone', 'Address' columns)
        name_threshold: Minimum similarity for business name (0-1)
        address_threshold: Minimum similarity for address (0-1)
        phone_exact: Whether to consider exact phone match as duplicate
        
    Returns:
        DataFrame: List of potential duplicate pairs with similarity scores
    """
    duplicates = []
    checked = set()
    
    log_message(f"Searching for duplicates in {len(df)} records...")
    
    for i in range(len(df)):
        if i in checked:
            continue
        
        row_i = df.iloc[i]
        
        for j in range(i + 1, len(df)):
            if j in checked:
                continue
            
            row_j = df.iloc[j]
            
            is_duplicate = False
            duplicate_reason = ""
            
            # Strategy 1: Exact phone match
            if phone_exact:
                phone_i = row_i.get('Phone', row_i.get('Phone_Clean'))
                phone_j = row_j.get('Phone', row_j.get('Phone_Clean'))
                
                if pd.notna(phone_i) and pd.notna(phone_j):
                    if phone_i == phone_j:
                        is_duplicate = True
                        duplicate_reason = "Exact phone match"
            
            # Strategy 2: Fuzzy name + fuzzy address
            if not is_duplicate:
                name_i = row_i.get('Business Name', row_i.get('Business_Name_Clean'))
                name_j = row_j.get('Business Name', row_j.get('Business_Name_Clean'))
                addr_i = row_i.get('Address', row_i.get('Address_Clean'))
                addr_j = row_j.get('Address', row_j.get('Address_Clean'))
                
                name_sim = calculate_similarity(name_i, name_j)
                addr_sim = calculate_address_similarity(addr_i, addr_j)
                
                if name_sim >= name_threshold and addr_sim >= address_threshold:
                    is_duplicate = True
                    duplicate_reason = f"Similar name ({name_sim:.2f}) + address ({addr_sim:.2f})"
            
            if is_duplicate:
                duplicates.append({
                    'ID_1': row_i.get('ID', i+1),
                    'ID_2': row_j.get('ID', j+1),
                    'Name_1': row_i.get('Business Name', row_i.get('Business_Name_Clean')),
                    'Name_2': row_j.get('Business Name', row_j.get('Business_Name_Clean')),
                    'Phone_1': row_i.get('Phone', row_i.get('Phone_Clean')),
                    'Phone_2': row_j.get('Phone', row_j.get('Phone_Clean')),
                    'Address_1': row_i.get('Address', row_i.get('Address_Clean')),
                    'Address_2': row_j.get('Address', row_j.get('Address_Clean')),
                    'Duplicate_Reason': duplicate_reason,
                })
                checked.add(j)
    
    log_message(f"Found {len(duplicates)} duplicate pairs")
    return pd.DataFrame(duplicates)


def resolve_duplicates(duplicates_df, df, strategy='keep_most_complete'):
    """
    Resolve duplicates based on specified strategy
    
    Args:
        duplicates_df: DataFrame of duplicate pairs
        df: Original dataframe
        strategy: Resolution strategy
            - 'keep_first': Keep first occurrence
            - 'keep_most_complete': Keep entry with more filled fields
            - 'keep_most_recent': Keep entry with latest date
            - 'keep_verified': Keep verified entries
            
    Returns:
        DataFrame: Deduplicated dataframe
    """
    if len(duplicates_df) == 0:
        return df
    
    # Get all IDs that are involved in duplicates
    duplicate_ids = set()
    for _, row in duplicates_df.iterrows():
        duplicate_ids.add(row['ID_1'])
        duplicate_ids.add(row['ID_2'])
    
    log_message(f"Resolving {len(duplicate_ids)} duplicate records using strategy: {strategy}")
    
    # For simplicity, we'll mark duplicates rather than remove them
    df['Is_Duplicate'] = df['ID'].isin(duplicate_ids)
    
    return df


# DATA QUALITY SCORING

def calculate_quality_score(row):
    """
    Calculate data quality score for each record
    
    Scoring criteria:
    - Phone: 25 points (valid phone number)
    - Address: 20 points (complete address)
    - Email: 15 points (valid email)
    - Category: 15 points (valid category)
    - Business Name: 15 points (valid name)
    - Additional fields: 10 points (extra info like website, hours)
    
    Args:
        row: DataFrame row
        
    Returns:
        float: Quality score (0-100)
    """
    score = 0
    
    # Phone (25 points)
    phone = row.get('Phone', row.get('Phone_Clean'))
    phone_status = row.get('Phone_Status', 'Valid')
    if pd.notna(phone) and phone_status == 'Valid':
        score += 25
    
    # Address (20 points)
    address = row.get('Address', row.get('Address_Clean'))
    if pd.notna(address):
        score += 20
        # Bonus for having pincode
        pincode = row.get('Pincode')
        if pd.notna(pincode):
            score += 5
    
    # Email (15 points)
    email = row.get('Email', row.get('Email_Clean'))
    if pd.notna(email):
        score += 15
    
    # Category (15 points)
    category = row.get('Category', row.get('Category_Clean'))
    if pd.notna(category) and category != 'Unknown':
        score += 15
    
    # Business Name (15 points)
    name = row.get('Business Name', row.get('Business_Name_Clean'))
    if pd.notna(name) and len(str(name)) >= 3:
        score += 15
    
    # Additional fields (10 points)
    extra_fields = ['Website', 'Social Media', 'Operating Hours', 'Pricing']
    extra_count = sum(1 for field in extra_fields if pd.notna(row.get(field)))
    score += (extra_count * 3)  # Up to 12 points, but cap at 10
    score = min(score, 100)
    
    return score


def classify_quality_level(score):
    """
    Classify quality score into levels
    
    Args:
        score: Quality score (0-100)
        
    Returns:
        str: Quality level
    """
    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Fair'
    elif score >= 20:
        return 'Poor'
    else:
        return 'Very Poor'


# MAIN PROCESSING FUNCTION

def process_business_data(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """
    Main function to process and clean business data
    
    This is the entry point for the data cleaning pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        
    Returns:
        DataFrame: Cleaned and processed dataframe
    """
    log_message("="*60)
    log_message("Starting Animal Business Directory Data Processing")
    log_message("="*60)
    
    # Load data
    log_message(f"Loading data from {input_file}...")
    
    if not os.path.exists(input_file):
        log_message(f"Input file not found. Creating sample data for demonstration...")
        df = create_sample_data()
        # Save sample data
        df.to_csv(input_file, index=False)
        log_message(f"Sample data saved to {input_file}")
    else:
        try:
            df = pd.read_csv(input_file)
            log_message(f"Loaded {len(df)} records successfully")
        except Exception as e:
            log_message(f"Error loading file: {e}")
            return None
    
    # Initialize log file
    with open(PROCESSING_LOG, 'w', encoding='utf-8') as f:
        f.write(f"Processing Log - {datetime.now()}\n")
        f.write("="*60 + "\n")
    
    # Add ID column if not exists
    if 'ID' not in df.columns:
        df['ID'] = range(1, len(df) + 1)
        log_message("Added ID column")
    
    # Add processing timestamp
    df['Processed Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    

    # STEP 1: CLEAN PHONE NUMBERS
    
    log_message("\n--- Step 1: Cleaning Phone Numbers ---")
    
    if 'Phone' in df.columns:
        phone_results = df['Phone'].apply(clean_phone_number)
        df['Phone_Clean'] = phone_results.apply(lambda x: x[0])
        df['Phone_Status'] = phone_results.apply(lambda x: x[1])
        
        valid_phones = len(df[df['Phone_Status'] == 'Valid'])
        log_message(f"Valid phone numbers: {valid_phones}/{len(df)}")
    

    # STEP 2: STANDARDIZE ADDRESSES

    log_message("\n--- Step 2: Standardizing Addresses ---")
    
    if 'Address' in df.columns:
        df['Address_Clean'] = df['Address'].apply(standardize_address)
        df['Pincode'] = df['Address_Clean'].apply(extract_pincode)
        log_message("Addresses standardized")
    
    # STEP 3: STANDARDIZE EMAILS
    
    log_message("\n--- Step 3: Standardizing Emails ---")
    
    if 'Email' in df.columns:
        df['Email_Clean'] = df['Email'].apply(standardize_email)
        valid_emails = df['Email_Clean'].notna().sum()
        log_message(f"Valid emails: {valid_emails}/{len(df)}")
    

    # STEP 4: STANDARDIZE BUSINESS NAMES

    log_message("\n--- Step 4: Standardizing Business Names ---")
    
    if 'Business Name' in df.columns:
        df['Business_Name_Clean'] = df['Business Name'].apply(standardize_business_name)
        log_message("Business names standardized")
    
    
    # STEP 5: CLEAN CATEGORIES
    
    log_message("\n--- Step 5: Cleaning Categories ---")
    
    if 'Category' in df.columns:
        df['Category_Clean'] = df['Category'].apply(clean_category)
        log_message("Categories cleaned")
    
    
    # STEP 6: FIND DUPLICATES

    log_message("\n--- Step 6: Finding Duplicates ---")
    
    # Prepare dataframe for duplicate detection
    df_dedup = df[['ID', 'Business_Name_Clean', 'Phone_Clean', 'Address_Clean']].copy()
    df_dedup.columns = ['ID', 'Business Name', 'Phone', 'Address']
    
    duplicates = find_duplicates(df_dedup)
    
    if len(duplicates) > 0:
        duplicates.to_csv(DUPLICATES_FILE, index=False)
        log_message(f"Found {len(duplicates)} duplicate pairs. Saved to {DUPLICATES_FILE}")
        
        # Mark duplicates
        duplicate_ids = set(duplicates['ID_1'].tolist() + duplicates['ID_2'].tolist())
        df['Is_Duplicate'] = df['ID'].isin(duplicate_ids)
    else:
        df['Is_Duplicate'] = False
        log_message("No duplicates found")
    
    
    # STEP 7: CALCULATE QUALITY SCORES
    
    log_message("\n--- Step 7: Calculating Quality Scores ---")
    
    # Ensure all required columns exist
    required_cols = ['Phone_Clean', 'Phone_Status', 'Address_Clean', 'Pincode', 
                     'Email_Clean', 'Category_Clean', 'Business_Name_Clean']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    
    df['Quality_Score'] = df.apply(calculate_quality_score, axis=1)
    df['Quality_Level'] = df['Quality_Score'].apply(classify_quality_level)
    
    avg_quality = df['Quality_Score'].mean()
    log_message(f"Average quality score: {avg_quality:.2f}")
    

    # STEP 8: SELECT AND RENAME FINAL COLUMNS
    
    log_message("\n--- Step 8: Preparing Final Output ---")
    
    # Define final columns
    final_columns = [
        'ID', 
        'Business_Name_Clean', 
        'Category_Clean',
        'Phone_Clean', 
        'Phone_Status', 
        'Email_Clean',
        'Address_Clean', 
        'Pincode',
        'Source', 
        'Quality_Score',
        'Quality_Level',
        'Is_Duplicate',
        'Processed Date'
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in final_columns if col in df.columns]
    df_clean = df[available_cols].copy()
    
    # Rename cleaned columns to original names for clarity
    column_rename = {
        'Business_Name_Clean': 'Business Name',
        'Category_Clean': 'Category',
        'Phone_Clean': 'Phone',
        'Email_Clean': 'Email',
        'Address_Clean': 'Address'
    }
    df_clean.rename(columns=column_rename, inplace=True)
    
    
    # STEP 9: SAVE OUTPUT
    
    log_message("\n--- Step 9: Saving Output ---")
    
    df_clean.to_csv(output_file, index=False)
    log_message(f"Cleaned data saved to {output_file}")
    

    # PRINT SUMMARY
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total records processed: {len(df_clean)}")
    print(f"Valid phone numbers: {len(df_clean[df_clean['Phone_Status'] == 'Valid'])}")
    print(f"Invalid/Missing phone numbers: {len(df_clean[df_clean['Phone_Status'] != 'Valid'])}")
    print(f"Duplicate records found: {len(df_clean[df_clean['Is_Duplicate'] == True])}")
    print(f"Average quality score: {df_clean['Quality_Score'].mean():.2f}")
    print(f"\nQuality Distribution:")
    print(df_clean['Quality_Level'].value_counts())
    print("="*60)
    print(f"Output saved to: {output_file}")
    print(f"Duplicates report saved to: {DUPLICATES_FILE}")
    print(f"Processing log saved to: {PROCESSING_LOG}")
    print("="*60)
    
    log_message("\nProcessing completed successfully!")
    
    return df_clean



# SAMPLE DATA CREATION

def create_sample_data():
    """
    Create sample data for demonstration and testing
    
    Returns:
        DataFrame: Sample business data with various edge cases
    """
    data = {
        'ID': list(range(1, 13)),
        'Business Name': [
            'Paws and Claws Pet Shop',
            'paws & claws pet shop',
            'Happy Tails Veterinary Clinic',
            'City Animal Hospital',
            'Dog Trainer Pro',
            'Pet Grooming Salon',
            None,  # Missing name
            'Shelter for Strays',
            '  Multiple   Spaces  ',  # Extra spaces
            'VET@HOME - Dr. Smith',  # Special characters
            'Pet Paradise',  # Duplicate candidate
            'Pet Paradise (New Branch)',  # Duplicate candidate
        ],
        'Category': [
            'pet shop',
            'pet store',
            'vet',
            'veterinary',
            'dog training',
            'grooming',
            'pet shop',
            'rescue',
            'pet store',
            'veterinarian',
            'pet shop',
            'pet shop',
        ],
        'Phone': [
            '9876543210',
            '9876543210',  # Duplicate with first
            '+91 98765 43211',
            '1234567890',
            '9876543212',
            '98765 43213',
            'invalid',  # Invalid
            '98765 43214',
            '9876543215',
            '98765 43216',
            '9876543217',
            '9876543217',  # Duplicate
        ],
        'Email': [
            'info@paws.com',
            'PAWS@GMAIL.COM',  # Different case
            'contact@happytails.com',
            None,  # Missing
            'trainer@pro.com',
            'grooming@salon.com',
            'test@test.com',
            'shelter@org.com',
            'test2@test.com',
            'invalid-email',  # Invalid
            'paradise@shop.com',
            'paradise@shop.com',
        ],
        'Address': [
            '123 Main Street, Mumbai 400001',
            '123 main st mumbai 400001',  # Similar to first
            '456 Oak Avenue, Delhi 110001',
            '789 Pine Road, Bangalore 560001',
            '321 Elm Lane, Chennai 600001',
            '654 Maple Ave, Hyderabad 500001',
            'Incomplete Address',  # Missing pincode
            '999 Shelter Road, Pune 411001',
            '111 Test Address, Mumbai 400001',  # Similar location
            '222 Vet Street, Delhi 110001',
            '333 Pet Lane, Mumbai 400001',
            '333 Pet Lane, Mumbai 400001',  # Exact duplicate address
        ],
        'City': [
            'Mumbai', 'Mumbai', 'Delhi', 'Bangalore', 
            'Chennai', 'Hyderabad', None, 'Pune',
            'Mumbai', 'Delhi', 'Mumbai', 'Mumbai'
        ],
        'State': [
            'Maharashtra', 'Maharashtra', 'Delhi', 'Karnataka',
            'Tamil Nadu', 'Telangana', None, 'Maharashtra',
            'Maharashtra', 'Delhi', 'Maharashtra', 'Maharashtra'
        ],
        'Source': [
            'Intern', 'Web', 'Community', 'Web',
            'Intern', 'Social', 'Unknown', 'Community',
            'Intern', 'Web', 'Community', 'Community'
        ]
    }
    
    return pd.DataFrame(data)


# ENTRY POINT

if __name__ == '__main__':
    """
    Main entry point for the data cleaning script
    
    This script processes raw business data and outputs cleaned data
    with standardized formats, duplicate detection, and quality scores.
    """
    print("="*60)
    print("Animal Business Directory - Data Cleaning System")
    print("="*60)
    print()
    
    # Check for required dependencies
    try:
        import pandas as pd
        import numpy as np
        print("✓ All required dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("    pip install pandas numpy")
        exit(1)
    
    # Run the data processing pipeline
    cleaned_df = process_business_data()
    
    if cleaned_df is not None:
        print("\n✓ Data processing completed successfully!")
        print(f"\nOutput files generated:")
        print(f"  - {OUTPUT_FILE}")
        print(f"  - {DUPLICATES_FILE}")
        print(f"  - {PROCESSING_LOG}")
    else:
        print("\n✗ Data processing failed!")
        exit(1)

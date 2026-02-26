# utils/validators.py
import re

def validate_iran_national_code(national_code: str) -> bool:
    """
    Validate Iranian national code (کد ملی)
    
    Rules:
    - Must be exactly 10 digits
    - Can't be all same digits (e.g., 1111111111, 0000000000)
    - Has a valid checksum
    
    Args:
        national_code: The national code string to validate (must be exactly 10 digits)
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string of exactly 10 digits
    if not national_code or not isinstance(national_code, str):
        return False
    
    # Must be exactly 10 digits, no more no less
    if not re.match(r'^\d{10}$', national_code):
        return False
    
    # Check if all digits are the same (invalid)
    if len(set(national_code)) == 1:
        return False
    
    try:
        # Convert to list of integers
        digits = [int(d) for d in national_code]
        
        # Calculate checksum
        sum_products = 0
        for i in range(9):
            sum_products += digits[i] * (10 - i)
        
        remainder = sum_products % 11
        
        # Check control digit
        if remainder < 2:
            return digits[9] == remainder
        else:
            return digits[9] == (11 - remainder)
            
    except (ValueError, TypeError, IndexError):
        return False

def normalize_national_code(national_code: str) -> str:
    """
    Only removes any non-digit characters but DOES NOT pad with zeros.
    Returns None if the result isn't exactly 10 digits.
    """
    if not national_code:
        return national_code
    
    # Remove non-digits only
    digits_only = re.sub(r'\D', '', national_code)
    
    # Must be exactly 10 digits after cleaning
    if len(digits_only) != 10:
        return None
    
    return digits_only
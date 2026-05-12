import jdatetime
from datetime import datetime, timedelta
import pytz
from typing import Optional, Tuple

# Define Iran timezone
IRAN_TZ = pytz.timezone('Asia/Tehran')
UTC = pytz.UTC

def gregorian_to_persian(dt: datetime, include_time: bool = True, persian_numbers: bool = False) -> str:
    """
    Convert Gregorian datetime to Persian (Jalali) date string
    
    Args:
        dt: Gregorian datetime (naive or aware)
        include_time: Include time in output
        persian_numbers: Use Persian digits (۰-۹) instead of English
    
    Returns:
        Persian date string: "1402/02/22 14:30:25" or "1402/02/22"
    """
    if dt is None:
        return None
    
    # Ensure timezone is set (assume UTC if naive)
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    
    # Convert to Iran timezone
    iran_time = dt.astimezone(IRAN_TZ)
    
    # Convert to Persian date
    persian_date = jdatetime.datetime.fromgregorian(
        year=iran_time.year,
        month=iran_time.month,
        day=iran_time.day,
        hour=iran_time.hour,
        minute=iran_time.minute,
        second=iran_time.second,
        microsecond=iran_time.microsecond
    )
    
    # Format output
    if include_time:
        result = persian_date.strftime("%Y/%m/%d - %H:%M:%S")
    else:
        result = persian_date.strftime("%Y/%m/%d")
    
    # Convert to Persian numbers if requested
    # if persian_numbers:
    #     result = to_persian_numbers(result)
    
    return result

def persian_to_gregorian(persian_date_str: str, as_time_range: bool = True) -> Tuple[datetime, datetime]:
    """
    Convert Persian date string to Gregorian datetime range for filtering
    
    Args:
        persian_date_str: Persian date in format "1402/02/22" or "1402/02/22 - 14:30:25"
        as_time_range: If True, returns (start_of_day, end_of_day) for date filtering
                      If False and time provided, returns exact datetime
    
    Returns:
        Tuple of (start_datetime, end_datetime) in UTC for range filtering
        Or single datetime if as_time_range=False and time provided
    """
    if persian_date_str is None:
        return None
    
    # Parse Persian date
    parts = persian_date_str.split(' - ')
    date_parts = parts[0].split('/')
    
    year = int(date_parts[0])
    month = int(date_parts[1])
    day = int(date_parts[2])
    
    # Extract time if available
    time_parts = None
    if len(parts) > 1:
        time_parts = parts[1].split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) > 2 else 0
    else:
        hour = minute = second = 0
    
    # Create Persian datetime
    if time_parts and not as_time_range:
        # Exact datetime with time
        persian_dt = jdatetime.datetime(
            year=year, month=month, day=day,
            hour=hour, minute=minute, second=second
        )
        # Convert to Gregorian
        gregorian_dt = persian_dt.togregorian()
        
        # Add timezone (Iran time) then convert to UTC
        if gregorian_dt.tzinfo is None:
            gregorian_dt = IRAN_TZ.localize(gregorian_dt)
        utc_dt = gregorian_dt.astimezone(UTC)
        
        return utc_dt
    else:
        # Date range (start to end of that day)
        # Start of day in Iran time
        persian_start = jdatetime.datetime(
            year=year, month=month, day=day,
            hour=0, minute=0, second=0
        )
        # End of day in Iran time
        persian_end = jdatetime.datetime(
            year=year, month=month, day=day,
            hour=23, minute=59, second=59
        )
        
        # Convert both to UTC
        gregorian_start = persian_start.togregorian()
        gregorian_end = persian_end.togregorian()
        
        if gregorian_start.tzinfo is None:
            gregorian_start = IRAN_TZ.localize(gregorian_start)
            gregorian_end = IRAN_TZ.localize(gregorian_end)
        
        utc_start = gregorian_start.astimezone(UTC)
        utc_end = gregorian_end.astimezone(UTC)
        
        return (utc_start, utc_end)

# def to_persian_numbers(text: str) -> str:
#     """Convert English numbers to Persian (Farsi) numbers"""
#     persian_digits = {
#         '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
#         '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
#     }
#     for en, fa in persian_digits.items():
#         text = text.replace(en, fa)
#     return text

# def to_english_numbers(text: str) -> str:
#     """Convert Persian numbers to English numbers"""
#     english_digits = {
#         '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
#         '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
#     }
#     for fa, en in english_digits.items():
#         text = text.replace(fa, en)
#     return text

def get_current_persian_time(include_time: bool = True, persian_numbers: bool = False) -> str:
    """Get current time in Persian format"""
    now_utc = datetime.now(UTC)
    return gregorian_to_persian(now_utc, include_time, persian_numbers)

def persian_date_range(months_back: int = 0, days_back: int = 0) -> Tuple[datetime, datetime]:
    """
    Get date range for filtering based on Persian calendar
    
    Args:
        months_back: Number of months to go back
        days_back: Number of days to go back
    
    Returns:
        Tuple of (start_datetime, end_datetime) in UTC
    """
    now_utc = datetime.now(UTC)
    now_iran = now_utc.astimezone(IRAN_TZ)
    
    # Convert to Persian
    persian_now = jdatetime.datetime.fromgregorian(
        year=now_iran.year,
        month=now_iran.month,
        day=now_iran.day,
        hour=0, minute=0, second=0
    )
    
    # Subtract months/days
    persian_start = persian_now - jdatetime.timedelta(days=days_back)
    if months_back > 0:
        # Subtract months (simplified - you may want more precision)
        persian_start = persian_start.replace(
            month=persian_start.month - months_back
        )
    
    # Convert back to Gregorian UTC range
    gregorian_start = persian_start.togregorian()
    gregorian_start = IRAN_TZ.localize(gregorian_start)
    utc_start = gregorian_start.astimezone(UTC)
    
    utc_end = now_utc
    
    return (utc_start, utc_end)
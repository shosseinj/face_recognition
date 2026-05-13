import random
import uuid
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import re
import urllib
import asyncio  # ← For asyncio.to_thread

# Add your project path to sys.path if needed
sys.path.append(r"C:\Users\mohammadloo.r\Desktop\ai")

# Import your models and functions
from backend.app.models.database import DetectionLog, Personnel, Base, FACE_STORAGE_DIR
from backend.app.utils.video_utils import save_video_with_ffmpeg
# Database connection - FIXED: Create engine first!
params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\sql,14330;"
    "DATABASE=AI_DB;"
    "UID=sa;"
    "PWD=Asd@12345;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=30;"
)

DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"

# params = urllib.parse.quote_plus(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=192.168.110.13,14330;"  # Remote server IP with port
#     "DATABASE=HR_DB_20;"             # 
#     "UID=sa;"
#     "PWD=Asd@12345;"
#     "TrustServerCertificate=yes;"     # Keep this for self-signed certs
#     "Encrypt=yes;"                    # Added encryption
#     "Connection Timeout=30;"
# )

# DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"


# ✅ Create engine FIRST
engine = create_engine(DATABASE_URL, echo=False)

# ✅ THEN create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def safe_filename(text):
    """Convert string to safe filename"""
    # Replace invalid characters
    text = re.sub(r'[^\w\-_. ]', '_', text)
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Remove any double underscores
    text = re.sub(r'_+', '_', text)
    return text

def create_fake_frame(width=640, height=480, text="", color=None):
    """Create a fake frame with some text and shapes"""
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Create blank frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with a gradient or solid color
    frame[:] = color
    
    # Add some shapes to make it look like a real frame
    # Rectangle
    cv2.rectangle(frame, (50, 50), (width-50, height-50), (255, 255, 255), 2)
    
    # Circle
    cv2.circle(frame, (width//2, height//2), 100, (255, 255, 255), 2)
    
    # Text
    if text:
        cv2.putText(frame, text, (width//4, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise to make it look realistic
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def create_fake_face(person_name):
    """Create a fake face image"""
    # Create a blank image
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    
    # Fill with skin tone color
    skin_tone = (random.randint(180, 220), random.randint(140, 180), random.randint(100, 140))
    face[:] = skin_tone
    
    # Add eyes (two black dots)
    cv2.circle(face, (40, 45), 5, (0, 0, 0), -1)
    cv2.circle(face, (72, 45), 5, (0, 0, 0), -1)
    
    # Add mouth (a line)
    cv2.line(face, (40, 75), (72, 75), (0, 0, 0), 2)
    
    # Add some texture
    face = cv2.GaussianBlur(face, (3, 3), 0)
    
    return face

def generate_fake_detections(num_detections=100):
    """Generate fake detections for the last month"""
    
    db = SessionLocal()
    
    try:
        # Get all existing personnel
        personnel_list = db.query(Personnel).all()
        
        if not personnel_list:
            print("❌ No personnel found in database. Please add personnel first.")
            return
        
        print(f"✅ Found {len(personnel_list)} personnel in database")
        
        # Create lists of known and unknown persons
        known_persons = [p.national_code for p in personnel_list]
        all_persons = known_persons + ["Unknown"] * 5  # Add Unknown as option
        
        # Generate detections for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"📅 Generating {num_detections} detections from {start_date.date()} to {end_date.date()}")
        
        created_count = 0
        skipped_count = 0
        
        for i in range(num_detections):
            try:
                # Random date within last month
                random_days = random.randint(0, 30)
                random_hours = random.randint(0, 23)
                random_minutes = random.randint(0, 59)
                random_seconds = random.randint(0, 59)
                
                detection_time = end_date - timedelta(
                    days=random_days,
                    hours=random_hours,
                    minutes=random_minutes,
                    seconds=random_seconds
                )
                
                # Choose person
                if random.random() < 0.7:  # 70% chance of known person
                    person = random.choice(known_persons)
                    confidence = random.uniform(0.75, 0.99)
                else:
                    person = "Unknown"
                    confidence = random.uniform(0.4, 0.7)
                
                # Create fake face image
                face_image = create_fake_face(person)
                
                # Create fake frames for video
                frames = []
                num_frames = random.randint(30, 100)
                for j in range(num_frames):
                    color = (
                        random.randint(100, 255),
                        random.randint(100, 255),
                        random.randint(100, 255)
                    )
                    frame = create_fake_frame(
                        width=640, 
                        height=480, 
                        text=f"Person: {person}",
                        color=color
                    )
                    frames.append(frame)
                
                # Create directories if they don't exist
                date_str = detection_time.strftime("%Y-%m-%d")
                base_media_dir = FACE_STORAGE_DIR.parent / "saved_media"
                video_dir = base_media_dir / date_str / "video"
                image_dir = base_media_dir / date_str / "image"
                
                video_dir.mkdir(parents=True, exist_ok=True)
                image_dir.mkdir(parents=True, exist_ok=True)
                
                # Save video
                timestamp = detection_time.strftime('%Y%m%d_%H%M%S')
                unique_id = uuid.uuid4().hex[:8]
                safe_person = safe_filename(person)
                video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
                video_path = video_dir / video_filename
                
                print(f"\n📹 Creating video {i+1}/{num_detections}: {video_filename}")
                
                # Save with FFmpeg
                # success = save_video_with_ffmpeg(
                #     frames=frames,
                #     output_path=str(video_path),
                #     fps=20,
                #     quality="medium",
                #     for_web=True
                # )
                
                success = asyncio.to_thread(
                    save_video_with_ffmpeg,  # Original sync function
                    frames, str(video_path), 20, "medium", True
                )
                                
                if not success:
                    print(f"⚠️ FFmpeg failed for video {i+1}, skipping...")
                    skipped_count += 1
                    continue
                
                # Save face image - FIXED FILENAME
                timestamp_long = detection_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]
                confidence_str = f"{confidence:.2f}".replace(".", "_")  # Replace dot with underscore
                unique_id_short = uuid.uuid4().hex[:6]
                
                face_filename = f"{safe_person}_{confidence_str}_{timestamp_long}_{unique_id_short}.jpg"
                face_path = image_dir / face_filename
                
                print(f"🖼️ Saving face: {face_filename}")
                
                # Save face image
                success_face = cv2.imwrite(str(face_path), face_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                if not success_face:
                    print(f"⚠️ Failed to save face image for {person}")
                    # Try with a simpler filename
                    face_filename = f"{safe_person}_{timestamp_long}.jpg"
                    face_path = image_dir / face_filename
                    cv2.imwrite(str(face_path), face_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                # Create database record
                detection = DetectionLog(
                    person=person,
                    confidence=confidence,
                    detection_time=detection_time,
                    face_image_path=str(face_path),
                    video_path=str(video_path),
                    camera_id=random.choice([1, 2, 3, None]),
                    created_at=detection_time
                )
                
                db.add(detection)
                created_count += 1
                
                if created_count % 10 == 0:
                    print(f"\n📊 Progress: Created {created_count} detections...")
                    db.flush()  # Partial commit to avoid memory issues
                
            except Exception as e:
                print(f"❌ Error creating detection {i+1}: {e}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue
        
        # Final commit
        db.commit()
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: Created {created_count} fake detections")
        print(f"⚠️ Skipped: {skipped_count} detections due to errors")
        
        # Show summary
        print(f"\n📊 Summary by person type:")
        known_count = db.query(DetectionLog).filter(DetectionLog.person != "Unknown").count()
        unknown_count = db.query(DetectionLog).filter(DetectionLog.person == "Unknown").count()
        print(f"  Known persons: {known_count}")
        print(f"  Unknown: {unknown_count}")
        
        # Show recent detections
        recent = db.query(DetectionLog).order_by(DetectionLog.detection_time.desc()).limit(5).all()
        print(f"\n🆕 Most recent detections:")
        for det in recent:
            display_name = det.person
            if det.person != "Unknown":
                personnel = db.query(Personnel).filter(Personnel.national_code == det.person).first()
                if personnel:
                    display_name = f"{personnel.fname} {personnel.lname} ({det.person})"
            print(f"  ID: {det.id}, Person: {display_name}, Time: {det.detection_time}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

def generate_detections_for_specific_person(national_code, num_detections=20):
    """Generate fake detections for a specific person"""
    
    db = SessionLocal()
    
    try:
        # Verify person exists
        person = db.query(Personnel).filter(Personnel.national_code == national_code).first()
        if not person:
            print(f"❌ Person with national code {national_code} not found")
            return
        
        print(f"✅ Generating {num_detections} detections for {person.fname} {person.lname} ({national_code})")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        created_count = 0
        
        for i in range(num_detections):
            # Random time in last 30 days
            random_days = random.randint(0, 30)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            random_seconds = random.randint(0, 59)
            
            detection_time = end_date - timedelta(
                days=random_days,
                hours=random_hours,
                minutes=random_minutes,
                seconds=random_seconds
            )
            
            confidence = random.uniform(0.8, 0.98)
            
            # Create fake face
            face_image = create_fake_face(national_code)
            
            # Create directories
            date_str = detection_time.strftime("%Y-%m-%d")
            base_media_dir = FACE_STORAGE_DIR.parent / "saved_media"
            video_dir = base_media_dir / date_str / "video"
            image_dir = base_media_dir / date_str / "image"
            
            video_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # Save video (minimal frames for specific person)
            frames = [create_fake_frame(text=f"{person.fname} {person.lname}") for _ in range(30)]
            
            timestamp = detection_time.strftime('%Y%m%d_%H%M%S')
            unique_id = uuid.uuid4().hex[:8]
            safe_person = safe_filename(national_code)
            video_filename = f"{safe_person}_{timestamp}_{unique_id}.mp4"
            video_path = video_dir / video_filename
            
            print(f"\n📹 Creating video for {person.fname}: {video_filename}")

            # save_video_with_ffmpeg(frames, str(video_path), fps=20)
            await asyncio.to_thread(
                save_video_with_ffmpeg,
                frames,
                str(video_path),
                20,      # fps
                "veryslow", # quality
                True     # for_web
            )
            


            # Save face - FIXED FILENAME
            timestamp_long = detection_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            confidence_str = f"{confidence:.2f}".replace(".", "_")  # Replace dot with underscore
            unique_id_short = uuid.uuid4().hex[:6]
            
            face_filename = f"{safe_person}_{confidence_str}_{timestamp_long}_{unique_id_short}.jpg"
            face_path = image_dir / face_filename
            
            print(f"🖼️ Saving face: {face_filename}")
            cv2.imwrite(str(face_path), face_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Create record
            detection = DetectionLog(
                person=national_code,
                confidence=confidence,
                detection_time=detection_time,
                face_image_path=str(face_path),
                video_path=str(video_path),
                camera_id=1,
                created_at=detection_time
            )
            
            db.add(detection)
            created_count += 1
            
            if created_count % 5 == 0:
                db.flush()
                print(f"  Created {created_count} detections for {national_code}")
        
        db.commit()
        print(f"\n✅ Created {created_count} detections for {national_code}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

def clear_all_detections():
    """Delete all detection records and files"""
    db = SessionLocal()
    try:
        # Get all detections
        detections = db.query(DetectionLog).all()
        
        if not detections:
            print("No detections to clear")
            return
        
        print(f"Found {len(detections)} detections to delete")
        
        # Delete files
        files_deleted = 0
        for det in detections:
            if det.face_image_path and os.path.exists(det.face_image_path):
                try:
                    os.remove(det.face_image_path)
                    files_deleted += 1
                except:
                    pass
            if det.video_path and os.path.exists(det.video_path):
                try:
                    os.remove(det.video_path)
                    files_deleted += 1
                except:
                    pass
        
        # Delete records
        count = db.query(DetectionLog).delete()
        db.commit()
        print(f"✅ Deleted {count} detection records and {files_deleted} associated files")
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 FAKE DETECTION GENERATOR")
    print("=" * 60)
    
    # Option to clear first
    print("\n🗑️ Option 0: Clear all existing detections")
    clear_option = input("Clear all detections first? (y/n): ").strip().lower()
    if clear_option == 'y':
        clear_all_detections()
    
    # Option 1: Generate random detections
    print("\n📊 Option 1: Generate random detections for all personnel")
    num = input("How many detections to generate? (default: 100): ").strip()
    num = int(num) if num else 100
    generate_fake_detections(num)
    
    # Option 2: Generate for specific person
    print("\n👤 Option 2: Generate detections for specific person")
    specific = input("Generate for specific person? (y/n): ").strip().lower()
    if specific == 'y':
        national_code = input("Enter national code: ").strip()
        if national_code:
            num_specific = input("How many detections? (default: 20): ").strip()
            num_specific = int(num_specific) if num_specific else 20
            generate_detections_for_specific_person(national_code, num_specific)
    
    print("\n✨ Done!")
# camera/tasks.py
import os
import cv2
import json
import time
import threading
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from zoneinfo import ZoneInfo
from django.core.mail import EmailMessage
from django.conf import settings
from ultralytics import YOLO
from .models import UserAiModel

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Table, TableStyle
import re

# -------------------------------
# Logger
# -------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def atomic_write_json(path, data):
    """
    Safely write JSON to disk:
    - Write into a temporary file first
    - Atomically replace the old file with the new one
    """
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2)
        tmp_name = tmp.name

    os.replace(tmp_name, path)

# -------------------------------
# Celery Test Task (optional)
# -------------------------------
def test_task():
    print(f"[{datetime.now()}] ✅ Celery test task ran successfully!")
    return "Done"


# -------------------------------
# Utilities
# -------------------------------
def sanitize_filename(filename):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', filename)


# -------------------------------
# PDF Report Generator
# -------------------------------
def create_daily_report_pdf(user_name, date, total_in, total_out, camera_details):
    """
    Generate a branded PDF daily report for a user with enhanced styling.
    Includes overall summary and individual camera details.
    
    Args:
        user_name: Name of the user
        date: Date of the report
        total_in: Total entries across all cameras
        total_out: Total exits across all cameras
        camera_details: List of dicts with camera-wise data
                       [{'camera_name': 'Camera 1', 'in': 5, 'out': 3}, ...]
    """
    try:
        tz = ZoneInfo("Asia/Kolkata")
        now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        reports_dir = os.path.join(settings.MEDIA_ROOT, "daily_reports_pdf")
        os.makedirs(reports_dir, exist_ok=True)
        safe_user = sanitize_filename(str(user_name))
        safe_date = sanitize_filename(str(date))
        file_path = os.path.join(reports_dir, f"daily_report_{safe_user}_{safe_date}.pdf")
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4
        
        # Colors
        primary_color = HexColor('#4f46e5')
        secondary_color = HexColor('#7c3aed')
        accent_color = HexColor('#f59e0b')
        light_bg = HexColor('#f8fafc')
        text_color = HexColor('#1f2937')
        success_color = HexColor('#10b981')
        danger_color = HexColor('#ef4444')
        warning_color = HexColor('#f97316')
        info_color = HexColor('#3b82f6')
        
        net_flow = total_in - total_out
        entry_rate = round((total_in / max(total_in + total_out, 1)) * 100, 1) if (total_in + total_out) > 0 else 0
        
        # Determine activity level and efficiency
        activity_level = "High Activity" if (total_in + total_out) > 10 else "Low Activity"
        flow_balance = "Balanced Flow" if abs(net_flow) <= 2 else ("Inflow Heavy" if net_flow > 0 else "Outflow Heavy")
        efficiency_score = min(96, max(60, 100 - abs(net_flow) * 2))
        
        # --- Header ---
        c.setFillColor(primary_color)
        c.rect(0, height - 2.2*inch, width, 2.2*inch, fill=1, stroke=0)
        c.setFillColor(secondary_color)
        c.rect(0, height - 1.8*inch, width, 0.4*inch, fill=1, stroke=0)
        
        # Logo
        logo_path = os.path.join(getattr(settings, "IMAGE_FILES", ""), "camex_logo.png")
        if os.path.exists(logo_path):
            try:
                logo = ImageReader(logo_path)
                c.drawImage(logo, width - 2.5*inch, height - 1.6*inch,
                            width=1.8*inch, height=0.8*inch, preserveAspectRatio=True)
            except Exception as e:
                logger.warning(f"Logo load failed: {e}")
        
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 28)
        c.drawString(0.8*inch, height - 1.2*inch, "Daily Analytics Report")
        c.setFont("Helvetica", 14)
        c.drawString(0.8*inch, height - 1.6*inch, f"Generated on {now_str}")
        
        # --- User Info ---
        info_y = height - 3*inch
        c.setFillColor(light_bg)
        c.rect(0.5*inch, info_y - 0.8*inch, width - 1*inch, 0.8*inch, fill=1, stroke=0)
        c.setFillColor(text_color)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.8*inch, info_y - 0.3*inch, f"Report for: {user_name}")
        c.setFont("Helvetica", 12)
        c.drawString(0.8*inch, info_y - 0.6*inch, f"Date: {date}")
        
        # --- Overall Summary Cards ---
        cards_y = info_y - 2.2*inch
        card_width = 2.3 * inch
        card_height = 1.2 * inch
        card_spacing = 0.5 * inch
        center_x = width / 2
        total_cards_width = 3 * card_width + 2 * card_spacing
        start_x = center_x - total_cards_width / 2
        
        # Entry Card
        c.setFillColor(success_color)
        c.roundRect(start_x, cards_y, card_width, card_height, 10, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 32)
        c.drawCentredString(start_x + card_width/2, cards_y + 0.75*inch, str(total_in))
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(start_x + card_width/2, cards_y + 0.45*inch, "TOTAL ENTRIES")
        c.setFont("Helvetica", 8)
        c.drawCentredString(start_x + card_width/2, cards_y + 0.25*inch, "People In")
        
        # Exit Card
        exit_x = start_x + card_width + card_spacing
        c.setFillColor(danger_color)
        c.roundRect(exit_x, cards_y, card_width, card_height, 10, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 32)
        c.drawCentredString(exit_x + card_width/2, cards_y + 0.75*inch, str(total_out))
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(exit_x + card_width/2, cards_y + 0.45*inch, "TOTAL EXITS")
        c.setFont("Helvetica", 8)
        c.drawCentredString(exit_x + card_width/2, cards_y + 0.25*inch, "People Out")
        
        # Net Flow Card
        net_x = exit_x + card_width + card_spacing
        net_color = info_color if net_flow >= 0 else warning_color
        c.setFillColor(net_color)
        c.roundRect(net_x, cards_y, card_width, card_height, 10, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 32)
        c.drawCentredString(net_x + card_width/2, cards_y + 0.75*inch, f"{net_flow:+d}")
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(net_x + card_width/2, cards_y + 0.45*inch, "NET FLOW")
        c.setFont("Helvetica", 8)
        flow_text = "Net Positive" if net_flow >= 0 else "Net Negative"
        c.drawCentredString(net_x + card_width/2, cards_y + 0.25*inch, flow_text)
        
        # --- Detailed Analytics Summary (Blue Section) ---
        analytics_y = cards_y - 1.5*inch
        c.setFillColor(primary_color)
        c.rect(0.5*inch, analytics_y - 1.2*inch, width - 1*inch, 0.4*inch, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width/2, analytics_y - 0.8*inch, "■ Detailed Analytics Summary")
        
        # Analytics table data
        table_y = analytics_y - 1.6*inch
        c.setFillColor(text_color)
        c.setFont("Helvetica-Bold", 11)
        
        # Table rows
        row_height = 0.25*inch
        col1_x = 1*inch
        col2_x = width - 2.5*inch
        
        # Entry Rate
        c.drawString(col1_x, table_y, "Entry Rate:")
        c.setFont("Helvetica", 11)
        c.drawRightString(col2_x, table_y, f"{entry_rate}%")
        
        # Activity Level
        c.setFont("Helvetica-Bold", 11)
        c.drawString(col1_x, table_y - row_height, "Activity Level:")
        c.setFont("Helvetica", 11)
        c.drawRightString(col2_x, table_y - row_height, activity_level)
        
        # Flow Balance
        c.setFont("Helvetica-Bold", 11)
        c.drawString(col1_x, table_y - 2*row_height, "Flow Balance:")
        c.setFont("Helvetica", 11)
        c.drawRightString(col2_x, table_y - 2*row_height, flow_balance)
        
        # Efficiency Score
        c.setFont("Helvetica-Bold", 11)
        c.drawString(col1_x, table_y - 3*row_height, "Efficiency Score:")
        c.setFont("Helvetica", 11)
        c.drawRightString(col2_x, table_y - 3*row_height, f"{efficiency_score}%")
        
        # --- Smart Insights Section (Orange Section) ---
        insights_y = table_y - 4.5*row_height
        c.setFillColor(accent_color)
        c.rect(0.5*inch, insights_y - 1.8*inch, width - 1*inch, 0.4*inch, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, insights_y - 0.8*inch, "■ Smart Insights & Recommendations")
        
        # Insights content
        insights_content_y = insights_y - 1.2*inch
        c.setFillColor(light_bg)
        c.rect(0.5*inch, insights_content_y - 1.4*inch, width - 1*inch, 1.4*inch, fill=1, stroke=0)
        
        c.setFillColor(text_color)
        c.setFont("Helvetica", 10)
        insight_text_y = insights_content_y - 0.3*inch
        
        # Generate insights based on data
        insights = []
        if activity_level == "Low Activity":
            insights.append("• Low activity day - review marketing or operational factors")
        else:
            insights.append("• High activity detected - monitor capacity management")
            
        if flow_balance == "Balanced Flow":
            insights.append("• Balanced in-out flow shows stable operations")
        elif "Inflow" in flow_balance:
            insights.append("• Higher inflow detected - ensure adequate space and services")
        else:
            insights.append("• Higher outflow detected - analyze visitor satisfaction factors")
            
        insights.append(f"• Entry efficiency at {entry_rate}% - {'needs improvement' if entry_rate < 50 else 'performing well'}")
        
        for insight in insights:
            c.drawString(1*inch, insight_text_y, insight)
            insight_text_y -= 0.25*inch
        
        # Overall Performance Score
        score_y = insights_content_y - 1.8*inch
        c.setFillColor(text_color)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, score_y, f"Overall Performance Score: {efficiency_score}%")
        
        # Performance bar
        bar_y = score_y - 0.3*inch
        bar_width = width - 2*inch
        bar_height = 0.2*inch
        
        # Background bar
        c.setFillColor(HexColor('#e5e7eb'))
        c.rect(1*inch, bar_y, bar_width, bar_height, fill=1, stroke=0)
        
        # Progress bar
        progress_width = (efficiency_score / 100) * bar_width
        progress_color = success_color if efficiency_score >= 80 else (warning_color if efficiency_score >= 60 else danger_color)
        c.setFillColor(progress_color)
        c.rect(1*inch, bar_y, progress_width, bar_height, fill=1, stroke=0)
        
        # --- Camera-wise Details Section ---
        if camera_details and len(camera_details) > 0:
            # Check if we need a new page
            print("Camera Details: ",camera_details)
            current_y = bar_y - 0.8*inch
            needed_space = 2*inch + len(camera_details) * 0.8*inch  # Header + camera cards
            
            if current_y < needed_space:
                c.showPage()
                current_y = height - 1*inch
            
            # Camera Details Header
            camera_header_y = current_y
            c.setFillColor(secondary_color)
            c.rect(0.5*inch, camera_header_y - 0.5*inch, width - 1*inch, 0.5*inch, fill=1, stroke=0)
            c.setFillColor(white)
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width/2, camera_header_y - 0.3*inch, "Individual Camera Analytics")
            
            # Camera cards
            camera_y = camera_header_y - 1*inch
            card_height = 0.7*inch
            card_margin = 0.1*inch
            
            for i, camera in enumerate(camera_details):
                if camera_y < 1*inch:  # Start new page if needed
                    c.showPage()
                    camera_y = height - 1*inch
                
                camera_name = camera.get('camera_name', f'Camera {i+1}')
                camera_in = camera.get('in', 0)
                camera_out = camera.get('out', 0)
                camera_net = camera_in - camera_out
                
                # Camera card background
                c.setFillColor(light_bg)
                c.rect(0.5*inch, camera_y - card_height, width - 1*inch, card_height, fill=1, stroke=0)
                
                # Camera name
                c.setFillColor(text_color)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(0.8*inch, camera_y - 0.25*inch, camera_name)
                
                # Camera stats - mini cards
                mini_card_width = 1.2*inch
                mini_card_height = 0.35*inch
                mini_card_y = camera_y - 0.6*inch
                
                # In count
                in_x = width - 4.2*inch
                c.setFillColor(success_color)
                c.roundRect(in_x, mini_card_y, mini_card_width, mini_card_height, 5, fill=1, stroke=0)
                c.setFillColor(white)
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(in_x + mini_card_width/2, mini_card_y + 0.18*inch, str(camera_in))
                c.setFont("Helvetica", 8)
                c.drawCentredString(in_x + mini_card_width/2, mini_card_y + 0.05*inch, "IN")
                
                # Out count
                out_x = in_x + mini_card_width + 0.1*inch
                c.setFillColor(danger_color)
                c.roundRect(out_x, mini_card_y, mini_card_width, mini_card_height, 5, fill=1, stroke=0)
                c.setFillColor(white)
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(out_x + mini_card_width/2, mini_card_y + 0.18*inch, str(camera_out))
                c.setFont("Helvetica", 8)
                c.drawCentredString(out_x + mini_card_width/2, mini_card_y + 0.05*inch, "OUT")
                
                # Net flow
                net_x = out_x + mini_card_width + 0.1*inch
                net_mini_color = info_color if camera_net >= 0 else warning_color
                c.setFillColor(net_mini_color)
                c.roundRect(net_x, mini_card_y, mini_card_width, mini_card_height, 5, fill=1, stroke=0)
                c.setFillColor(white)
                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(net_x + mini_card_width/2, mini_card_y + 0.18*inch, f"{camera_net:+d}")
                c.setFont("Helvetica", 8)
                c.drawCentredString(net_x + mini_card_width/2, mini_card_y + 0.05*inch, "NET")
                
                camera_y -= (card_height + card_margin)
        
        # --- Footer ---
        c.setFillColor(HexColor('#374151'))
        c.rect(0, 0, width, 1.5*inch, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width/2, 1*inch, "Thank you for choosing Camex Platform")
        c.setFont("Helvetica", 10)
        c.drawCentredString(width/2, 0.7*inch, "AI-Powered Monitoring & Analytics Solutions")
        c.drawCentredString(width/2, 0.5*inch, "Our team is available 24/7 to support your business needs")
        c.setFont("Helvetica", 8)
        c.drawCentredString(width/2, 0.2*inch, "© 2025 Camex Platform. All rights reserved. | Generated with advanced AI analytics")
        
        c.save()
        return file_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None



# -------------------------------
# Email Sender
# -------------------------------

from camera.models import Camera, InOutStats, User

def send_daily_report_email(user):
    """
    Generate daily report PDF for a user (all cameras + totals) and send via email.
    """
    print("📧 Sending daily report email")

    tz = ZoneInfo("Asia/Kolkata")
    today = datetime.now(tz).date()

    # Fetch user and active cameras
    user_instance = User.objects.get(email=user)
    cameras = Camera.objects.filter(created_by=user_instance, is_active=True)

    print(f"🔎 User {user} has {cameras.count()} active cameras")

    total_in, total_out = 0, 0
    camera_details = []

    for cam in cameras:
        stats = InOutStats.objects.filter(user=user_instance, camera=cam, date=today)

        print(f"🔎 Camera {cam.name} has {stats.count()} records for {today}")

        cam_in = sum(s.total_in for s in stats)
        cam_out = sum(s.total_out for s in stats)

        total_in += cam_in
        total_out += cam_out

        camera_details.append({
            "camera_name": cam.name,
            "in": cam_in,
            "out": cam_out
        })

    # ✅ Generate PDF report
    pdf_file = create_daily_report_pdf(
        user_name=user,
        date=today,
        total_in=total_in,
        total_out=total_out,
        camera_details=camera_details
    )

    subject = "Camex Platform - Daily Report"
    body = f"""
Hello,

Hope you are having a great day! Please find the attached PDF report 
with detailed statistics for {today}.

Best regards,  
Camex Platform Team
"""

    from_email = settings.DEFAULT_FROM_EMAIL
    to_email = [user]

    email = EmailMessage(subject, body, from_email, to_email)

    if pdf_file and os.path.exists(pdf_file):
        email.attach_file(pdf_file)
        print(f"📎 Attached report: {pdf_file}")
    else:
        print(f"⚠️ Report file not found: {pdf_file}")

    email.send()
    print("✅ Daily report email sent successfully")


# -------------------------------
# Daily Reports
# -------------------------------
# camera/tasks.py
def generate_daily_in_out_reports():
    """
    Generate a single daily summary file per user and send it by email.
    """
    from django.contrib.auth import get_user_model

    tz = ZoneInfo("Asia/Kolkata")
    today = datetime.now(tz).date()

    base_dir = os.path.join(settings.MEDIA_ROOT, "in_out_stats")
    summary_dir = os.path.join(settings.MEDIA_ROOT, "daily_summary")
    os.makedirs(summary_dir, exist_ok=True)

    user_totals = {}

    if os.path.exists(base_dir):
        for file in os.listdir(base_dir):
            if file.endswith(f"{today}.json") and file.startswith("inout_user"):
                path = os.path.join(base_dir, file)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                if not data:
                    continue

                parts = file.split("_")
                try:
                    user_id = int(parts[1].replace("user", ""))
                except Exception:
                    continue

                user_totals.setdefault(user_id, {"in": 0, "out": 0})
                last_entry = data[-1]
                last_stats = last_entry.get("stats", {})

                # 🔹 FIX: use correct keys ("in" / "out")
                user_totals[user_id]["in"] += last_stats.get("in", 0)
                user_totals[user_id]["out"] += last_stats.get("out", 0)

    for user_id, totals in user_totals.items():
        summary_file = os.path.join(
            summary_dir, f"daily_in_out_summary_user{user_id}_{today}.json"
        )
        with open(summary_file, "w") as f:
            json.dump({"date": str(today), "totals": totals}, f, indent=2)

    send_daily_report_email(settings.TO_EMAIL)


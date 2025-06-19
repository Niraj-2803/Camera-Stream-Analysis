from django.core.management.base import BaseCommand
from camera.models import AiModel


class Command(BaseCommand):
    help = "Load AiModel instances from predefined app_stack_data"

    def handle(self, *args, **kwargs):
        app_stack_data = [
            {
                "title": "Track Posture & Occupancy",
                "icon": "/icons/posture.png",
                "version": "3.2",
                "status": "Active",
                "model_name": "track_posture_and_occupancy",
                "description": "Tracks human posture and seat occupancy in real-time. It monitors posture (e.g., standing, sitting) and updates seat occupancy statistics.",
            },
            {
                "title": "camera Tampering",
                "icon": "/icons/nest_cam_iq.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "ANPR",
                "icon": "/icons/car1.svg",
                "version": "3.2",
                "status": "Active",
            },
            {
                "title": "Wrong Lane",
                "icon": "/icons/dangerous.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "Wrong Parking",
                "icon": "/icons/Group.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "Vehicle Trajectory & Heatmap",
                "icon": "/icons/car (6).png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "Wrong Way",
                "icon": "/icons/road (1) 1.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "Traffic Volume Estimation",
                "icon": "/icons/traffic-lights.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "People Count",
                "icon": "/icons/people count.svg",
                "version": "3.2",
                "status": "Active",
            },
            {
                "title": "People IN/OUT",
                "icon": "/icons/people in.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "Intrusion",
                "icon": "/icons/intrusion.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "No Helmet Violation",
                "icon": "/icons/no-helmet.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "no seatbelt",
                "icon": "/icons/seatbelt.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "crowd estimation",
                "icon": "/icons/Group.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "stopped vehicle",
                "icon": "/icons/drive.png",
                "version": "3.2",
                "status": "Active",
            },
            {
                "title": "axie count",
                "icon": "/icons/3d-cube.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "face recognition System",
                "icon": "/icons/electronics.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "poor visibilty",
                "icon": "/icons/visible.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "train detection",
                "icon": "/icons/train.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "fence jumping",
                "icon": "/icons/fence.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "incident detection",
                "icon": "/icons/incident.png",
                "version": "3.2",
                "status": "Active",
            },
            {
                "title": "loitering",
                "icon": "/icons/hacker.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "smoke & fire",
                "icon": "/icons/fire.svg",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "PPE Detection",
                "icon": "/icons/ppe.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "unattended object",
                "icon": "/icons/detection.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "slow moving vehicle",
                "icon": "/icons/car (6).png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "animal detection",
                "icon": "/icons/animal.svg",
                "version": "3.2",
                "status": "Active",
            },
            {
                "title": "mobile detection",
                "icon": "/icons/phone.png",
                "version": "3.2",
                "status": "Inactive",
            },
            {
                "title": "crowd count",
                "icon": "/icons/people.png",
                "version": "3.2",
                "status": "Inactive",
            },
        ]

        for item in app_stack_data:
            AiModel.objects.get_or_create(
                name=item["title"],
                defaults={
                    "function_name": item["title"].lower().replace(" ", "_"),
                    "icon": item["icon"],
                    "version": item["version"],
                    "status": item["status"],
                },
            )

        self.stdout.write(
            self.style.SUCCESS("✅ AiModel instances created successfully.")
        )

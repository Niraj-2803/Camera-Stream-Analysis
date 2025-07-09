from django.core.management.base import BaseCommand
from camera.models import AiModel


class Command(BaseCommand):
    help = "Load AiModel instances from predefined app_stack_data"

    def handle(self, *args, **kwargs):
        app_stack_data = [
            {
                "title": "camera Tampering",
                "function_name": "camera_tampering",
                "icon": "/icons/nest_cam_iq.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Wrong Lane",
                "function_name": "wrong_lane",
                "icon": "/icons/dangerous.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Wrong Parking",
                "function_name": "wrong_parking",
                "icon": "/icons/Group.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Vehicle Trajectory & Heatmap",
                "function_name": "vehicle_trajectory_&_heatmap",
                "icon": "/icons/car (6).png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Wrong Way",
                "function_name": "wrong_way",
                "icon": "/icons/road (1) 1.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Traffic Volume Estimation",
                "function_name": "traffic_volume_estimation",
                "icon": "/icons/traffic-lights.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "People IN/OUT",
                "function_name": "people_in/out",
                "icon": "/icons/people in.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Intrusion",
                "function_name": "intrusion",
                "icon": "/icons/intrusion.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "No Helmet Violation",
                "function_name": "no_helmet_violation",
                "icon": "/icons/no-helmet.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "no seatbelt",
                "function_name": "no_seatbelt",
                "icon": "/icons/seatbelt.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "crowd estimation",
                "function_name": "crowd_estimation",
                "icon": "/icons/Group.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "axie count",
                "function_name": "axie_count",
                "icon": "/icons/3d-cube.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "face recognition System",
                "function_name": "face_recognition_system",
                "icon": "/icons/electronics.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "poor visibilty",
                "function_name": "poor_visibilty",
                "icon": "/icons/visible.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "train detection",
                "function_name": "train_detection",
                "icon": "/icons/train.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "fence jumping",
                "function_name": "fence_jumping",
                "icon": "/icons/fence.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "loitering",
                "function_name": "loitering",
                "icon": "/icons/hacker.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "smoke & fire",
                "function_name": "smoke_&_fire",
                "icon": "/icons/fire.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "PPE Detection",
                "function_name": "ppe_detection",
                "icon": "/icons/ppe.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "unattended object",
                "function_name": "unattended_object",
                "icon": "/icons/detection.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "slow moving vehicle",
                "function_name": "slow_moving_vehicle",
                "icon": "/icons/car (6).png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "mobile detection",
                "function_name": "mobile_detection",
                "icon": "/icons/phone.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "crowd count",
                "function_name": "crowd_count",
                "icon": "/icons/people.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "Face blur",
                "function_name": "blur_faces",
                "icon": "abc",
                "status": "Active",
                "version": "1",
            },
            {
                "title": "People Count",
                "function_name": "count_people",
                "icon": "abc",
                "status": "Active",
                "version": "1",
            },
            {
                "title": "Seat Status",
                "function_name": "seat_status",
                "icon": "string",
                "status": "Active",
                "version": "1",
            },
            {
                "title": "Track Posture and Occupancy",
                "function_name": "track_posture_and_occupancy",
                "icon": "/icons/posture.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "ANPR",
                "function_name": "anpr",
                "icon": "/icons/car1.svg",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "stopped vehicle",
                "function_name": "stopped_vehicle",
                "icon": "/icons/drive.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "incident detection",
                "function_name": "incident_detection",
                "icon": "/icons/incident.png",
                "status": "Inactive",
                "version": "3.2",
            },
            {
                "title": "animal detection",
                "function_name": "animal_detection",
                "icon": "/icons/animal.svg",
                "status": "Inactive",
                "version": "3.2",
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

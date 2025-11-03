from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run live Indian Sign Language detection using webcam."

    def add_arguments(self, parser):
        parser.add_argument('--camera-index', type=int, default=0, help='Webcam index to use (default: 0)')
        parser.add_argument('--window-name', type=str, default='Indian sign language detector', help='OpenCV window title')

    def handle(self, *args, **options):
        try:
            from isl_detection import run_detection
        except Exception as e:
            raise CommandError(f"Failed to import detection module: {e}")

        camera_index = options['camera_index']
        window_name = options['window_name']

        try:
            run_detection(camera_index=camera_index, window_name=window_name)
        except Exception as e:
            raise CommandError(f"Detection failed: {e}")

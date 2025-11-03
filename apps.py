from django.apps import AppConfig
import threading
import subprocess
import os

class A2SLConfig(AppConfig):
    name = 'A2SL'
    verbose_name = "AI-Powered ISL Communication Assistant"

    def ready(self):
        # Prevent double execution due to Django autoreloader
        if os.environ.get('RUN_MAIN') != 'true':
            return

        def run_isl_detection():
            isl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'isl_detection.py')
            subprocess.Popen(['python', isl_path], shell=True)

        thread = threading.Thread(target=run_isl_detection)
        thread.daemon = True
        thread.start()
from django.apps import AppConfig
import os
import threading
import subprocess

class A2SLConfig(AppConfig):
    name = 'A2SL'
    verbose_name = "AI-Powered ISL Communication Assistant"

    def ready(self):
        print('Ready method is called')  # Debug line
        # Do not auto-start ISL detection at server startup.
        # Detection is started on-demand from views (live page or explicit start route).

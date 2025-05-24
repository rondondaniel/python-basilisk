"""
Context helper to add the parent directory to the path
This allows tests to import the main package modules
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import modules from the parent directory
# import agent, env, models, etc.

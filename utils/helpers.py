import os
import json
import uuid

SESSIONS_FILE = "RESEARCH_CHAT.json"
  
def load_sessions():
    """Load all chat sessions from file."""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_sessions(sessions):
    """Save all chat sessions to file."""
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def get_or_create_session(sessions, session_id=None):
    """Return existing session if found, otherwise create a new one."""
    if session_id and session_id in sessions:
        return session_id
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = []
    return new_id
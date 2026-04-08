import speech_recognition as sr
import pyautogui
import subprocess
import threading
import time
import os

# ---------- Volume Control (Windows) ----------
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    HAS_VOLUME = True
except Exception:
    HAS_VOLUME = False
    volume = None

# ---------- Spotify (optional) ----------
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    HAS_SPOTIFY = False  # Will be enabled after auth
    sp = None
except ImportError:
    HAS_SPOTIFY = False
    sp = None


def init_spotify(token=None, client_id=None, client_secret=None, redirect_uri="http://localhost:8888/callback"):
    """Initialize Spotify with a token or client credentials."""
    global sp, HAS_SPOTIFY
    try:
        if token:
            sp = spotipy.Spotify(auth=token)
        else:
            sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
            ))
        # Test connection
        sp.current_playback()
        HAS_SPOTIFY = True
        print("[Voice] Spotify connected!")
        return True
    except Exception as e:
        print(f"[Voice] Spotify connection failed: {e}")
        HAS_SPOTIFY = False
        return False


# ---------- Command Handlers ----------
def handle_command(text, status_callback=None):
    """Process a voice command. Returns True if a command was matched."""
    text = text.lower().strip()

    def set_status(msg):
        if status_callback:
            status_callback(msg)
        print(f"[Voice] {msg}")

    # --- App Control ---
    if text.startswith("open "):
        app_name = text[5:].strip()
        set_status(f"Opening {app_name}...")
        open_app(app_name)
        return True

    if text in ("close", "close window", "close this"):
        set_status("Closing window")
        pyautogui.hotkey("alt", "F4")
        return True

    if text in ("minimize", "minimize window"):
        set_status("Minimizing")
        pyautogui.hotkey("win", "down")
        return True

    if text in ("maximize", "maximize window"):
        set_status("Maximizing")
        pyautogui.hotkey("win", "up")
        return True

    if text in ("switch window", "switch", "alt tab"):
        set_status("Switching window")
        pyautogui.hotkey("alt", "tab")
        return True

    if text in ("show desktop", "desktop"):
        set_status("Showing desktop")
        pyautogui.hotkey("win", "d")
        return True

    # --- Typing / Dictation ---
    if text.startswith("type "):
        content = text[5:].strip()
        set_status(f"Typing: {content}")
        pyautogui.typewrite(content, interval=0.03) if content.isascii() else pyautogui.write(content)
        return True

    if text.startswith("press "):
        key = text[6:].strip()
        set_status(f"Pressing {key}")
        press_key(key)
        return True

    if text in ("enter", "press enter"):
        pyautogui.press("enter")
        set_status("Enter")
        return True

    if text in ("backspace", "delete that"):
        pyautogui.press("backspace")
        set_status("Backspace")
        return True

    if text in ("select all",):
        pyautogui.hotkey("ctrl", "a")
        set_status("Select all")
        return True

    if text in ("copy", "copy that"):
        pyautogui.hotkey("ctrl", "c")
        set_status("Copied")
        return True

    if text in ("paste", "paste that"):
        pyautogui.hotkey("ctrl", "v")
        set_status("Pasted")
        return True

    if text in ("undo",):
        pyautogui.hotkey("ctrl", "z")
        set_status("Undo")
        return True

    if text in ("redo",):
        pyautogui.hotkey("ctrl", "y")
        set_status("Redo")
        return True

    # --- Volume ---
    if text in ("volume up", "louder"):
        change_volume(0.1)
        set_status("Volume up")
        return True

    if text in ("volume down", "quieter", "softer"):
        change_volume(-0.1)
        set_status("Volume down")
        return True

    if text in ("mute", "mute volume"):
        toggle_mute()
        set_status("Mute toggled")
        return True

    if text in ("unmute",):
        toggle_mute(force_unmute=True)
        set_status("Unmuted")
        return True

    # --- Spotify / Media ---
    if text in ("play", "resume", "pause", "play pause"):
        if HAS_SPOTIFY and sp:
            try:
                playback = sp.current_playback()
                if playback and playback["is_playing"]:
                    sp.pause_playback()
                    set_status("Spotify paused")
                else:
                    sp.start_playback()
                    set_status("Spotify playing")
                return True
            except Exception:
                pass
        # Fallback: media key
        pyautogui.press("playpause")
        set_status("Play/Pause")
        return True

    if text in ("next", "next song", "skip", "next track"):
        if HAS_SPOTIFY and sp:
            try:
                sp.next_track()
                set_status("Spotify: next track")
                return True
            except Exception:
                pass
        pyautogui.press("nexttrack")
        set_status("Next track")
        return True

    if text in ("previous", "previous song", "go back", "last song", "previous track"):
        if HAS_SPOTIFY and sp:
            try:
                sp.previous_track()
                set_status("Spotify: previous track")
                return True
            except Exception:
                pass
        pyautogui.press("prevtrack")
        set_status("Previous track")
        return True

    if text.startswith("play ") and HAS_SPOTIFY and sp:
        query = text[5:].strip()
        try:
            results = sp.search(q=query, limit=1, type="track")
            if results["tracks"]["items"]:
                track = results["tracks"]["items"][0]
                sp.start_playback(uris=[track["uri"]])
                set_status(f"Playing: {track['name']} - {track['artists'][0]['name']}")
                return True
        except Exception as e:
            set_status(f"Spotify search failed: {e}")

    if text in ("what's playing", "what song is this", "current song"):
        if HAS_SPOTIFY and sp:
            try:
                current = sp.current_playback()
                if current and current["item"]:
                    track = current["item"]
                    set_status(f"Now playing: {track['name']} - {track['artists'][0]['name']}")
                else:
                    set_status("Nothing playing")
                return True
            except Exception:
                pass

    # --- Screenshot ---
    if text in ("screenshot", "take screenshot", "take a screenshot"):
        pyautogui.hotkey("win", "shift", "s")
        set_status("Screenshot tool opened")
        return True

    return False


# ---------- Helpers ----------
def open_app(name):
    """Try to open an application by name."""
    app_map = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "firefox": "firefox",
        "edge": "msedge",
        "microsoft edge": "msedge",
        "notepad": "notepad",
        "calculator": "calc",
        "file explorer": "explorer",
        "explorer": "explorer",
        "spotify": "spotify",
        "discord": "discord",
        "terminal": "wt",
        "command prompt": "cmd",
        "cmd": "cmd",
        "task manager": "taskmgr",
        "settings": "ms-settings:",
        "paint": "mspaint",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
    }

    cmd = app_map.get(name, name)
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception:
        # Try searching via Start menu
        pyautogui.hotkey("win")
        time.sleep(0.5)
        pyautogui.typewrite(name, interval=0.05)
        time.sleep(0.5)
        pyautogui.press("enter")


def press_key(key_name):
    """Press a key by name."""
    key_map = {
        "space": "space",
        "tab": "tab",
        "escape": "escape",
        "esc": "escape",
        "delete": "delete",
        "home": "home",
        "end": "end",
        "page up": "pageup",
        "page down": "pagedown",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    }
    key = key_map.get(key_name, key_name)
    try:
        pyautogui.press(key)
    except Exception:
        pass


def change_volume(delta):
    """Change system volume by delta (-1.0 to 1.0)."""
    if HAS_VOLUME and volume:
        current = volume.GetMasterVolumeLevelScalar()
        new_vol = max(0.0, min(1.0, current + delta))
        volume.SetMasterVolumeLevelScalar(new_vol, None)
    else:
        # Fallback: media keys
        if delta > 0:
            pyautogui.press("volumeup", presses=5)
        else:
            pyautogui.press("volumedown", presses=5)


def toggle_mute(force_unmute=False):
    """Toggle or force-unmute system volume."""
    if HAS_VOLUME and volume:
        if force_unmute:
            volume.SetMute(0, None)
        else:
            current = volume.GetMute()
            volume.SetMute(0 if current else 1, None)
    else:
        pyautogui.press("volumemute")


# ---------- Voice Listener Thread ----------
class VoiceController:
    def __init__(self, status_callback=None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.running = False
        self.thread = None
        self.status_callback = status_callback
        self.last_status = ""
        self.last_status_time = 0

        # Adjust for ambient noise
        print("[Voice] Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[Voice] Microphone ready. Listening...")

    def start(self):
        """Start listening in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the listener."""
        self.running = False

    def get_status(self):
        """Get the latest status message (clears after 2 seconds)."""
        if time.time() - self.last_status_time > 2:
            return ""
        return self.last_status

    def _set_status(self, msg):
        self.last_status = msg
        self.last_status_time = time.time()

    def _listen_loop(self):
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"[Voice] Heard: '{text}'")
                    matched = handle_command(text, self._set_status)
                    if not matched:
                        self._set_status(f"Unknown: '{text}'")
                except sr.UnknownValueError:
                    pass  # Couldn't understand — ignore
                except sr.RequestError as e:
                    self._set_status(f"API error: {e}")

            except sr.WaitTimeoutError:
                pass  # No speech detected — loop back
            except Exception as e:
                print(f"[Voice] Error: {e}")
                time.sleep(1)

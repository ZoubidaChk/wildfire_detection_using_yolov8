"""
Beep + terminal log + cooldown. No GPIO — this is a laptop.
"""
import sys
import time
import threading

import config


def _beep():
    """Cross-platform best-effort beep. Silent failure is fine."""
    try:
        if sys.platform.startswith('win'):
            import winsound
            winsound.Beep(1200, 400)  # 1.2 kHz, 400 ms
        elif sys.platform == 'darwin':
            import os
            os.system('afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 &')
        else:
            # Linux: terminal bell. Most desktops play a sound for it.
            print('\a', end='', flush=True)
    except Exception:
        pass


class AlertSystem:
    """Cooldown-gated alert. trigger() returns immediately."""

    def __init__(self):
        self._last_alert_time = 0.0
        self._lock = threading.Lock()

    def trigger(self):
        now = time.time()
        with self._lock:
            if now - self._last_alert_time < config.ALERT_COOLDOWN_SEC:
                return
            self._last_alert_time = now
        threading.Thread(target=self._fire, daemon=True).start()

    def _fire(self):
        print(f"[ALERT] Fire detected at {time.strftime('%H:%M:%S')}")
        if config.PLAY_SOUND:
            _beep()

    @property
    def in_cooldown(self) -> bool:
        return (time.time() - self._last_alert_time) < config.ALERT_COOLDOWN_SEC

    def cleanup(self):
        # Nothing to release — kept for API symmetry with the Pi version.
        pass

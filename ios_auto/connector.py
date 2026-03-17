"""iOS device connection management using WDA."""

from __future__ import annotations

import time
from typing import Optional

import wda


class DeviceConnector:
    """Manages connection to iOS device via WDA (WebDriverAgent)."""

    def __init__(
        self,
        url: str = "http://localhost:8100",
        udid: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize device connector.

        Args:
            url: WDA server URL (default: http://localhost:8100)
            udid: Device UDID for USB connection (optional)
            timeout: Connection timeout in seconds
        """
        self.url = url
        self.udid = udid
        self.timeout = timeout
        self._client: Optional[wda.Client] = None
        self._session: Optional[wda.Session] = None

    @property
    def client(self) -> wda.Client:
        """Get WDA client, creating if needed."""
        if self._client is None:
            if self.udid:
                self._client = wda.USBClient(self.udid)
            else:
                self._client = wda.Client(self.url)
        return self._client

    @property
    def session(self) -> wda.Session:
        """Get WDA session, creating if needed."""
        if self._session is None:
            self._session = self.client.session()
        return self._session

    def connect(self, retry: int = 3, retry_delay: float = 2.0) -> bool:
        """Connect to device with retries.

        Args:
            retry: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if connection successful
        """
        for attempt in range(retry):
            try:
                # Test connection
                _ = self.client.status()
                # Create session
                _ = self.session
                print(f"Connected to device: {self.udid or self.url}")
                return True
            except Exception as e:
                if attempt < retry - 1:
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to connect after {retry} attempts: {e}")
                    return False
        return False

    def disconnect(self) -> None:
        """Disconnect from device."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def get_screen_size(self) -> tuple[int, int]:
        """Get device screen size.

        Returns:
            (width, height) in pixels
        """
        info = self.session.window_size()
        return int(info.width), int(info.height)

    def __enter__(self) -> "DeviceConnector":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()

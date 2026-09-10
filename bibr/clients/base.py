import logging

import httpx

logger = logging.getLogger(__name__)


class BaseClient:
    """
    Base client for interacting with external services via HTTP.
    Provides common functionality for initialization, connection pooling, and resource cleanup.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
    ):
        """
        Initialize the base client.

        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds
            max_connections: Maximum number of connections in the pool
            max_keepalive_connections: Maximum number of keep-alive connections
        """
        if not base_url:
            raise ValueError("Base URL must be provided")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=limits,
            http2=True,
        )

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        await self.close()

    async def close(self):
        """Close the HTTP client and cleanup resources."""
        if not self.client.is_closed:
            await self.client.aclose()

    def __del__(self):
        # __init__ may have raised before assigning self.client, so guard
        # access — __del__ must never raise.
        client = getattr(self, "client", None)
        if client is not None and not client.is_closed:
            logger.warning(
                "Unclosed %s (base_url=%s). Use 'async with' or call await .close().",
                type(self).__name__,
                getattr(self, "base_url", "<unknown>"),
            )

    async def ping(self, endpoint: str = "/health") -> tuple[bool, int]:
        """
        Check if the service is available.

        Args:
            endpoint: Health check endpoint relative to base_url. Defaults to "/health".

        Returns:
            Tuple of (is_available, status_code)
        """
        try:
            # usage of self.client.get with absolute path (since base_url is set in client)
            # or relative path if we rely on httpx base_url support.
            # Using relative path here as httpx.AsyncClient(base_url=...) handles it.
            # However, to be safe with varying implementations of ping, we'll strip leading slash.
            target = endpoint.lstrip("/")
            response = await self.client.get(target, timeout=10.0)
            is_available = response.status_code == 200
            return is_available, response.status_code
        except httpx.RequestError as e:
            logger.error(f"Failed to ping service at {self.base_url}: {e}")
            return False, 0

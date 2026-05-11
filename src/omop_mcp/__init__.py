from . import server
from .client import OMOPHubClient
from .service import MCPAgentService


def main():
    """Main entry point for the package."""
    server.main()


__all__ = ["main", "server", "OMOPHubClient", "MCPAgentService"]

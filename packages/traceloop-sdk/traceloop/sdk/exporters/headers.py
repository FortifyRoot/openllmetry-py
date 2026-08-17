from typing import Dict


def grpc_metadata_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Normalize HTTP-style headers for gRPC metadata.

    gRPC metadata keys must be lowercase. The public SDK accepts conventional
    HTTP casing (for example ``Authorization``), so every gRPC signal exporter
    must normalize keys before passing them to grpcio.
    """
    return {str(key).lower(): str(value) for key, value in headers.items()}

"""Self-signed HTTPS certificate for the classroom server.

Browsers only allow the webcam on https:// pages (or on localhost), so students' phones
need HTTPS. We make a certificate for this computer's LAN address with openssl. Browsers
will warn that it is not trusted; each student accepts it once ("visit this website").
"""

import subprocess

from .data import DATA_DIR

TLS_DIR = DATA_DIR / "tls"


def ensure_cert(ip):
    """Return (cert, key) paths for `ip`, creating them if missing or made for another address."""
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    cert, key, tag = TLS_DIR / "cert.pem", TLS_DIR / "key.pem", TLS_DIR / "ip.txt"
    if cert.exists() and key.exists() and tag.exists() and tag.read_text().strip() == ip:
        return cert, key
    san = f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes",
        "-keyout", str(key), "-out", str(cert), "-days", "825",
        "-subj", f"/CN=Fly Brain ({ip})", "-addext", san,
        "-addext", "extendedKeyUsage=serverAuth", "-addext", "basicConstraints=CA:FALSE",
    ], check=True, capture_output=True)
    tag.write_text(ip)
    return cert, key

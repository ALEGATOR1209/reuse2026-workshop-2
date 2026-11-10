"""Self-signed HTTPS certificate for the classroom server.

Browsers only allow the webcam on https:// pages (or on localhost), so students' phones
need HTTPS. We make a certificate for this computer's LAN address in Python (with the
`cryptography` package, so it works the same on Windows, macOS and Linux). Browsers
will warn that it is not trusted; each student accepts it once ("visit this website").
"""

import datetime as dt
import ipaddress

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from .data import DATA_DIR

TLS_DIR = DATA_DIR / "tls"
VALID_DAYS = 825   # the longest validity Apple devices accept for a TLS certificate


def ensure_cert(ip, tls_dir=TLS_DIR):
    """Return (cert, key) paths for `ip`, creating them if missing or made for another address."""
    tls_dir.mkdir(parents=True, exist_ok=True)
    cert, key, tag = tls_dir / "cert.pem", tls_dir / "key.pem", tls_dir / "ip.txt"
    if cert.exists() and key.exists() and tag.exists() and tag.read_text(encoding="utf-8").strip() == ip:
        return cert, key

    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Fly Brain ({ip})")])
    now = dt.datetime.now(dt.timezone.utc)
    crt = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)   # self-signed
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))   # tolerate a slightly wrong clock
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName([
            x509.IPAddress(ipaddress.ip_address(ip)),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            x509.DNSName("localhost"),
        ]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(pkey.public_key()), critical=False)
        .sign(pkey, hashes.SHA256())
    )
    key.write_bytes(pkey.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                       serialization.NoEncryption()))
    cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    tag.write_text(ip, encoding="utf-8")
    return cert, key

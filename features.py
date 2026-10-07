
import re
from urllib.parse import urlparse

ipv4_pattern = re.compile(
    r'^('
    r'(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}'
    r'(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$'
)
digit_in_word = re.compile(r'[a-z][01]+[a-z]')
common_tlds = {"com", "org", "net", "edu", "gov"}


def check_ip(url):
    try:
        hostname = urlparse(url).hostname
        if hostname and ipv4_pattern.match(hostname):
            return 1
        return 0
    except Exception:
        return 0


def check_lookalike(url):
    try:
        if "://" not in url:
            url = "http://" + url
        hostname = urlparse(url).hostname
        if not hostname:
            return 0
        if not hostname.isascii():
            return 1
        if "xn--" in hostname:
            return 1
        if not ipv4_pattern.match(hostname):
            labels = hostname.split(".")[:-1]
            if any(digit_in_word.search(label) for label in labels):
                return 1
        return 0
    except Exception:
        return 0


def get_tld(url):
    try:
        if "://" not in url:
            url = "http://" + url
        hostname = urlparse(url).hostname
        if not hostname or ipv4_pattern.match(hostname):
            return ""
        parts = hostname.lower().split(".")
        if len(parts) < 2:
            return ""
        return parts[-1]
    except Exception:
        return ""


def extract_features(url, feature_cols):
    """Return a one-row list of feature values in the model's column order."""
    tld = get_tld(url)
    values = {
        "url_length": len(url),
        "num_digits": len(re.findall(r"\d", url)),
        "num_letters": len(re.findall(r"[A-Za-z]", url)),
        "num_special_chars": len(re.findall(r"[^A-Za-z0-9]", url)),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "has_ip": check_ip(url),
        "has_lookalike": check_lookalike(url),
        "tld_length": len(tld),
        "is_common_tld": int(tld in common_tlds),
    }
    return [values[c] for c in feature_cols], values


# ---------------------------------------------------------------------------
# App-only helpers (validation and human-readable signals).
# These do NOT feed the model; the model still uses extract_features() above.
# ---------------------------------------------------------------------------
LONG_URL_CHARS = 75   # display threshold only; tune from df["url_length"].quantile(0.95)
_label_ok = re.compile(r"^[a-z0-9_-]+$")
_lookalike_display = re.compile(r"[a-z][01]+([a-z]|$)")  # also catches trailing digit, e.g. paypa1


def validate_url(raw):
    """Return (ok, cleaned_url, message). Message explains how to fix bad input."""
    url = (raw or "").strip()
    if not url:
        return False, url, "Enter a URL to classify."
    if len(url) > 2048:
        return False, url, "That URL is longer than 2,048 characters. Paste a shorter one."
    if re.search(r"\s", url):
        return False, url, "A URL cannot contain spaces. Remove them and try again."

    full = url if "://" in url else "http://" + url
    try:
        parsed = urlparse(full)
        host = parsed.hostname
        parsed.port  # raises ValueError for a bad port
    except ValueError:
        return False, url, "This URL is not well formed (check the port number and brackets)."

    if parsed.scheme not in ("http", "https"):
        return False, url, "Only http and https URLs are supported."
    if not host:
        return False, url, "No domain found. Enter something like example.com/page."
    if ipv4_pattern.match(host):
        return True, url, ""
    if "." not in host:
        return False, url, "The domain needs a dot, like example.com."

    labels = host.split(".")
    for label in labels:
        if not label:
            return False, url, "The domain has an empty part (two dots in a row or a trailing dot)."
        if label.isascii() and (not _label_ok.match(label) or label.startswith("-") or label.endswith("-")):
            return False, url, f"'{label}' is not a valid domain part."
    if labels[-1].isdigit():
        return False, url, "The ending of the domain cannot be a number. If this is an IP address, check each part is 0-255."
    return True, url, ""


def get_indicators(url, values):
    """Return a list of (status, text). status is 'warn', 'info' or 'ok'."""
    full = url if "://" in url else "http://" + url
    host = urlparse(full).hostname or ""
    tld = get_tld(url)
    out = []

    if values["url_length"] > LONG_URL_CHARS:
        out.append(("warn", f"Long URL: {values['url_length']} characters (flagged above {LONG_URL_CHARS})."))
    else:
        out.append(("ok", f"URL length is normal ({values['url_length']} characters)."))

    if ipv4_pattern.match(host):
        out.append(("warn", f"IP address used instead of a domain name ({host})."))
    else:
        out.append(("ok", "Uses a domain name, not an IP address."))

    labels = host.split(".")[:-1] if "." in host else []
    lookalike = bool(values["has_lookalike"]) or any(_lookalike_display.search(l) for l in labels)
    if lookalike:
        out.append(("warn", "Possible look-alike characters (non-ASCII, punycode, or a 0/1 inside a word)."))
    else:
        out.append(("ok", "No look-alike characters detected."))

    if not tld:
        pass
    elif tld in common_tlds:
        out.append(("ok", f"Common ending (.{tld})."))
    else:
        out.append(("info", f"Uncommon ending (.{tld}). Many legitimate sites use country endings, so this is a weak signal."))
    return out

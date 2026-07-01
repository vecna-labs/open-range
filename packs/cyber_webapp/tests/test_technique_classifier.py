from __future__ import annotations

from cyber_webapp.technique_classifier import classify_technique


def test_nmap_returns_T1046():
    result = classify_technique("nmap -sV 192.168.1.1")
    assert result == "T1046", f"Expected T1046 but got {result}"


def test_sqlmap_returns_T1190():
    result = classify_technique("sqlmap -u http://target.com")
    assert result == "T1190", f"Expected T1190 but got {result}"


def test_ssh_returns_T1078():
    result = classify_technique("ssh admin@192.168.1.1")
    assert result == "T1078", f"Expected T1078 but got {result}"


def test_unknown_command_returns_none():
    result = classify_technique("ls -la")
    assert result is None, f"Expected None but got {result}"


def test_hydra_returns_T1110():
    result = classify_technique("hydra -l admin -P passwords.txt")
    assert result == "T1110", f"Expected T1110 but got {result}"

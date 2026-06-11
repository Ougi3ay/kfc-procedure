# Security Policy

## Supported versions

The project is currently in early development. Security fixes are generally applied to the latest released version.

| Version | Supported |
|---|---|
| `0.1.x` | Yes |
| Older versions | No |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

To report a vulnerability, use one of the following methods:

1. Open a private security advisory on GitHub, if available for this repository.
2. Contact the maintainer privately using the email listed in `pyproject.toml`.

Include as much information as possible:

- affected version,
- operating system and Python version,
- minimal reproduction steps,
- impact of the vulnerability,
- whether the vulnerability is already public,
- suggested fix, if known.

## Response expectations

The maintainer will try to acknowledge valid security reports within a reasonable time. Response time may vary because this is a research-oriented open-source project.

## Public disclosure

Please allow the maintainer time to investigate and prepare a fix before publicly disclosing the vulnerability.

## Scope

Security reports may include:

- unsafe file handling,
- dependency-related vulnerabilities,
- arbitrary code execution risks,
- insecure defaults,
- exposure of sensitive information,
- packaging or release-process issues.

Reports about general bugs that do not create a security risk should be opened as normal GitHub issues.

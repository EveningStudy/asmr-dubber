[中文](SECURITY.md) | English

# Security

Report vulnerabilities privately through the repository's **Security → Report a vulnerability**. Include affected version/commit, OS/install method, minimal reproduction, impact/preconditions and sanitized evidence. Do not publish exploit details, credentials, private media or identifying paths. Do not access other people's systems to demonstrate a bug.

Security fixes target the default branch and latest release. Upstream runtime/model/service issues should also be reported upstream.

## Threat model

This is a single-user local tool, not a public multi-tenant media service. Default UI binding is loopback; the OS account protects the application directory. Local projects/models are trusted inputs. Installers execute pinned third-party code; external services are separate trust boundaries.

Keys are **plaintext** in `.asmr-dubber/config/secrets.json`. Protect directory permissions and backups; never publish the data/config directory. Use scoped keys and rotate immediately after exposure. Keys must not enter project manifests, performance reports, subtitles or downloadable logs.

## Web and files

Non-loopback binding requires username/password; an unset password is generated and printed in the terminal. Public sharing tunnels are disabled. Media URLs only serve registered resources; writes validate session token/Origin and loopback checks Host; upload limit defaults to 20 GB. Authentication is not a reason to expose the development UI directly to the internet.

Untrusted media exercises parsers/codecs. Do not run as administrator/root. Path escape, symlink bypass and arbitrary file disclosure are security issues. Keep schema/revision validation enabled.

## Supply chain and privacy

Fixed artifact hashes detect corruption/substitution but do not prove upstream safety. Preserve pinned revisions, license files and immutable paths. Do not bypass verification or install random replacement DLLs.

Local multi-model review does not call an LLM. Script matching sends script/recognition text. Translation sends bounded text context. ASR APIs upload audio; external TTS/separation may upload references. Check authorization and provider retention/training policies. Output URLs do not inherit service credentials; redirects and response sizes are constrained.

Diagnostic script reports may contain private dialogue. Review attachments before sharing. Coordinate disclosure after fixes/mitigations are agreed; community conduct reports can use the same private entry point with an appropriate title.

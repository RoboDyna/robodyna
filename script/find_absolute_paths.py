#!/usr/bin/env python3
"""Report hardcoded absolute paths in tracked config files.

Several embodiment configs (``assets/embodiments/**/curobo*.yml``) are generated
per machine by ``script/update_embodiment_config_path.py``, which expands
``${ASSETS_PATH}`` in the ``*_tmp.yml`` templates into whatever absolute path
the repository lives at. Those generated files are committed, so a clone always
arrives carrying the *previous* committer's absolute paths -- CuRobo then fails
with a FileNotFoundError pointing at a home directory that does not exist here.

This script finds those paths and says which ones are broken on this machine, so
the fix (rerun update_embodiment_config_path.py) is obvious instead of arriving
as a stack trace.

Each path is reported as one of:

    ok       resolves inside this checkout
    FOREIGN  exists, but in a different directory tree -- nothing raises, yet the
             assets actually loaded are not the ones tracked here
    MISSING  does not exist on this machine

Usage:
    python script/find_absolute_paths.py            # scan the default config globs
    python script/find_absolute_paths.py --all      # scan every tracked text file
    python script/find_absolute_paths.py --broken   # only report paths missing here
    python script/find_absolute_paths.py --fix      # rerun the generator if broken

Exit code is 1 when any reported path is missing on this machine, so CI or a
setup script can gate on it.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

NC = '\033[0m'
BLUE = '\033[0;34m'
YELLOW = '\033[0;33m'
GREEN = '\033[0;32m'
RED = '\033[0;31m'


def print_color(message, color_code):
    print(f"{color_code}{message}{NC}")


# Config files that are known to carry generated absolute paths.
DEFAULT_GLOBS = [
    'assets/embodiments/**/*.yml',
    'task_config/**/*.yml',
]

# Absolute POSIX paths, but not the placeholder templates use.
PATH_RE = re.compile(r'(?<![\w$}])/(?:[\w.+-]+/)+[\w.+-]+')

# Files the generator owns, so --fix has something to offer.
EMBODIMENT_RE = re.compile(r'assets/embodiments/.*\.ya?ml$')

# Roots that never indicate a checkout-local path problem.
SYSTEM_PREFIXES = (
    '/usr/', '/bin/', '/sbin/', '/lib/', '/lib64/', '/etc/', '/opt/',
    '/proc/', '/sys/', '/dev/', '/var/', '/tmp/', '/run/',
)

# First path segments that name repo-internal trees. A leading-slash path
# starting with one of these is a repo-relative fragment, not a real absolute
# path -- some are output roots that only exist after a run.
REPO_OWNED_DIRS = {
    'assets', 'urdf', 'srdf', 'data', 'data_lerobot', 'task_config',
    'envs', 'script', 'policy', 'models', 'logs',
}

TEXT_SUFFIXES = ('.yml', '.yaml', '.json', '.py', '.sh', '.cfg', '.ini', '.toml', '.txt')


def repo_root():
    """Directory containing assets/embodiments, starting from this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    if os.path.isdir(os.path.join(root, 'assets', 'embodiments')):
        return root
    return os.getcwd()


def tracked_files(root):
    """Every tracked text file, or None when git is unavailable."""
    try:
        out = subprocess.run(
            ['git', '-C', root, 'ls-files', '-z'],
            capture_output=True, check=True,
        ).stdout.decode('utf-8', 'replace')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [p for p in out.split('\0') if p.endswith(TEXT_SUFFIXES)]


def candidate_files(root, scan_all):
    if scan_all:
        files = tracked_files(root)
        if files is None:
            print_color('Not a git checkout; falling back to the default globs', YELLOW)
        else:
            return files
    seen, files = set(), []
    for pattern in DEFAULT_GLOBS:
        for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
            rel = os.path.relpath(path, root)
            if rel not in seen and os.path.isfile(path):
                seen.add(rel)
                files.append(rel)
    return files


def classify(root, path):
    """Return one of 'local', 'foreign', 'missing', or None to skip.

    'local'   - resolves inside this checkout
    'foreign' - exists, but belongs to some other directory tree
    'missing' - does not exist on this machine

    A leading-slash fragment that names something inside the repo (``file_path:
    /assets/embodiments/piper``) is repo-relative by convention, not an absolute
    path, so it is skipped -- either because it already resolves under the root,
    or because its first segment is a directory this repo owns (which covers
    output roots like ``/data_lerobot/...`` that do not exist until a run
    produces them).
    """
    root = os.path.realpath(root) + os.sep
    if os.path.exists(os.path.join(root, path.lstrip('/'))):
        return None
    head = path.lstrip('/').split('/', 1)[0]
    if head in REPO_OWNED_DIRS:
        return None
    if not os.path.exists(path):
        return 'missing'
    return 'local' if os.path.realpath(path).startswith(root) else 'foreign'


def scan(root, files):
    """Yield (relpath, lineno, key, abspath, state) for each absolute path found."""
    for rel in files:
        full = os.path.join(root, rel)
        try:
            with open(full, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue
            for match in PATH_RE.findall(line):
                if match.startswith(SYSTEM_PREFIXES):
                    continue
                state = classify(root, match)
                if state is None:
                    continue
                key = stripped.split(':', 1)[0].strip() if ':' in stripped else ''
                yield rel, lineno, key, match, state


def run_generator(root):
    script = os.path.join(root, 'script', 'update_embodiment_config_path.py')
    if not os.path.isfile(script):
        print_color(f'Cannot find {script}', RED)
        return 1
    print_color(f'Running {os.path.relpath(script, root)} in {root}', BLUE)
    return subprocess.run([sys.executable, script], cwd=root).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all', action='store_true',
                        help='scan every tracked text file, not just the config globs')
    parser.add_argument('--broken', action='store_true',
                        help='only report paths that do not exist on this machine')
    parser.add_argument('--fix', action='store_true',
                        help='rerun update_embodiment_config_path.py when paths are broken')
    args = parser.parse_args()

    root = repo_root()
    print_color(f'Repository root: {root}', BLUE)

    files = candidate_files(root, args.all)
    print_color(f'Scanning {len(files)} file(s)...', BLUE)

    findings = list(scan(root, files))
    broken = [f for f in findings if f[4] == 'missing']
    foreign = [f for f in findings if f[4] == 'foreign']
    shown = broken if args.broken else findings

    if not shown:
        print_color('No hardcoded absolute paths found.' if not args.broken
                    else 'No broken absolute paths on this machine.', GREEN)
        return 0

    current = None
    for rel, lineno, key, path, state in shown:
        if rel != current:
            current = rel
            print()
            print_color(rel, BLUE)
        mark, color = {
            'local': ('ok     ', GREEN),
            'foreign': ('FOREIGN', YELLOW),
            'missing': ('MISSING', RED),
        }[state]
        label = f'{key}: ' if key else ''
        print_color(f'  {lineno:>5}  [{mark}]  {label}{path}', color)

    print()
    print(f'Total absolute paths: {len(findings)}')
    if foreign and not args.broken:
        print_color(f'Pointing outside this checkout: {len(foreign)}', YELLOW)
        print('  These resolve to a different directory tree, so nothing raises --')
        print('  but the assets in use are not the ones tracked here.')
    if broken:
        print_color(f'Missing on this machine: {len(broken)}', RED)
        generated = [f for f in broken if EMBODIMENT_RE.match(f[0])]
        if generated:
            print()
            print(f'{len(generated)} of these are in the generated embodiment configs.')
            print('To point them at this checkout, regenerate them from the templates:')
            print()
            print('    python script/update_embodiment_config_path.py')
            print()
            print('Leave the result uncommitted -- it is machine-local by design.')
        other = len(broken) - len(generated)
        if other:
            print()
            print(f'{other} are hardcoded elsewhere and have no generator; they need')
            print('editing by hand, or reading from a config or environment variable.')
        if args.fix:
            if not generated:
                print()
                print_color('Nothing for --fix to regenerate.', YELLOW)
                return 1
            print()
            return run_generator(root)
        return 1

    print_color('All absolute paths exist on this machine.', GREEN)
    return 0


if __name__ == '__main__':
    sys.exit(main())

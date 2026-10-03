"""Run a paper tool with the existing project Python environment."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    """Run a paper tool with the existing project Python environment."""
    directory = Path(__file__).resolve().parent
    choices = sorted(p.stem for p in directory.glob('*_codexgen.py')
                     if p.stem != 'paths_codexgen')
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m lattmc.lattconferencetools.run_codexgen'
              ' TOOL [ARGS]')
        print('Tools (the .py suffix is optional):')
        print('\n'.join('  ' + name for name in choices))
        return
    name = sys.argv.pop(1).removesuffix('.py')
    if name not in choices:
        raise SystemExit(f'Unknown paper tool: {name}')
    runpy.run_module('lattmc.lattconferencetools.' + name, run_name='__main__',
                         alter_sys=True)


if __name__ == '__main__':
    main()

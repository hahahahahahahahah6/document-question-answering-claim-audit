"""Deterministic offline claim scanning; no target code is executed."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

from .phrases import (DOCUMENTQUESTIONANSWERING, NODOCUMENTQUESTIONANSWERING,
                      GENERIC_NAMES, DONUT_CLASSES, LAYOUTLM_CLASSES)


def prose(text):
    output = []
    fenced = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            output.append('\n')
        elif not fenced:
            output.append(line.replace('`', ''))
    return ''.join(output)


def match_phrases(text):
    text = text.lower()
    def extract(names):
        nonlocal text
        hits = []
        for phrase in sorted(names, key=len, reverse=True):
            for match in list(re.finditer(re.escape(phrase), text)):
                hits.append({'phrase': phrase, 'offset': match.start()})
                text = text[:match.start()] + ' ' * len(phrase) + text[match.end():]
        return sorted(hits, key=lambda hit: hit['offset'])
    negative = extract(NODOCUMENTQUESTIONANSWERING)
    return extract(DOCUMENTQUESTIONANSWERING), negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    nodes = list(ast.walk(tree))
    shadow = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = (node.module or '') + '.' + alias.name

    def resolved(node):
        if isinstance(node, ast.Name):
            if node.id in shadow:
                return ''
            value = imports.get(node.id, node.id)
            return '' if value.split('.')[-1] in shadow else value
        if isinstance(node, ast.Attribute):
            if node.attr in shadow:
                return ''
            base = resolved(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    def key(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = key(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    bindings = {}
    def constructed(node):
        if not isinstance(node, ast.Call):
            return bindings.get(key(node), '')
        name = resolved(node.func)
        if name.endswith('.from_pretrained'):
            candidate = name.rsplit('.', 1)[0]
            if candidate.split('.')[-1] not in DONUT_CLASSES + LAYOUTLM_CLASSES:
                return ''
            name = candidate
        tail = name.split('.')[-1]
        if tail in DONUT_CLASSES + LAYOUTLM_CLASSES:
            return tail
        if name == 'transformers.pipeline':
            task = node.args[0] if node.args else None
            keywords = [kw.value for kw in node.keywords if kw.arg == 'task']
            if keywords:
                task = keywords[0]
            if isinstance(task, ast.Constant) and task.value == 'document-question-answering':
                return 'pipeline'
        return ''

    assignments = [n for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            kind = constructed(node.value)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                target_key = key(target)
                if target_key and kind and bindings.get(target_key) != kind:
                    bindings[target_key] = kind
                    changed = True
        if not changed:
            break
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    processor = any(constructed(n) == 'DonutProcessor' for n in calls)
    for node in calls:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in GENERIC_NAMES and func.attr not in shadow:
                return True
            if func.attr == 'generate' and processor and constructed(func.value) == 'VisionEncoderDecoderModel':
                return True
        elif isinstance(func, ast.Name):
            name = resolved(func)
            if func.id in imports and name.split('.')[-1] in GENERIC_NAMES:
                return True
        if constructed(func) in ['pipeline'] + LAYOUTLM_CLASSES:
            return True
    return False


SKIP = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env'}
POSITIVE_RULE = 'documented-document-question-answering-claim-false'
NEGATIVE_RULE = 'documented-no-document-question-answering-claim-false'


def audit(repo, ignore=(), verbose=False):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository must be an existing directory')
    files = []
    def visit(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or any(relative.startswith(prefix) for prefix in ignore):
                continue
            if path.is_dir():
                if path.name not in SKIP and not (path / 'pyvenv.cfg').exists():
                    visit(path)
            else:
                files.append(path)
    visit(root)
    has_evidence = False
    documents = []
    if verbose:
        print('Scope: ' + str(root), file=sys.stderr)
    for path in files:
        rel = path.relative_to(root)
        shipped = rel.as_posix() == 'README.md' or (rel.parts[0] == 'docs' and path.suffix == '.md')
        fixture = 'tests' in rel.parts[:-1]
        source = path.suffix == '.py'
        if not (shipped or fixture or source):
            continue
        text = path.read_text(encoding='utf-8')
        found = evidence(text) if source else False
        has_evidence |= found
        if shipped or fixture:
            documents.append((rel.as_posix(), shipped, match_phrases(prose(text))))
        if verbose:
            print('{}: evidence={}, prose={}'.format(rel, found, shipped or fixture), file=sys.stderr)
    findings = []
    shipped_positive = any(shipped and matches[0] for _, shipped, matches in documents)
    for path, shipped, (positive, negative) in documents:
        selections = []
        if not has_evidence and (shipped or not shipped_positive):
            selections.extend((hit, POSITIVE_RULE, 'HIGH' if shipped else 'MEDIUM') for hit in positive)
        if has_evidence and shipped:
            selections.extend((hit, NEGATIVE_RULE, 'HIGH') for hit in negative)
        for hit, rule, severity in selections:
            findings.append(dict(rule=rule, severity=severity, path=path, phrase=hit['phrase'],
                                 verdict='FAIL', message='Documentation and static call evidence disagree.'))
    return findings


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = Parser(description='Offline documentation claim auditor')
    parser.add_argument('--version', action='version', version='0.1.0')
    subs = parser.add_subparsers(dest='command', required=True)
    scan = subs.add_parser('audit')
    scan.add_argument('repo', nargs='?', default='.')
    scan.add_argument('--json', action='store_true')
    scan.add_argument('--verbose', action='store_true')
    scan.add_argument('--ignore', action='append', default=[])
    scan.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    try:
        args = parser.parse_args(argv)
        findings = audit(args.repo, args.ignore, args.verbose)
        if args.json:
            print(json.dumps(findings, indent=2))
        elif findings:
            for finding in findings:
                print('{rule} {severity} {path}: {message}'.format(**finding))
        else:
            print('PASS')
        return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
    except (ValueError, OSError, UnicodeError) as error:
        print('error: ' + str(error), file=sys.stderr)
        return 1

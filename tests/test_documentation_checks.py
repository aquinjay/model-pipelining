"""Documentation checks reject stale links/examples without executing code."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/check_documentation.py'
spec = importlib.util.spec_from_file_location('check_documentation', SCRIPT)
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


class DocumentationChecksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, text):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_inline_reference_image_and_encoded_links_and_duplicate_anchors(self):
        self.write('other page.md','# A `code` heading!\n## Repeat\n## Repeat\n')
        self.write('image.png','fixture')
        path = self.write('README.md','''# Start
[section](other%20page.md#a-code-heading)
[duplicate](<other page.md#repeat-1>)
![image](image.png)
[reference][Mixed Label]
[mixed label]: other%20page.md#repeat
[external](https://example.invalid/not-fetched)
''')
        self.assertEqual(docs.check_document(path,self.root)['links'],4)

    def test_missing_or_escaping_local_links_fail_with_location(self):
        for target in ('absent.md','../outside.md'):
            path = self.write('README.md',f'# Read\n[bad]({target})\n')
            with self.assertRaisesRegex(ValueError,'README.md:2: missing/escaping'):
                docs.check_document(path,self.root)

    def test_missing_local_and_same_file_anchors_are_detected(self):
        path = self.write('README.md','# Read\n[bad](#missing)\n')
        with self.assertRaisesRegex(ValueError,'missing local anchor'):
            docs.check_document(path,self.root)
        path.write_text('# Read\n[good](#read)\n')
        self.assertEqual(docs.check_document(path,self.root)['links'],1)

    def test_undefined_reference_links_are_detected_but_inline_code_is_ignored(self):
        path = self.write('README.md','`record["key"]["value"]`\n[bad][missing]\n')
        with self.assertRaisesRegex(ValueError,'undefined reference'):
            docs.check_document(path,self.root)
        path.write_text('`record["key"]["value"]`\n')
        self.assertEqual(docs.check_document(path,self.root)['links'],0)

    def test_python_examples_are_parsed_without_executing_valid_code(self):
        sentinel = self.root/'should-not-exist'
        path = self.write('README.md',f'```python\nopen({str(sentinel)!r}, "w").write("no")\n```\n')
        self.assertEqual(docs.check_document(path,self.root)['python'],1)
        self.assertFalse(sentinel.exists())
        path.write_text('```python\nif True\n    pass\n```\n')
        with self.assertRaisesRegex(ValueError,'invalid python example'):
            docs.check_document(path,self.root)

    def test_bash_examples_are_parsed_without_executing_commands(self):
        sentinel = self.root/'should-not-exist'
        path = self.write('README.md',f'```bash\ntouch {sentinel}\n```\n')
        self.assertEqual(docs.check_document(path,self.root)['bash'],1)
        self.assertFalse(sentinel.exists())
        path.write_text('```bash\nif true; then\n```\n')
        with self.assertRaisesRegex(ValueError,'invalid bash example'):
            docs.check_document(path,self.root)

    def test_fenced_link_examples_are_ignored_and_unclosed_fences_fail(self):
        path = self.write('README.md','~~~text\n[example](does-not-exist.md)\n~~~\n')
        self.assertEqual(docs.check_document(path,self.root)['links'],0)
        path.write_text('```python\nprint("incomplete")\n')
        with self.assertRaisesRegex(ValueError,'Unclosed fenced block'):
            docs.check_document(path,self.root)

    def test_discovery_includes_new_docs_and_excludes_ignored_local_artifacts(self):
        self.write('.gitignore','runs/*\n!runs/README.md\n.venv/\n')
        self.write('README.md','# Root\n');self.write('docs/new.md','# New\n')
        self.write('runs/README.md','# Runs\n');self.write('runs/generated/report.md','local output')
        self.write('.venv/README.md','dependency documentation')
        subprocess.run(['git','init','-q',str(self.root)],check=True,capture_output=True)
        subprocess.run(['git','-C',str(self.root),'add','README.md'],check=True,capture_output=True)
        found = {p.relative_to(self.root).as_posix() for p in docs.documentation_paths(self.root)}
        self.assertEqual(found,{'README.md','docs/new.md','runs/README.md'})

    def test_cli_returns_nonzero_for_a_stale_local_link(self):
        path = self.write('README.md','# Read\n[bad](missing.md)\n')
        command = [sys.executable,str(SCRIPT),'--root',str(self.root),path.name]
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('missing/escaping local link',result.stderr)
        path.write_text('# Read\n')
        result = subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('1 Markdown files',result.stdout)

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from document_question_answering_claim_audit import phrases
from document_question_answering_claim_audit.audit import evidence, match_phrases, prose, audit, main

EXPECTED_POSITIVE = ['document question answering', 'document-question-answering', 'documentquestionanswering', 'document question answering model', 'document question answering models', 'document question answering system', 'document question answering systems', 'document question answering feature', 'document question answering capability', 'document question answering support', 'supports document question answering', 'document question answering tool', 'document question answering api', 'document question answering pipeline', 'document question answering algorithm', 'document question answering library', 'document question answering framework', 'document question answering service', 'document question answering function', 'document question answering engine', 'document question answering engines', 'document question answerer', 'document question answerers', 'answer questions about documents', 'answers questions about documents', 'answering questions about documents', 'answer questions from documents', 'answers questions from documents', 'ask questions about documents', 'document qa', 'document question-answer', 'docvqa', 'document visual question answering', 'document understanding question answering', 'question answering over documents', 'question answering on documents', 'answer document questions', 'answers document questions', 'document question answering task', 'document question answering benchmark', 'ask questions of documents', 'asks questions of documents', 'query documents with questions', 'question answering for documents', 'document-based question answering', 'document based question answering', 'visual document question answering', 'answer questions on documents', 'document qa model', 'document qa system']
EXPECTED_NEGATIVE = ['does not support document question answering', 'do not support document question answering', "doesn't support document question answering", "don't support document question answering", 'no document question answering', 'without document question answering', 'never performs document question answering', 'does not perform document question answering', 'document question answering not supported', 'document question answering unsupported', 'no document question answering support', 'lacks document question answering', 'missing document question answering', 'document question answering disabled', 'no document question answering capability', 'without document question answering support', 'does not provide document question answering', 'no document question answering model', 'document question answering not available', 'does not use document question answering', 'no document question answerer', 'without document question answerer', 'does not support document question answerer', 'no document qa']
EXPECTED_GENERIC = ['answer_document_question', 'document_question_answering', 'answer_question_from_document', 'document_qa', 'doc_qa', 'answer_doc_question', 'extract_answer_from_document', 'document_question_answer', 'answer_questions_from_document', 'query_document', 'document_query_answering', 'ask_document', 'ask_doc', 'document_answer_extraction', 'answer_from_document', 'doc_question_answering', 'document_question_answerer', 'answer_document_questions', 'extract_document_answer', 'document_visual_question_answering', 'docvqa', 'document_qa_answer', 'answer_pdf_question', 'pdf_question_answering']
EXPECTED_DONUT = ['DonutProcessor', 'VisionEncoderDecoderModel']
EXPECTED_LAYOUTLM = ['LayoutLMForQuestionAnswering', 'LayoutLMv2ForQuestionAnswering', 'LayoutLMv3ForQuestionAnswering']


class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(phrases.DOCUMENTQUESTIONANSWERING, EXPECTED_POSITIVE)
        self.assertEqual(phrases.NODOCUMENTQUESTIONANSWERING, EXPECTED_NEGATIVE)

    def test_generic_oracle(self):
        self.assertEqual(phrases.GENERIC_NAMES, EXPECTED_GENERIC)

    def test_dedicated_oracle(self):
        self.assertEqual(phrases.DONUT_CLASSES, EXPECTED_DONUT)
        self.assertEqual(phrases.LAYOUTLM_CLASSES, EXPECTED_LAYOUTLM)

    def fixture(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        return root

    def cli(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                code = main(list(args))
            except SystemExit as exit_error:
                code = exit_error.code
        return code, output.getvalue(), errors.getvalue()

    def test_all_phrases(self):
        for phrase in EXPECTED_POSITIVE:
            with self.subTest(phrase=phrase):
                self.assertEqual(match_phrases(phrase), ([{'phrase': phrase, 'offset': 0}], []))
        for phrase in EXPECTED_NEGATIVE:
            with self.subTest(phrase=phrase):
                self.assertEqual(match_phrases(phrase), ([], [{'phrase': phrase, 'offset': 0}]))

    def test_pipeline_pairs(self):
        for import_line, name in [('import transformers', 'transformers.pipeline'),
                                  ('import transformers as t', 't.pipeline'),
                                  ('from transformers import pipeline', 'pipeline'),
                                  ('from transformers import pipeline as p', 'p')]:
            for task in ['"document-question-answering"', 'task="document-question-answering"']:
                for body in ['m={name}({task})\nm(image, question="q")', '{name}({task})(image, question="q")']:
                    source = import_line + '\n' + body.format(name=name, task=task)
                    with self.subTest(source=source):
                        self.assertTrue(evidence(source))

    def test_donut_triples(self):
        for prefix, processor, model in [('', 'DonutProcessor', 'VisionEncoderDecoderModel'),
                ('import transformers as t\n', 't.DonutProcessor', 't.VisionEncoderDecoderModel'),
                ('from transformers import DonutProcessor as P, VisionEncoderDecoderModel as M\n', 'P', 'M')]:
            for constructor in ['()', '.from_pretrained("x")']:
                for call in ['m={model}{constructor}\nm.generate()', '{model}{constructor}.generate()']:
                    source = prefix + processor + constructor + '\n' + call.format(model=model, constructor=constructor)
                    with self.subTest(source=source):
                        self.assertTrue(evidence(source))

    def test_layout_pairs(self):
        for name in EXPECTED_LAYOUTLM:
            for prefix, constructor in [('', name), ('import models as m\n', 'm.' + name),
                    ('from models import ' + name + ' as M\n', 'M')]:
                for suffix in ['()', '.from_pretrained("x")']:
                    for call in ['m={constructor}{suffix}\nm()', '{constructor}{suffix}()']:
                        source = prefix + call.format(constructor=constructor, suffix=suffix)
                        with self.subTest(source=source):
                            self.assertTrue(evidence(source))

    def test_generic_routes(self):
        for name in EXPECTED_GENERIC:
            for source in ['x.' + name + '()', 'factory().' + name + '()',
                    'class Object: pass\nObject.' + name + '()',
                    'from api import ' + name + '\n' + name + '()',
                    'from api import ' + name + ' as f\nf()']:
                with self.subTest(source=source):
                    self.assertTrue(evidence(source))

    def test_assignment_aliases(self):
        self.assertTrue(evidence('m=LayoutLMForQuestionAnswering()\nn=m\nn()'))
        self.assertTrue(evidence('self.m: object = LayoutLMForQuestionAnswering()\nself.m()'))

    def test_rejections(self):
        sources = ['import transformers, torch, pdfplumber, pytesseract',
                   'from transformers import pipeline, DonutProcessor, VisionEncoderDecoderModel, LayoutLMForQuestionAnswering',
                   '"document question answering"\n# document qa',
                   'x.answer_document_question', 'answer_document_question()',
                   'subprocess.run(["document-question-answering", "x"])',
                   'm.generate()', 'm()', 'DonutProcessor()',
                   'VisionEncoderDecoderModel()', 'VisionEncoderDecoderModel().generate()',
                   'DonutProcessor()\nm.generate()',
                   'DonutProcessor()\nVisionEncoderDecoderModel()',
                   'DonutProcessor()\nx.generate()',
                   'import transformers\ntransformers.pipeline("document-question-answering")',
                   'pipeline("document-question-answering")()',
                   'import transformers\ntransformers.pipeline.from_pretrained("document-question-answering")()',
                   'import other\nother.pipeline("document-question-answering")()',
                   'pytesseract.image_to_string(image)', 'easyocr.Reader(["en"]).readtext(image)',
                   'PaddleOCR().ocr(image)', 'pdfplumber.open("x")', 'PyPDF2.PdfReader("x")',
                   'BertForQuestionAnswering()()', 'DistilBertForQuestionAnswering()()',
                   'BlipForQuestionAnswering()()', 'TapasForQuestionAnswering()()']
        for task in ['question-answering', 'visual-question-answering', 'table-question-answering', 'image-classification']:
            sources.append('from transformers import pipeline\npipeline(' + repr(task) + ')()')
        for name in EXPECTED_LAYOUTLM:
            sources.extend([name + '()', name + '.from_pretrained("x")'])
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_shadowing(self):
        sources = [
            'from api import answer_document_question\ndef answer_document_question(): pass\nanswer_document_question()',
            'from api import answer_document_question as f\nclass answer_document_question: pass\nf()',
            'def answer_document_question(): pass\nx.answer_document_question()',
            'from transformers import pipeline\ndef pipeline(): pass\npipeline("document-question-answering")()',
            'import transformers as t\nclass pipeline: pass\nt.pipeline("document-question-answering")()',
            'from transformers import DonutProcessor as P\ndef DonutProcessor(): pass\nP()\nVisionEncoderDecoderModel().generate()',
            'class VisionEncoderDecoderModel: pass\nDonutProcessor()\nVisionEncoderDecoderModel().generate()',
            'class LayoutLMForQuestionAnswering: pass\nLayoutLMForQuestionAnswering()()',
            'import transformers as t\ndef t(): pass\nt.pipeline("document-question-answering")()']
        for name in EXPECTED_GENERIC:
            sources.append('class ' + name + ': pass\nx.' + name + '()')
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_split_modules(self):
        for first, second in [
            ('from transformers import pipeline\nm=pipeline("document-question-answering")', 'm()'),
            ('DonutProcessor()', 'VisionEncoderDecoderModel().generate()'),
            ('DonutProcessor()\nm=VisionEncoderDecoderModel()', 'm.generate()'),
            ('m=LayoutLMForQuestionAnswering()', 'm()')]:
            root = self.fixture({'README.md': 'document qa', 'a.py': first, 'b.py': second})
            self.assertEqual(audit(root)[0]['severity'], 'HIGH')

    def test_span_removal(self):
        root = self.fixture({'README.md': 'does not support document question answering'})
        self.assertEqual(audit(root), [])

    def test_longest_first(self):
        for phrase in ['document question answering model', 'document qa model']:
            self.assertEqual(match_phrases(phrase)[0], [{'phrase': phrase, 'offset': 0}])
        self.assertEqual(len(match_phrases('document qa model document qa model')[0]), 2)

    def test_case_and_collision(self):
        for text in ['DOCUMENT QUESTION ANSWERING', 'document qa review process']:
            root = self.fixture({'README.md': text})
            self.assertEqual(audit(root)[0]['severity'], 'HIGH')

    def test_fenced_and_inline(self):
        self.assertEqual(match_phrases(prose('```text\ndocument qa\n```'))[0], [])
        self.assertEqual(len(match_phrases(prose('`document qa`'))[0]), 1)
        self.assertEqual(match_phrases('document-question answering')[0], [])

    def test_cli_good(self):
        root = self.fixture({'README.md': 'document qa', 'a.py': 'x.document_qa()'})
        self.assertEqual(self.cli('audit', str(root))[0:2], (0, 'PASS\n'))

    def test_cli_bad_json(self):
        root = self.fixture({'README.md': 'document qa'})
        code, output, _ = self.cli('audit', str(root), '--json')
        self.assertEqual(code, 2)
        finding = json.loads(output)[0]
        self.assertEqual(finding['rule'], 'documented-document-question-answering-claim-false')
        self.assertEqual(finding['severity'], 'HIGH')
        self.assertEqual(finding['verdict'], 'FAIL')
        self.assertTrue(finding['message'])

    def test_cli_test_only(self):
        root = self.fixture({'tests/fixture.md': 'document qa'})
        self.assertEqual(self.cli('audit', str(root))[0], 0)
        self.assertEqual(self.cli('audit', str(root), '--fail-on', 'medium')[0], 2)
        self.assertEqual(audit(root)[0]['severity'], 'MEDIUM')

    def test_cli_denial(self):
        root = self.fixture({'README.md': 'no document qa', 'a.py': 'x.document_qa()'})
        code, output, _ = self.cli('audit', str(root), '--json')
        self.assertEqual(code, 2)
        finding = json.loads(output)[0]
        self.assertEqual(finding['rule'], 'documented-no-document-question-answering-claim-false')
        self.assertEqual(finding['severity'], 'HIGH')

    def test_cli_no_claim(self):
        root = self.fixture({'README.md': 'Hello'})
        self.assertEqual(self.cli('audit', str(root))[0], 0)

    def test_cli_version(self):
        self.assertEqual(self.cli('--version')[0:2], (0, '0.1.0\n'))

    def test_cli_ignore(self):
        root = self.fixture({'docs/draft.md': 'document qa', 'docs/tmp.md': 'document qa'})
        self.assertEqual(self.cli('audit', str(root), '--ignore', 'docs/draft', '--ignore', 'docs/tmp')[0], 0)
        self.assertEqual(self.cli('audit', '--ignore', 'docs/', str(root))[0], 0)
        self.assertEqual(self.cli('audit', str(root))[0], 2)

    def test_ignore_evidence(self):
        root = self.fixture({'README.md': 'document qa', 'vendor/a.py': 'x.document_qa()'})
        self.assertEqual(self.cli('audit', str(root), '--ignore', 'vendor/')[0], 2)

    def test_cli_errors_verbose_default(self):
        self.assertEqual(self.cli('audit', '/nonexistent/audit-repo')[0], 1)
        self.assertEqual(self.cli('audit', '--fail-on', 'low')[0], 1)
        root = self.fixture({'README.md': 'hello'})
        self.assertIn('README.md', self.cli('audit', str(root), '--verbose')[2])
        import os
        old = os.getcwd()
        try:
            os.chdir(root)
            self.assertEqual(self.cli('audit')[0], 0)
        finally:
            os.chdir(old)

    def test_scope(self):
        root = self.fixture({'other.md': 'document qa', '.git/a.py': 'x.document_qa()',
                             'venv/a.py': 'x.document_qa()', 'custom/pyvenv.cfg': '',
                             'custom/a.py': 'x.document_qa()', 'docs/nested/guide.md': 'document qa'})
        self.assertEqual(audit(root)[0]['severity'], 'HIGH')

    def test_invalid_source_and_io(self):
        self.assertFalse(evidence('bad syntax !'))
        root = self.fixture({'README.md': 'hello'})
        (root / 'README.md').write_bytes(b'\xff')
        self.assertEqual(self.cli('audit', str(root))[0], 1)

    def test_self_readme(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(match_phrases(prose((root / 'README.md').read_text(encoding='utf-8'))), ([], []))
        readme_repo = self.fixture({'README.md': (root / 'README.md').read_text(encoding='utf-8')})
        self.assertEqual(audit(readme_repo), [])


if __name__ == '__main__':
    unittest.main()

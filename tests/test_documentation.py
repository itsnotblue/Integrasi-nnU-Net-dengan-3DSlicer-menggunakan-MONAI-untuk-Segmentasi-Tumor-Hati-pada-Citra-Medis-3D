"""Check public documentation navigation without models or third-party packages."""

import posixpath
from pathlib import Path, PurePosixPath
import re
import sys
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_source import source_snapshot


def markdown_links(content):
    # Examples in fenced code are not rendered documentation links.
    prose = re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*(?:\n|$)", "", content)
    return re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", prose)


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = dict(source_snapshot(ROOT))
        cls.directories = {"."}
        for name in cls.files:
            cls.directories.update(str(parent) for parent in PurePosixPath(name).parents)

    def test_public_relative_links_point_to_published_paths(self):
        for name, content in self.files.items():
            if not name.endswith(".md"):
                continue
            for link in markdown_links(content.decode("utf-8-sig")):
                target = link.strip().split(' "', 1)[0].strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                relative = unquote(parsed.path)
                destination = posixpath.normpath(posixpath.join(str(PurePosixPath(name).parent), relative))
                with self.subTest(document=name, target=relative):
                    self.assertFalse(destination.startswith(("../", "/")), "Link escapes source repository")
                    self.assertIn(destination, self.files.keys() | self.directories,
                                  "Missing or excluded link target (including filename case)")

    def test_index_links_to_each_top_level_guide(self):
        index = self.files["docs/README.md"].decode("utf-8-sig")
        links = {urlsplit(link).path for link in markdown_links(index)}
        guides = {PurePosixPath(name).name for name in self.files
                  if name.startswith("docs/") and name.endswith(".md")
                  and len(PurePosixPath(name).parts) == 2 and name != "docs/README.md"}
        self.assertTrue(guides)
        self.assertLessEqual(guides, links, "A public guide is missing from the documentation index")


if __name__ == "__main__":
    unittest.main()

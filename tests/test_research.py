import unittest
from urllib.parse import parse_qs, urlparse

from story_factory import db, research
from tests.fakes import test_config


def fake_fetch(url: str) -> dict:
    query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    if "categorymembers" in url:
        if query["cmtitle"] == "Category:Broken":
            raise OSError("network down")
        return {"query": {"categorymembers": [
            {"pageid": 1, "title": "La Llorona"},
            {"pageid": 2, "title": "La Llorona (film)"},
            {"pageid": 3, "title": "List of ghosts"},
            {"pageid": 4, "title": "Kelpie"},
        ]}}
    if "extracts" in url:
        pages = {pid: {"pageid": int(pid), "title": f"Page {pid}", "fullurl": f"https://w/{pid}",
                       "extract": f"Legend text {pid}."} for pid in query["pageids"].split("|")}
        return {"query": {"pages": pages}}
    if "/search" in url:
        return {"items": [{"id": {"videoId": "v1"}, "snippet": {"title": "Night shift rules", "channelTitle": "C"}}]}
    if "/videos" in url:
        return {"items": [{"id": "v1", "statistics": {"viewCount": "123456"}}]}
    raise AssertionError(url)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config()
        self.conn = db.connect(":memory:")

    def test_wikipedia_skips_works_lists_and_bad_categories(self):
        self.cfg["research"]["wikipedia_categories"] = [
            {"category": "Category:Broken", "region": "X"},
            {"category": "Category:Ghosts", "region": "Global"},
        ]
        self.assertEqual(research.research_wikipedia(self.conn, self.cfg, fetch=fake_fetch), 2)
        ids = {r["source_id"] for r in self.conn.execute("SELECT source_id FROM inspirations")}
        self.assertEqual(ids, {"1", "4"})
        # Running again updates rows instead of duplicating them.
        research.research_wikipedia(self.conn, self.cfg, fetch=fake_fetch)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM inspirations").fetchone()[0], 2)

    def test_youtube_needs_key(self):
        self.assertEqual(research.research_youtube(self.conn, self.cfg, api_key="", fetch=fake_fetch), 0)

    def test_youtube_trends_saved_with_views(self):
        self.cfg["research"]["youtube_queries"] = ["horror"]
        self.assertEqual(research.research_youtube(self.conn, self.cfg, api_key="k", fetch=fake_fetch), 1)
        trend = db.top_trends(self.conn, 5)[0]
        self.assertEqual((trend["title"], trend["popularity"]), ("Night shift rules", 123456))


if __name__ == "__main__":
    unittest.main()

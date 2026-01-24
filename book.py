import re

guten_start_regex = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK (.+) \*\*\*"
guten_end_regex = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+ \*\*\*"


class GutenbergBook:
    def __init__(self, path):
        with open(path, "r") as f:
            text = f.read()

        sm = re.search(guten_start_regex, text)
        em = re.search(guten_end_regex, text)
        self.text = text[sm.end() : em.start()].lower()
        self.title = sm.groups()[0]

import re
import os
import warnings

import requests

from scripts.GEOSpyder.user_agents import randagent as ragent


class ChrFasterDownloader(object):

    DEFAULT_URL = "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi"
    PATTERN = r"^>NC[^:]+:(\d+)-(\d+).*(chromosome[0-9XY]+).*([NATGC]{%s})$"
    SAVE_ROOT = os.path.dirname(__file__)

    QUERY_DICT = {
        "id": "372099090",
        "db": "nuccore",
        "report": "fasta",
        "extrafeat": "null",
        "conwithfeat": "on",
        "hide-cdd": "on",
        "retmode": "html",
        "withmarkup": "on",
        "tool": "portal",
        "log$": "seqview",
        "maxdownloadsize": "1000000",
        }

    def __init__(self, width, url=None, pattern=None, params=None, save_dir='', unicode='utf-8'):
        self.url = url or self.DEFAULT_URL
        self.params = params or self.QUERY_DICT
        self.header = {'User-Agent': ragent(ends='pc')}
        self.unicode = unicode
        pattern = pattern or self.PATTERN
        pattern = pattern % (width*2)
        self.pattern = re.compile(pattern)
        self.save_path = self._init_save_path(save_dir)

        self.save_file = os.path.join(self.save_path, self.__class__.__name__.replace('Downloader', '.txt'))

        self.center = 0
        self.width = width

        self.dataset = list()

    def _init_save_path(self, save_dir):
        path = os.path.join(self.SAVE_ROOT, save_dir)
        if os.path.exists(path):
            warnings.warn('%s has been EXISTED!!!' % save_dir)
        else:
            os.mkdir(path)
        return path

    def compile_query_dict(self):
        if self.center < self.width:
            raise ValueError
        self.params["from"] = self.center - self.width
        self.params["to"] = self.center + self.width

    def send(self, url=None, params=None):
        resp = requests.get(url or self.url, params=params)
        return resp.content.decode(self.unicode)

    def href_resolute(self, text: str):
        text = text.strip()
        text = re.sub(r"[\s]", '', text)
        ret = re.search(self.pattern, text)
        assert ret, 'Match Nothing from the pattern: %s' % self.pattern
        return ret.groups()

    def write(self):
        print("Save to %s" % self.save_file)
        with open(self.save_file, 'w') as f:
            for s, e, loc, seq in self.dataset:
                content = ">{loc}from{s}to{e}\n{seq}\n".format(s=s, e=e, loc=loc, seq=seq)
                f.write(content)

    def __call__(self, center, *args, **kwargs):
        self.center = center
        self.compile_query_dict()
        # print('Connecting...')
        try:
            recv_dat = self.send(params=self.params)
        except Exception as e:
            print('CONNECT FAILED: %s' % e)
            return
        assert recv_dat, 'No data stream has been accept!'
        # print('Receiving...')
        try:
            href_dat = self.href_resolute(recv_dat)
        except AssertionError as e:
            print('RESOLUTION ERROR: %s' % e)
            return

        self.dataset.append(href_dat)

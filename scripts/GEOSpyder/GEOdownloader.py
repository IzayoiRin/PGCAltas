import re
import os
import urllib.parse as parse
import warnings
from urllib.request import urlopen

import requests
import time
from tqdm import tqdm

from scripts.GEOSpyder.user_agents import randagent as ragent


class GEODownloader(object):

    DEFAULT_URL = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi'
    PATTERN = r'<a href="(ftp://[^:]*?suppl/[^>]*?)">'
    SAVE_ROOT = os.path.dirname(__file__)

    def __init__(self, geo_id, url=None, pattern=None, save_dir='', unicode='utf-8'):
        self.geo_id = geo_id.upper()
        self.url = url or self.DEFAULT_URL
        self.params = {'acc': self.geo_id}
        self.header = {'User-Agent': ragent(ends='pc')}
        self.unicode = unicode
        self.pattern = re.compile(pattern or self.PATTERN)
        self.save_path = self._init_save_path(save_dir)

    def _init_save_path(self, save_dir):
        path = os.path.join(self.SAVE_ROOT, save_dir)
        if os.path.exists(path):
            warnings.warn('%s has been EXISTED!!!' % save_dir)
        else:
            os.mkdir(path)
        return path

    def send(self, url=None, params=None):
        resp = requests.get(url or self.url, params=params)
        return resp.content.decode(self.unicode)

    def href_resolute(self, text):
        match_ret = re.findall(self.pattern, text)
        assert match_ret, 'Match Nothing from the pattern: %s' % self.pattern
        return match_ret

    def write(self, data, calling_fn=lambda x: x, many=True, namespace=None, dtype='txt/w'):
        namespace = re.compile('suppl/{}(.*)'.format(self.geo_id or namespace))
        dtype_patten = re.compile(r'(\w*)/(\w+)')
        ftype, wtype = re.match(dtype_patten, dtype).groups()
        encoding = None if wtype == 'wb' else self.unicode

        def _write(content, _instance=self, _name=None):
            path = os.path.join(_instance.save_path, _name or _instance.geo_id + ftype)
            with open(path, wtype, encoding=encoding) as f:
                f.write(content)

        if not many:
            _write(data)
            return
        bar = tqdm(enumerate(data), total=len(data))
        for idx, url in bar:
            name = parse.unquote(re.search(namespace, url).groups()[0], encoding=self.unicode)
            bar.set_description_str('Now: {name}'.format(name=name))
            _write(calling_fn(url), _name=name)

    def __call__(self, **kwargs):
        print('Connecting...')
        try:
            recv_dat = self.send(params=self.params)
        except Exception as e:
            print('CONNECT FAILED: %s' % e)
            return
        assert recv_dat, 'No data stream has been accept!'
        print('Receiving...')
        try:
            href_dat = self.href_resolute(recv_dat)
        except AssertionError as e:
            print('RESOLUTION ERROR: %s' % e)
            return
        try:
            self.write(href_dat, **kwargs)
        except Exception as e:
            print('I/O ERROR: %s' % e)
            return
        time.sleep(0.1)
        print('Done!')


if __name__ == '__main__':
    GEODownloader('GSE120963', save_dir='GSE120963')(calling_fn=lambda x: urlopen(x).read(), dtype='/wb')

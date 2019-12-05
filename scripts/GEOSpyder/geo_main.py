import sys
import getopt
from urllib.request import urlopen

from .GEOdownloader import GEODownloader


def main():
    try:
        opt, argv = getopt.getopt(sys.argv[1:], "s:o:m:u:p:U", ['search=', 'outputdir='])
    except getopt.GetoptError:
        print('geo_main.py -s <search> -o <outputdir> -m<many> -n<namespace> -t<dtype> -u <url> -p <patten> -U <unicode>')
        sys.exit(2)
    for _ in range(2-len(argv)):
        argv.append(None)
    opt = dict(opt)
    search = opt.get('-s', argv[0])
    assert search, 'Search ID Must Exists'
    output_dir = opt.get('-o') or argv[1] or ''
    kwargs = {'url': opt.get('-u'),
              'pattern': opt.get('-p'),
              'save_dir': output_dir,
              'unicode': opt.get('-U')}
    params = {'calling_fn': lambda x: urlopen(x).read(),
              'many': opt.get('-m'),
              'namespace': opt.get('-n'),
              'dtype': opt.get('-t')}
    GEODownloader(search, kwargs)(params)


if __name__ == '__main__':
    main()

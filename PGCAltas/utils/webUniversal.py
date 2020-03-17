import uuid, base64

UUID_MAPPING = {1: uuid.uuid1, 3: uuid.uuid3, 4: uuid.uuid4, 5: uuid.uuid5}


def UUID2base64(uu: str, coden):
    uu = uu.replace('-', '')
    s = base64.b64encode(uuid.UUID(uu).bytes)
    return str(s, coden)


def base64UUID(model='en', coden='utf-8'):
    """
    s = base64UUID()()
    print(s, base64UUID('de')(s))
    """

    def encode(uumod=1, **kwargs):
        uu = str(UUID_MAPPING[uumod](**kwargs)).replace('-', '')
        return uu, UUID2base64(uu, coden)

    def decode(s: str):
        assert isinstance(s, str), 'INPUT Must be STRING'
        s = uuid.UUID(bytes=base64.b64decode(s.encode(coden)))
        return str(s).replace('-', '')

    mapping = {'en': encode, 'de': decode}
    return mapping[model]


if __name__ == '__main__':
    s = UUID2base64('176636a4611611ea927f28f10e1c42c5', 'utf-8')
    print(s)

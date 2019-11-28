import json
import numpy as np


class Node(object):

    def __init__(self, name):
        self.name = name
        self._children = list()
        self.itemstyle = {'color': None}
        self._value = None

    def set_color(self, colour):
        self.itemstyle['color'] = colour
        return self

    def set_value(self, value):
        self._value = value
        return self

    def set_children(self, *children):
        for c in children:
            self._children.append(c)
        return self

    @property
    def _json_dict(self):

        json_dict = {
            "name": self.name,
            "itemStyle": self.itemstyle,
        }

        if self._children:
            json_dict["children"] = [c._json_dict for c in self._children]
        if self._value is not None:
            json_dict['value'] = self._value

        return json_dict

    def to_json(self):
        return json.dumps(self._json_dict)

    def __str__(self):
        return "<NodeObject:%s,%s, %s>" % (self.name, self.itemstyle['color'], self._value)


class NodeArray(object):

    def __init__(self):
        self.nodes = list()
        self.root = list()

    def build_from_array(self, arr: np.ndarray, dtype=int):
        """

        :param arr: name, color, father_id, value
        :param dtype:
        :return:
        """
        assert arr.shape[1] == 4
        import time
        i = 1
        for name, color, father, value in arr:
            node = Node(str(name)).set_color(str(color))
            self.nodes.append(node)
            if father == -1:
                self.root.append(node)
            else:
                self.nodes[int(father)].set_children(node.set_value(dtype(value)))
            time.sleep(0.001)
            print('\rWorking:{obj}\tRate:{rate}%'.format(obj=node, rate=round(i / arr.shape[0] * 100, 2)), end='')
            i += 1
        print()
        return self

    def _initialize_root_nodes(self, color, name='Root'):
        return Node(name).set_color(color).set_children(*self.root)

    def to_json(self, include_root=False, **root_attr):
        if include_root:
            return self._initialize_root_nodes(**root_attr).to_json()
        return json.dumps([father._json_dict for father in self.root])

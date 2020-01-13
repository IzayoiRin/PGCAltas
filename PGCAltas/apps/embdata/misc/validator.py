class ValidatorBase(object):

    class Mata:

        models = []
        input_foos = []
        output_foos = []

        def __new__(cls, *args, **kwargs):
            inflow = None
            tempout = None
            for i in range(len(cls.models)):
                model, infoo, outfoo = cls.models[i], cls.input_foos[i], cls.output_foos[i]
                if inflow is None:
                    inflow = getattr(model, infoo)
                    tempout = getattr(model, outfoo)
                    continue


    def validate_flow(self):
        return self.Mata()

    def set_flow(self, *pipe):
        for model, infoo, outfoo in pipe:
            self.Mata.models.append(model)
            self.Mata.input_foos.append(infoo)
            self.Mata.output_foos.append(outfoo)


class Validator(ValidatorBase):

    pass


v = Validator()
v.set_flow(
    (1, lambda x: x+1, None),
    (2, None, lambda x: x),
)
print(v.validate_flow())

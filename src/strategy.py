# Defines the `Strategy` interface


class Strategy:
    name = "Interface"

    def pvalue(self) -> float:
        raise Exception('Base class should not implement')
